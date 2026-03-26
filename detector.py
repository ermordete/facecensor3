#!/usr/bin/python3
"""
detector.py – Gesichtserkennung mit zwei wählbaren Modellen
=============================================================

Modell 1: Haar Cascade (haarcascade_frontalface_default.xml)
  - Extrem leicht, keine Download-Abhängigkeit, immer verfügbar
  - Sehr schnell auf Jetson Nano (~1–2 ms pro Frame bei 300px)
  - Schwächer bei Seitenansicht / schlechter Beleuchtung
  - Gut als "Schnell"-Modus geeignet

Modell 2: OpenCV DNN ResNet-SSD (res10_300x300_ssd_iter_140000)
  - Ca. ~10 MB Caffe-Modell, einmaliger Download
  - Deutlich robuster: erkennt Gesichter in mehr Winkeln/Lagen
  - Ca. 5–15 ms pro Frame bei 300px auf Jetson Nano
  - Gut als "Präzise"-Modus geeignet

Beide Modelle teilen dieselbe Tracking-Logik (IoU + gleitendes Smoothing),
sodass der Wechsel zur Laufzeit nahtlos funktioniert.
"""

import cv2
import numpy as np
import os
import urllib.request
from typing import List, Tuple
from dataclasses import dataclass, field


# Modell-Konstanten für externe Referenz (z. B. UI)
MODEL_HAAR = "haar"
MODEL_DNN  = "dnn"
MODEL_NAMES = {
    MODEL_HAAR: "Schnell (Haar Cascade)",
    MODEL_DNN:  "Präziser (OpenCV DNN)",
}


@dataclass
class TrackedFace:
    """Tracktes Gesicht mit glättendem Positions-Puffer."""
    x: float
    y: float
    w: float
    h: float
    confidence: float
    missed_frames: int = 0
    history: List[Tuple[float, float, float, float]] = field(default_factory=list)

    def to_int_rect(self) -> Tuple[int, int, int, int]:
        return (int(self.x), int(self.y), int(self.w), int(self.h))


class FaceDetector:
    """
    Gesichtsdetektor mit wählbarem Modell und temporalem Smoothing.

    Nutzung:
        detector = FaceDetector()
        detector.set_model(MODEL_HAAR)   # oder MODEL_DNN
        faces = detector.detect(frame)
    """

    # DNN-Modell Quellen (Caffe ResNet-SSD, ~10 MB)
    _DNN_MODEL_URL  = (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
    _DNN_CONFIG_URL = (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    )

    # Haar Cascade Suchpfade (systemweit auf Ubuntu/Jetson)
    _HAAR_PATHS = [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ]

    def __init__(
        self,
        model: str = MODEL_DNN,
        confidence_threshold: float = 0.5,
        smooth_frames: int = 5,
        max_missed_frames: int = 8,
        face_padding: float = 0.15,
        model_dir: str = "models",
    ):
        self.confidence_threshold = confidence_threshold
        self.smooth_frames        = smooth_frames
        self.max_missed_frames    = max_missed_frames
        self.face_padding         = face_padding
        self.model_dir            = model_dir

        # Aktives Modell
        self._current_model: str = model

        # DNN-Objekt (lazy geladen)
        self._dnn_net  = None
        self._dnn_ok   = False

        # Haar Cascade Objekt (lazy geladen)
        self._haar     = None
        self._haar_ok  = False

        # Tracking-State
        self._tracks: List[TrackedFace] = []

        # Beide Modelle beim Start laden
        self._load_dnn(model_dir)
        self._load_haar()

        # Sicherstellen dass gewünschtes Modell verfügbar
        self._current_model = self._resolve_model(model)
        print(f"Gesichtsmodell aktiv: {MODEL_NAMES.get(self._current_model)}")

    # ── Modell laden ─────────────────────────────────────────────────────────

    def _load_dnn(self, model_dir: str):
        """
        Lädt OpenCV DNN ResNet-SSD.
        Datei-Download nur wenn noch nicht vorhanden.
        """
        os.makedirs(model_dir, exist_ok=True)
        model_path  = os.path.join(model_dir, "face_detector.caffemodel")
        config_path = os.path.join(model_dir, "deploy.prototxt")

        if os.path.exists(model_path) and os.path.exists(config_path):
            try:
                self._dnn_net = cv2.dnn.readNetFromCaffe(config_path, model_path)
                self._dnn_ok  = True
                print("DNN Face Detector geladen")
                return
            except Exception as e:
                print(f"DNN Ladefehler: {e}")

        print("Lade DNN Face Detector herunter (~10 MB) ...")
        try:
            urllib.request.urlretrieve(self._DNN_MODEL_URL,  model_path)
            urllib.request.urlretrieve(self._DNN_CONFIG_URL, config_path)
            self._dnn_net = cv2.dnn.readNetFromCaffe(config_path, model_path)
            self._dnn_ok  = True
            print("DNN Face Detector geladen (heruntergeladen)")
        except Exception as e:
            print(f"DNN Download fehlgeschlagen: {e}")

    def _load_haar(self):
        """Lädt Haar Cascade aus Systempfaden."""
        for p in self._HAAR_PATHS:
            if os.path.exists(p):
                clf = cv2.CascadeClassifier(p)
                if not clf.empty():
                    self._haar    = clf
                    self._haar_ok = True
                    print(f"Haar Cascade geladen: {p}")
                    return
        print("Haar Cascade nicht gefunden (kein kritischer Fehler)")

    def _resolve_model(self, requested: str) -> str:
        """Wählt verfügbares Modell; fällt auf Alternative zurück."""
        if requested == MODEL_DNN and self._dnn_ok:
            return MODEL_DNN
        if requested == MODEL_HAAR and self._haar_ok:
            return MODEL_HAAR
        # Fallback: was auch immer verfügbar ist
        if self._dnn_ok:
            return MODEL_DNN
        if self._haar_ok:
            return MODEL_HAAR
        return MODEL_HAAR   # gibt leere Liste zurück wenn beides fehlt

    # ── Öffentliche API ───────────────────────────────────────────────────────

    def set_model(self, model: str):
        """Wechselt das aktive Erkennungsmodell zur Laufzeit."""
        resolved = self._resolve_model(model)
        if resolved != self._current_model:
            self._current_model = resolved
            self._tracks.clear()   # Tracks verwerfen bei Modellwechsel
            print(f"Gesichtsmodell gewechselt: {MODEL_NAMES.get(resolved)}")

    def get_current_model(self) -> str:
        return self._current_model

    def is_model_available(self, model: str) -> bool:
        if model == MODEL_DNN:
            return self._dnn_ok
        if model == MODEL_HAAR:
            return self._haar_ok
        return False

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Erkennt Gesichter, gibt geglättete Bounding Boxes (x,y,w,h) zurück.

        PERF: Intern auf 300px Breite skalieren → deutlich schneller.
        Bounding Boxes werden danach zurückskaliert.
        """
        small, scale = self._shrink(frame, target_w=300)
        raw = self._detect_raw(small)

        # Zurück auf Original-Größe skalieren
        scaled = [
            (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
            for (x, y, w, h) in raw
        ]

        self._update_tracks(scaled, frame.shape)

        result = []
        for t in self._tracks:
            if t.missed_frames == 0:
                result.append(self._padded(t, frame.shape))
        return result

    def reset_tracks(self):
        self._tracks.clear()

    def set_confidence_threshold(self, value: float):
        self.confidence_threshold = max(0.1, min(1.0, value))

    # ── Interne Detektion ─────────────────────────────────────────────────────

    def _detect_raw(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self._current_model == MODEL_DNN and self._dnn_ok:
            return self._run_dnn(frame)
        if self._haar_ok:
            return self._run_haar(frame)
        return []

    def _run_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """OpenCV DNN ResNet-SSD Detektion."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False
        )
        self._dnn_net.setInput(blob)
        dets = self._dnn_net.forward()

        result = []
        for i in range(dets.shape[2]):
            conf = dets[0, 0, i, 2]
            if conf < self.confidence_threshold:
                continue
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            fw, fh = x2 - x1, y2 - y1
            if fw > 10 and fh > 10:
                result.append((x1, y1, fw, fh))
        return result

    def _run_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Haar Cascade Detektion."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = self._haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(dets) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in dets]

    # ── Tracking + Smoothing ──────────────────────────────────────────────────

    def _update_tracks(
        self,
        detections: List[Tuple[int, int, int, int]],
        frame_shape: Tuple[int, ...],
    ):
        matched_tracks = set()
        matched_dets   = set()

        for det_id, det in enumerate(detections):
            best_iou, best_tid = 0.3, -1
            for tid, t in enumerate(self._tracks):
                if tid in matched_tracks:
                    continue
                iou = _iou(det, t.to_int_rect())
                if iou > best_iou:
                    best_iou, best_tid = iou, tid

            if best_tid >= 0:
                t = self._tracks[best_tid]
                t.missed_frames = 0
                t.history.append(det)
                if len(t.history) > self.smooth_frames:
                    t.history.pop(0)
                avg = np.mean(t.history, axis=0)
                t.x, t.y, t.w, t.h = avg
                matched_tracks.add(best_tid)
                matched_dets.add(det_id)

        for tid, t in enumerate(self._tracks):
            if tid not in matched_tracks:
                t.missed_frames += 1

        for det_id, det in enumerate(detections):
            if det_id not in matched_dets:
                x, y, w, h = det
                self._tracks.append(TrackedFace(
                    x=float(x), y=float(y), w=float(w), h=float(h),
                    confidence=1.0, history=[det],
                ))

        self._tracks = [
            t for t in self._tracks
            if t.missed_frames <= self.max_missed_frames
        ]

    def _padded(
        self, t: TrackedFace, frame_shape: Tuple[int, ...]
    ) -> Tuple[int, int, int, int]:
        pad_x = int(t.w * self.face_padding)
        pad_y = int(t.h * self.face_padding)
        x = max(0, int(t.x) - pad_x)
        y = max(0, int(t.y) - pad_y)
        w = min(frame_shape[1] - x, int(t.w) + 2 * pad_x)
        h = min(frame_shape[0] - y, int(t.h) + 2 * pad_y)
        return (x, y, w, h)

    @staticmethod
    def _shrink(
        frame: np.ndarray, target_w: int
    ) -> Tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        scale = target_w / w
        return cv2.resize(frame, (target_w, int(h * scale))), scale


# ── Hilfsfunktion ─────────────────────────────────────────────────────────────

def _iou(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx);  iy1 = max(ay, by)
    ix2 = min(ax+aw, bx+bw); iy2 = min(ay+ah, by+bh)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = (ix2-ix1) * (iy2-iy1)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0
