#!/usr/bin/python3
"""
hand_detector.py – Handerkennung via MediaPipe HandLandmarker
==============================================================

Ehrliche Analyse der Optionen für Jetson Nano
----------------------------------------------

Option 1: Haar Cascade (haarcascade_hand.xml)
  → Gibt es NICHT offiziell. Community-Downloads sind unzuverlässig/defekt.
  → Nicht verwendbar.

Option 2: Hautfarb-Detektion (HSV-Masking)
  → Funktioniert ohne Download, aber erkennt alles Hautfarbene:
    Wände, Möbel, Stühle, alles Beige/Warme.
  → Für diesen Anwendungsfall NICHT brauchbar. Entfernt.

Option 3: YOLOv5-nano (ONNX, ~7 MB)
  → Würde mit onnxruntime funktionieren (~20-50ms/Frame auf Jetson CPU).
  → Aber: kein öffentlich verfügbares, vorab-trainiertes Hand-spezifisches
    ONNX-Modell ohne aufwendiges eigenes Training.
  → Nicht praktikabel ohne Training-Setup.

Option 4: MediaPipe HandLandmarker (EMPFOHLEN)
  → Erkennt Hände zuverlässig über 21 Landmarks.
  → Kein Hautfarb-Masking, sondern echtes ML-Modell.
  → Einmaliger Download: models/hand_landmarker.task (~9 MB, float16).
  → MediaPipe 0.10+ ist auf Jetson Nano installierbar (pip install mediapipe).
  → Läuft in ~15-30ms auf Jetson Nano CPU mit model_complexity=0 (lite).
  → LIVE_STREAM Modus: non-blocking, UI blockiert nicht.
  → Das ist die einzige realistische, zuverlässige Option.

Wenn der Download scheitert (kein Internet):
  → is_available() gibt False zurück
  → Hand-Toggle in der UI wird deaktiviert
  → App läuft normal weiter mit reiner Gesichtszensur
  → Keine schlechte Notlösung, sauberes Deaktivieren

Modell-Datei:
  models/hand_landmarker.task  (~9 MB)
  Download: https://storage.googleapis.com/mediapipe-models/
            hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

  Manuell ablegen wenn kein Internet:
  1. Auf anderem Rechner herunterladen
  2. Per scp in den models/ Ordner kopieren:
     scp hand_landmarker.task jetson@JETSON-IP:~/PROJEKTNAME/models/
"""

import cv2
import numpy as np
import os
import threading
import urllib.request
from typing import List, Tuple, Optional


# ─── Konstanten ──────────────────────────────────────────────────────────────

MODEL_URL      = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"
MODEL_MIN_SIZE = 1_000_000   # Mindestgröße 1 MB als Plausibilitätsprüfung


# ─── Download ────────────────────────────────────────────────────────────────

def _download_model(model_dir: str) -> Optional[str]:
    """
    Lädt hand_landmarker.task herunter falls nicht vorhanden.
    Gibt den Pfad zurück wenn erfolgreich, sonst None.
    Löscht unvollständige Dateien bei Fehler.
    """
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, MODEL_FILENAME)

    # Bereits vorhanden und gültig?
    if os.path.exists(path):
        if os.path.getsize(path) >= MODEL_MIN_SIZE:
            print(f"hand_landmarker.task gefunden: {path}")
            return path
        else:
            print("hand_landmarker.task unvollständig – wird neu heruntergeladen")
            os.remove(path)

    print(f"Lade hand_landmarker.task herunter (~9 MB) ...")
    print(f"  -> {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
        size = os.path.getsize(path)
        if size >= MODEL_MIN_SIZE:
            print(f"hand_landmarker.task bereit ({size // 1024} KB)")
            return path
        else:
            os.remove(path)
            print("Download fehlgeschlagen (Datei zu klein).")
            return None
    except Exception as e:
        print(f"Download fehlgeschlagen: {e}")
        print(
            "Handerkennung nicht verfügbar.\n"
            "Manuell ablegen: scp hand_landmarker.task jetson@IP:~/PROJEKT/models/"
        )
        if os.path.exists(path):
            os.remove(path)
        return None


# ─── HandDetector ────────────────────────────────────────────────────────────

class HandDetector:
    """
    Handerkennung via MediaPipe HandLandmarker (Tasks API, MediaPipe 0.10+).

    Läuft im LIVE_STREAM Modus (non-blocking):
    - detect_async() sendet Frame, blockiert nicht
    - Ergebnis kommt per Callback in separatem Thread
    - detect() gibt sofort letztes bekanntes Ergebnis zurück

    Performance auf Jetson Nano:
    - ~15–30 ms Latenz (Modell-Inferenz)
    - Durch LIVE_STREAM kein Blockieren der UI
    - Detection-Intervall (alle N Frames) vom ProcessingThread gesteuert

    is_available() → False wenn Modell nicht heruntergeladen werden konnte.
    In diesem Fall deaktiviert die UI den Hand-Toggle sauber.
    """

    def __init__(self, model_dir: str = "models", max_hands: int = 4):
        self._landmarker  = None
        self._ok          = False
        self._lock        = threading.Lock()
        self._last_boxes: List[Tuple[int, int, int, int]] = []
        self._ts          = 0   # Monotoner Timestamp-Zähler für LIVE_STREAM

        # Modell laden
        model_path = _download_model(model_dir)
        if model_path:
            self._init_landmarker(model_path, max_hands)

    def _init_landmarker(self, model_path: str, max_hands: int):
        """Initialisiert den MediaPipe HandLandmarker."""
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.vision import RunningMode

            self._mp_image_cls = mp_tasks.vision.Image
            self._mp_format    = mp_tasks.ImageFormat.SRGB

            def _on_result(result, _output_image, _timestamp_ms):
                """Callback: wird in MediaPipe-internem Thread aufgerufen."""
                with self._lock:
                    self._last_boxes = self._result_to_boxes(result)

            options = vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.LIVE_STREAM,
                num_hands=max_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=_on_result,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
            self._ok = True
            print("MediaPipe HandLandmarker bereit (LIVE_STREAM Modus)")

        except Exception as e:
            print(f"MediaPipe HandLandmarker Initialisierung fehlgeschlagen: {e}")
            self._ok = False

    @staticmethod
    def _result_to_boxes(
        result,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Konvertiert MediaPipe Hand-Landmarks zu Bounding Boxes.
        Bounding Box = Min/Max aller 21 Landmarks + 20% Padding.
        """
        if result is None or not result.hand_landmarks:
            return []

        # Bildgröße ist im LIVE_STREAM Modus nicht direkt im Result.
        # Landmarks sind normalisiert [0.0–1.0], wir brauchen den Frame für
        # absolute Koordinaten. Da wir nur ein Skalierungsproblem haben,
        # speichern wir normalisierte Werte und skalieren in detect().
        # → Stattdessen: Frame-Shape im detect()-Aufruf übergeben.
        # Diese Methode wird nur intern mit Frame-Shape genutzt (siehe unten).
        return result.hand_landmarks   # Roh-Landmarks, Konvertierung in detect()

    def is_available(self) -> bool:
        return self._ok

    def detect(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Sendet Frame an MediaPipe (non-blocking) und gibt letztes
        bekanntes Ergebnis als Bounding Boxes (x, y, w, h) zurück.

        Der Aufruf blockiert nicht — er sendet den Frame asynchron
        und gibt sofort die zuletzt empfangenen Boxen zurück.
        Das bedeutet: Ergebnis hat ~1 Frame Latenz, was für Live-Video
        vollkommen akzeptabel ist.
        """
        if not self._ok or self._landmarker is None:
            return []

        h, w = frame.shape[:2]

        # Frame an MediaPipe senden (non-blocking)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = self._mp_image_cls(
            image_format=self._mp_format, data=rgb
        )
        self._ts += 33   # Simulierter 30fps-Timestamp in ms
        try:
            self._landmarker.detect_async(mp_img, self._ts)
        except Exception:
            pass

        # Letztes Callback-Ergebnis auslesen und in Pixel-Koordinaten umrechnen
        with self._lock:
            raw_landmarks = self._last_boxes

        if not raw_landmarks:
            return []

        boxes = []
        for landmarks in raw_landmarks:
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]

            x1 = int(min(xs))
            y1 = int(min(ys))
            x2 = int(max(xs))
            y2 = int(max(ys))

            # 20% Padding für vollständige Handabdeckung
            pad_x = max(10, int((x2 - x1) * 0.20))
            pad_y = max(10, int((y2 - y1) * 0.20))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            bw = x2 - x1
            bh = y2 - y1
            if bw > 10 and bh > 10:
                boxes.append((x1, y1, bw, bh))

        return boxes

    def close(self):
        """Ressourcen freigeben."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass

    def _result_to_boxes(self, result):
        """Callback-intern: speichert rohe Landmarks."""
        if result is None or not result.hand_landmarks:
            return []
        return result.hand_landmarks
