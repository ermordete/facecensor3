#!/usr/bin/python3
"""
ui.py – Hauptfenster
======================
Korrekturen v5:
- Screenshot-Bug behoben: verwendet letzten verarbeiteten Frame aus dem
  ProcessingThread statt einen neuen Raw-Frame + parallele Detektion.
  (Vorher: Race Condition + concurrent detector access → falsches Ergebnis)
- Hautfarb-Handerkennung vollständig entfernt
- Hand-Toggle wird sauber deaktiviert wenn MediaPipe-Modell nicht verfügbar
- Hand-Toggle zeigt klaren Hinweis wenn Modell fehlt
"""

import cv2
import numpy as np
import os
import time
import threading
from collections import deque

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSlider, QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from camera import CameraThread
from detector import FaceDetector, MODEL_HAAR, MODEL_DNN, MODEL_NAMES
from hand_detector import HandDetector
from effects import BlurProcessor
from recorder import Recorder


# ═══════════════════════════════════════════════════════════════════════════════
#  FARBEN
# ═══════════════════════════════════════════════════════════════════════════════
C_BG         = "#1F2A36"
C_PANEL      = "#263545"
C_PANEL2     = "#2C3E50"
C_BUTTON     = "#E1DACA"
C_BUTTON_HOV = "#EDE9E0"
C_BUTTON_ACT = "#CAC6B6"
C_TEXT       = "#CBCCBE"
C_TEXT_DARK  = "#1F2A36"
C_TEXT_DIM   = "#7A8490"
C_ACCENT     = "#4A9EBF"
C_DANGER     = "#BF4A4A"
C_SUCCESS    = "#4A9E72"
C_BORDER     = "#33495C"
C_VIDEO_BG   = "#0D1720"

FONT = '"Noto Sans", "DejaVu Sans", "Liberation Sans", Arial, sans-serif'

# ═══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
FACE_DETECT_EVERY = 3   # Gesichtserkennung alle 3 Frames (~10x/Sek bei 30fps)
HAND_DETECT_EVERY = 4   # Handerkennung alle 4 Frames  (~7x/Sek bei 30fps)
FPS_WINDOW        = 30  # Gleitender Durchschnitt über 30 Frames

# ═══════════════════════════════════════════════════════════════════════════════
#  STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════
STYLE = f"""
QMainWindow, QWidget {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: {FONT};
    font-size: 13px;
}}
QPushButton {{
    background-color: {C_BUTTON};
    color: {C_TEXT_DARK};
    border: none;
    border-radius: 8px;
    padding: 9px 14px;
    font-family: {FONT};
    font-size: 12px;
    font-weight: 500;
    text-align: left;
}}
QPushButton:hover   {{ background-color: {C_BUTTON_HOV}; }}
QPushButton:pressed {{ background-color: {C_BUTTON_ACT}; }}
QPushButton:disabled {{
    background-color: {C_PANEL};
    color: {C_TEXT_DIM};
    border: 1px solid {C_BORDER};
}}
QLabel {{
    background: transparent;
    color: {C_TEXT};
    font-family: {FONT};
}}
QSlider::groove:horizontal {{
    height: 3px; background: {C_BORDER}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {C_BUTTON}; width: 14px; height: 14px;
    margin: -6px 0; border-radius: 7px; border: none;
}}
QSlider::sub-page:horizontal {{
    background: {C_ACCENT}; border-radius: 2px;
}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════

def _lbl(text: str) -> QLabel:
    l = QLabel(text.upper())
    l.setStyleSheet(
        f"color: {C_TEXT_DIM}; font-size: 10px; letter-spacing: 1.5px; "
        f"font-weight: 600; font-family: {FONT}; background: transparent;"
    )
    return l


def _card() -> QFrame:
    f = QFrame()
    f.setStyleSheet(
        f"QFrame {{ background-color: {C_PANEL}; border-radius: 10px; "
        f"border: 1px solid {C_BORDER}; }}"
    )
    return f


def _vline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.VLine)
    f.setFixedWidth(1)
    f.setStyleSheet(f"background: {C_BORDER}; border: none;")
    return f


def _ss(bg: str, fg: str = "#FFFFFF", hover: str = "") -> str:
    h = hover if hover else bg
    return (
        f"QPushButton {{ background-color: {bg}; color: {fg}; border: none; "
        f"border-radius: 8px; padding: 9px 14px; font-family: {FONT}; "
        f"font-size: 12px; font-weight: 600; text-align: left; }} "
        f"QPushButton:hover {{ background-color: {h}; }}"
    )


def _ss_beige() -> str:
    return (
        f"QPushButton {{ background-color: {C_BUTTON}; color: {C_TEXT_DARK}; "
        f"border: none; border-radius: 8px; padding: 9px 14px; "
        f"font-family: {FONT}; font-size: 12px; font-weight: 500; text-align: left; }} "
        f"QPushButton:hover {{ background-color: {C_BUTTON_HOV}; }} "
        f"QPushButton:pressed {{ background-color: {C_BUTTON_ACT}; }}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  VERARBEITUNGS-THREAD
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessingThread(QThread):
    """
    Frame-Verarbeitung: Detection + Blur, entkoppelt vom UI-Thread.

    FIX Screenshot-Bug:
    - Letzter fertig verarbeiteter Frame wird in _last_processed_frame
      gecacht (thread-safe via Lock).
    - Screenshot liest diesen Cache statt einen neuen Raw-Frame zu holen
      und parallel detect() aufzurufen (das war die Race Condition).

    Performance:
    - Gesicht alle FACE_DETECT_EVERY Frames
    - Hände alle HAND_DETECT_EVERY Frames (versetzt um 1 Frame Offset)
    - Zwischen Detection-Frames: letzte Bounding Boxes weiterverwenden
    """
    frame_ready = pyqtSignal(np.ndarray, int, int, float)

    def __init__(
        self,
        camera:        CameraThread,
        face_detector: FaceDetector,
        hand_detector: HandDetector,
        blur:          BlurProcessor,
        recorder:      Recorder,
    ):
        super().__init__()
        self.camera        = camera
        self.face_detector = face_detector
        self.hand_detector = hand_detector
        self.blur          = blur
        self.recorder      = recorder

        self._running      = True
        self._face_on      = True
        self._hand_on      = False

        self._fc           = 0
        self._last_faces: list = []
        self._last_hands: list = []
        self._ts: deque        = deque(maxlen=FPS_WINDOW)

        # Cache für Screenshot (thread-safe)
        self._frame_lock              = threading.Lock()
        self._last_processed_frame: np.ndarray = None

    def run(self):
        while self._running:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            self._fc += 1

            # ── Gesichtserkennung ──
            if self._face_on:
                if self._fc % FACE_DETECT_EVERY == 0:
                    self._last_faces = self.face_detector.detect(frame)
                frame = self.blur.apply(frame, self._last_faces)
                face_count = len(self._last_faces)
            else:
                self._last_faces = []
                face_count = 0

            # ── Handerkennung (versetzt um 1 Frame) ──
            if self._hand_on and self.hand_detector.is_available():
                if (self._fc + 1) % HAND_DETECT_EVERY == 0:
                    self._last_hands = self.hand_detector.detect(frame)
                frame = self.blur.apply(frame, self._last_hands)
                hand_count = len(self._last_hands)
            else:
                self._last_hands = []
                hand_count = 0

            # ── Aufnahme ──
            if self.recorder.is_recording:
                self.recorder.write_frame(frame)

            # ── Letzten verarbeiteten Frame cachen (für Screenshot) ──
            with self._frame_lock:
                self._last_processed_frame = frame.copy()

            # ── FPS (gleitender Durchschnitt) ──
            now = time.monotonic()
            self._ts.append(now)
            fps = (
                (len(self._ts) - 1) / (self._ts[-1] - self._ts[0])
                if len(self._ts) >= 2 else 0.0
            )

            self.frame_ready.emit(frame, face_count, hand_count, fps)

    def get_last_processed_frame(self) -> np.ndarray:
        """
        Gibt den letzten fertig verarbeiteten Frame zurück (thread-safe).
        Wird für Screenshot verwendet — kein paralleler detect()-Aufruf nötig.
        """
        with self._frame_lock:
            if self._last_processed_frame is not None:
                return self._last_processed_frame.copy()
            return None

    def set_face(self, enabled: bool):
        self._face_on = enabled
        if not enabled:
            self._last_faces = []

    def set_hand(self, enabled: bool):
        self._hand_on = enabled
        if not enabled:
            self._last_hands = []

    def stop(self):
        self._running = False
        self.wait()


# ═══════════════════════════════════════════════════════════════════════════════
#  HAUPTFENSTER
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceCensor Pro")
        self.setMinimumSize(960, 620)
        self.resize(1100, 700)
        self.setStyleSheet(STYLE)

        self.camera        = CameraThread(use_csi=True)
        self.face_detector = FaceDetector(model=MODEL_DNN)
        self.hand_detector = HandDetector()
        self.blur          = BlurProcessor()
        self.recorder      = Recorder()

        self._build_ui()

        self.proc = ProcessingThread(
            self.camera, self.face_detector, self.hand_detector,
            self.blur, self.recorder,
        )
        self.proc.frame_ready.connect(self._on_frame)
        self.camera.start()
        self.proc.start()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._poll_status)
        self._status_timer.start(1000)

        self._rec_blink = False
        self._rec_timer = QTimer(self)
        self._rec_timer.timeout.connect(self._blink_rec)

    # ── UI-Aufbau ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        vbox.addWidget(self._build_header())

        body = QWidget()
        hbox = QHBoxLayout(body)
        hbox.setContentsMargins(14, 14, 14, 10)
        hbox.setSpacing(12)
        hbox.addWidget(self._build_left_panel(),  0)
        hbox.addWidget(self._build_video_area(),  1)
        hbox.addWidget(self._build_right_panel(), 0)
        vbox.addWidget(body, 1)

        vbox.addWidget(self._build_statusbar())

    # ── Header ───────────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(50)
        w.setStyleSheet(
            f"background-color: {C_PANEL2}; border-bottom: 1px solid {C_BORDER};"
        )
        hbox = QHBoxLayout(w)
        hbox.setContentsMargins(20, 0, 20, 0)

        title = QLabel("FaceCensor Pro")
        title.setStyleSheet(
            f"color: {C_BUTTON}; font-size: 16px; font-weight: 700; "
            f"font-family: {FONT}; background: transparent;"
        )
        sub = QLabel("  ·  Jetson Nano Edition")
        sub.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 12px; font-family: {FONT}; background: transparent;"
        )
        hbox.addWidget(title)
        hbox.addWidget(sub)
        hbox.addStretch()

        self.rec_indicator = QLabel("● AUFNAHME")
        self.rec_indicator.setStyleSheet(
            f"color: {C_DANGER}; font-size: 11px; font-weight: 700; "
            f"font-family: {FONT}; background: transparent;"
        )
        self.rec_indicator.setVisible(False)
        hbox.addWidget(self.rec_indicator)
        return w

    # ── Linkes Panel ─────────────────────────────────────────────────────────

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(215)
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)

        # ── Zensur-Toggles ──
        censor_card = _card()
        cv = QVBoxLayout(censor_card)
        cv.setContentsMargins(14, 14, 14, 14)
        cv.setSpacing(7)
        cv.addWidget(_lbl("Zensur"))

        # Gesichtszensur
        self.face_btn = QPushButton("Gesichter  ·  aktiv")
        self.face_btn.setCheckable(True)
        self.face_btn.setChecked(True)
        self.face_btn.setStyleSheet(_ss(C_ACCENT, "#FFF", "#5BAECE"))
        self.face_btn.toggled.connect(self._on_face_toggled)
        cv.addWidget(self.face_btn)

        # Handzensur – nur aktivierbar wenn MediaPipe-Modell vorhanden
        hand_available = self.hand_detector.is_available()
        self.hand_btn = QPushButton(
            "Hände  ·  inaktiv" if hand_available
            else "Hände  ·  Modell fehlt"
        )
        self.hand_btn.setCheckable(True)
        self.hand_btn.setChecked(False)
        self.hand_btn.setEnabled(hand_available)
        self.hand_btn.setStyleSheet(
            _ss_beige() if hand_available
            else (
                f"QPushButton {{ background-color: {C_PANEL}; color: {C_TEXT_DIM}; "
                f"border: 1px solid {C_BORDER}; border-radius: 8px; padding: 9px 14px; "
                f"font-family: {FONT}; font-size: 12px; text-align: left; }}"
            )
        )
        if not hand_available:
            self.hand_btn.setToolTip(
                "models/hand_landmarker.task nicht gefunden.\n"
                "Beim ersten Start mit Internet wird es automatisch heruntergeladen."
            )
        self.hand_btn.toggled.connect(self._on_hand_toggled)
        cv.addWidget(self.hand_btn)

        vbox.addWidget(censor_card)

        # ── Blur-Stärke ──
        blur_card = _card()
        bv = QVBoxLayout(blur_card)
        bv.setContentsMargins(14, 14, 14, 16)
        bv.setSpacing(8)
        bv.addWidget(_lbl("Blur-Stärke"))

        row = QHBoxLayout()
        row.setSpacing(8)
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(1, 100)
        self.strength_slider.setValue(50)
        self.strength_slider.valueChanged.connect(self._on_strength)

        self.strength_val = QLabel("50")
        self.strength_val.setFixedWidth(26)
        self.strength_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.strength_val.setStyleSheet(
            f"color: {C_TEXT}; font-size: 12px; font-weight: 600; "
            f"font-family: {FONT}; background: transparent;"
        )
        row.addWidget(self.strength_slider)
        row.addWidget(self.strength_val)
        bv.addLayout(row)

        scale_row = QHBoxLayout()
        for t in ("Sanft", "Stark"):
            l = QLabel(t)
            l.setStyleSheet(
                f"color: {C_TEXT_DIM}; font-size: 9px; "
                f"font-family: {FONT}; background: transparent;"
            )
            scale_row.addWidget(l)
            if t == "Sanft":
                scale_row.addStretch()
        bv.addLayout(scale_row)
        vbox.addWidget(blur_card)

        # ── Gesichtsmodell ──
        model_card = _card()
        mv = QVBoxLayout(model_card)
        mv.setContentsMargins(14, 14, 14, 14)
        mv.setSpacing(7)
        mv.addWidget(_lbl("Gesichtsmodell"))

        self.model_haar_btn = QPushButton("Schnell  ·  Haar Cascade")
        self.model_haar_btn.setCheckable(True)
        self.model_haar_btn.setEnabled(
            self.face_detector.is_model_available(MODEL_HAAR)
        )
        self.model_haar_btn.setStyleSheet(_ss_beige())
        self.model_haar_btn.clicked.connect(
            lambda: self._on_model_select(MODEL_HAAR)
        )
        mv.addWidget(self.model_haar_btn)

        self.model_dnn_btn = QPushButton("Präziser  ·  OpenCV DNN")
        self.model_dnn_btn.setCheckable(True)
        self.model_dnn_btn.setEnabled(
            self.face_detector.is_model_available(MODEL_DNN)
        )
        self.model_dnn_btn.setStyleSheet(_ss_beige())
        self.model_dnn_btn.clicked.connect(
            lambda: self._on_model_select(MODEL_DNN)
        )
        mv.addWidget(self.model_dnn_btn)

        self._highlight_model_btn(self.face_detector.get_current_model())
        vbox.addWidget(model_card)
        vbox.addStretch()
        return panel

    # ── Video-Bereich ─────────────────────────────────────────────────────────

    def _build_video_area(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(8)

        self.video_label = QLabel("Kamera wird gestartet …")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(480, 360)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet(
            f"background-color: {C_VIDEO_BG}; border-radius: 10px; "
            f"border: 1px solid {C_BORDER}; color: {C_TEXT_DIM}; font-size: 13px;"
        )
        vbox.addWidget(self.video_label, 1)
        vbox.addWidget(self._build_stats_bar())
        return w

    def _build_stats_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(58)
        bar.setStyleSheet(
            f"background-color: {C_PANEL2}; border-radius: 10px; "
            f"border: 1px solid {C_BORDER};"
        )
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(16, 6, 16, 6)
        hbox.setSpacing(0)

        def stat(label_txt, default):
            cell = QWidget()
            cell.setStyleSheet("background: transparent;")
            v = QVBoxLayout(cell)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(1)
            val = QLabel(default)
            val.setAlignment(Qt.AlignCenter)
            val.setStyleSheet(
                f"color: #FFFFFF; font-size: 17px; font-weight: 700; "
                f"font-family: {FONT}; background: transparent;"
            )
            lbl = QLabel(label_txt)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                f"color: {C_TEXT_DIM}; font-size: 9px; letter-spacing: 1.2px; "
                f"font-family: {FONT}; background: transparent;"
            )
            v.addWidget(val)
            v.addWidget(lbl)
            return cell, val

        fps_w,  self.fps_lbl  = stat("BILDRATE",  "—")
        face_w, self.face_lbl = stat("GESICHTER", "0")
        hand_w, self.hand_lbl = stat("HÄNDE",     "0")

        hbox.addWidget(fps_w)
        hbox.addWidget(_vline())
        hbox.addWidget(face_w)
        hbox.addWidget(_vline())
        hbox.addWidget(hand_w)
        return bar

    # ── Rechtes Panel ─────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(200)
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)

        card = _card()
        cv = QVBoxLayout(card)
        cv.setContentsMargins(14, 14, 14, 14)
        cv.setSpacing(8)
        cv.addWidget(_lbl("Aktionen"))

        self.screenshot_btn = QPushButton("Screenshot")
        self.screenshot_btn.setStyleSheet(_ss_beige())
        self.screenshot_btn.clicked.connect(self._take_screenshot)
        cv.addWidget(self.screenshot_btn)

        self.record_btn = QPushButton("Aufnahme starten")
        self.record_btn.setStyleSheet(_ss(C_SUCCESS, "#FFF", "#5AAD80"))
        self.record_btn.clicked.connect(self._toggle_recording)
        cv.addWidget(self.record_btn)

        vbox.addWidget(card)
        vbox.addStretch()
        return panel

    # ── Statusleiste ──────────────────────────────────────────────────────────

    def _build_statusbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(28)
        bar.setStyleSheet(
            f"background-color: {C_PANEL2}; border-top: 1px solid {C_BORDER};"
        )
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(16, 0, 16, 0)

        self.status_lbl = QLabel("Bereit  ·  Kamera wird initialisiert …")
        self.status_lbl.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 11px; font-family: {FONT}; background: transparent;"
        )
        hint = QLabel("ESC = Beenden   S = Screenshot   R = Aufnahme   Leertaste = Gesichter")
        hint.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 10px; font-family: {FONT}; background: transparent;"
        )
        hbox.addWidget(self.status_lbl)
        hbox.addStretch()
        hbox.addWidget(hint)
        return bar

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    def _highlight_model_btn(self, active_model: str):
        if active_model == MODEL_HAAR:
            self.model_haar_btn.setStyleSheet(_ss(C_ACCENT, "#FFF", "#5BAECE"))
            self.model_dnn_btn.setStyleSheet(_ss_beige())
        else:
            self.model_dnn_btn.setStyleSheet(_ss(C_ACCENT, "#FFF", "#5BAECE"))
            self.model_haar_btn.setStyleSheet(_ss_beige())

    # ── Event Handler ─────────────────────────────────────────────────────────

    def _on_frame(
        self, frame: np.ndarray, face_count: int, hand_count: int, fps: float
    ):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        pixmap = QPixmap.fromImage(
            QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        ).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation
        )
        self.video_label.setPixmap(pixmap)
        self.fps_lbl.setText(f"{fps:.1f}" if fps > 0.5 else "—")
        self.face_lbl.setText(str(face_count))
        self.hand_lbl.setText(str(hand_count))

    def _on_face_toggled(self, enabled: bool):
        self.proc.set_face(enabled)
        self.face_btn.setText(
            "Gesichter  ·  aktiv" if enabled else "Gesichter  ·  inaktiv"
        )
        self.face_btn.setStyleSheet(
            _ss(C_ACCENT, "#FFF", "#5BAECE") if enabled else _ss_beige()
        )
        self.status_lbl.setText(
            "Gesichtszensur " + ("aktiviert" if enabled else "deaktiviert")
        )

    def _on_hand_toggled(self, enabled: bool):
        self.proc.set_hand(enabled)
        self.hand_btn.setText(
            "Hände  ·  aktiv" if enabled else "Hände  ·  inaktiv"
        )
        self.hand_btn.setStyleSheet(
            _ss(C_ACCENT, "#FFF", "#5BAECE") if enabled else _ss_beige()
        )
        self.status_lbl.setText(
            "Handzensur " + ("aktiviert" if enabled else "deaktiviert")
        )

    def _on_model_select(self, model: str):
        self.face_detector.set_model(model)
        self._highlight_model_btn(model)
        self.status_lbl.setText(
            f"Gesichtsmodell: {MODEL_NAMES.get(model, model)}"
        )

    def _on_strength(self, value: int):
        self.strength_val.setText(str(value))
        self.blur.set_strength(value)

    def _take_screenshot(self):
        """
        Screenshot-Fix:
        Verwendet den letzten fertig verarbeiteten Frame aus dem ProcessingThread.
        Damit wird exakt das gespeichert, was der Nutzer in der Vorschau sieht:
        - Originalbild
        - Nur erkannte Gesichter/Hände geblurrt
        - Kein paralleler detect()-Aufruf, keine Race Condition
        """
        frame = self.proc.get_last_processed_frame()
        if frame is not None:
            filename = self.recorder.save_screenshot(frame)
            self.status_lbl.setText(f"Screenshot: {os.path.basename(filename)}")
        else:
            self.status_lbl.setText("Kein Bild verfügbar – kurz warten")

    def _toggle_recording(self):
        if not self.recorder.is_recording:
            frame = self.proc.get_last_processed_frame()
            if frame is not None:
                self.recorder.start_recording(frame.shape)
                self.record_btn.setText("Aufnahme stoppen")
                self.record_btn.setStyleSheet(_ss(C_DANGER, "#FFF", "#CC5555"))
                self.rec_indicator.setVisible(True)
                self._rec_timer.start(600)
                self.status_lbl.setText(
                    f"Aufnahme läuft: {os.path.basename(self.recorder.current_file)}"
                )
        else:
            saved = self.recorder.stop_recording()
            self.record_btn.setText("Aufnahme starten")
            self.record_btn.setStyleSheet(_ss(C_SUCCESS, "#FFF", "#5AAD80"))
            self._rec_timer.stop()
            self.rec_indicator.setVisible(False)
            self.status_lbl.setText(f"Gespeichert: {os.path.basename(saved)}")

    def _blink_rec(self):
        self._rec_blink = not self._rec_blink
        self.rec_indicator.setVisible(self._rec_blink)

    def _poll_status(self):
        err = self.camera.get_error()
        if err:
            self.status_lbl.setText(f"Kamera-Fehler: {err}")
        elif self.camera.is_running():
            txt = self.status_lbl.text()
            if any(x in txt for x in ("Bereit", "initialisiert")):
                model_name = MODEL_NAMES.get(self.face_detector.get_current_model(), "")
                self.status_lbl.setText(f"Kamera aktiv  ·  {model_name}")

    # ── Tastaturkürzel ────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Escape:
            self.close()
        elif k == Qt.Key_S:
            self._take_screenshot()
        elif k == Qt.Key_R:
            self._toggle_recording()
        elif k == Qt.Key_Space:
            self.face_btn.setChecked(not self.face_btn.isChecked())
        super().keyPressEvent(event)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.recorder.is_recording:
            self.recorder.stop_recording()
        self.proc.stop()
        self.hand_detector.close()
        self.camera.stop()
        self.camera.join(timeout=2.0)
        event.accept()
