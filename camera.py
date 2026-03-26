#!/usr/bin/python3
"""
camera.py - Kamera-Handling Modul
===================================
Separater Thread für Kamera-Capture.

Grünes Bild Fix:
- appsink bekommt max-buffers=1 drop=true sync=false
  → verhindert Buffer-Stau der zu grünen/leeren Frames führt
- USB-Kamera Fallback mit minimiertem Buffer (CAP_PROP_BUFFERSIZE=1)
- Falls CSI-Kamera grün bleibt: flip_method probieren (0, 2, 4, 6)
"""

import cv2
import threading
import queue
import time
import numpy as np
from typing import Optional


def build_gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1280,
    capture_height: int = 720,
    display_width: int = 640,
    display_height: int = 480,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    """
    GStreamer-Pipeline für Jetson Nano CSI-Kamera.

    Grünes Bild tritt auf wenn:
    - Der appsink Buffer sich staut (sync=false + drop=true behebt das)
    - flip_method falsch gesetzt ist (0=keine Drehung, 2=180°, 4=90°, 6=270°)
    - nvarguscamerasrc noch nicht bereit ist (kurz warten nach Start)
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, "
        f"height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink max-buffers=1 drop=true sync=false"
    )


def _check_green_frame(frame: np.ndarray) -> bool:
    """
    Erkennt ob ein Frame rein grün ist (defekter CSI-Frame).
    Gibt True zurück wenn der Frame als grün gilt und verworfen werden soll.
    """
    if frame is None:
        return True
    # Prüfe ob der Frame fast vollständig grün ist
    # (B und R Kanal nahe 0, G Kanal dominant)
    b_mean = float(frame[:, :, 0].mean())
    g_mean = float(frame[:, :, 1].mean())
    r_mean = float(frame[:, :, 2].mean())
    # Grüner Frame: G >> B und G >> R
    if g_mean > 100 and b_mean < 30 and r_mean < 30:
        return True
    return False


class CameraThread(threading.Thread):
    """
    Kamera-Capture in einem eigenen Thread.
    Hält immer den aktuellsten Frame bereit (Queue maxsize=1).
    Verwirft grüne/defekte Frames automatisch.
    """

    def __init__(self, use_csi: bool = True, sensor_id: int = 0):
        super().__init__(daemon=True)
        self.use_csi   = use_csi
        self.sensor_id = sensor_id

        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event  = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running  = False
        self._error: Optional[str] = None

        # FPS-Tracking
        self._fps         = 0.0
        self._frame_count = 0
        self._fps_timer   = time.time()

    def run(self):
        self._cap = self._open_camera()

        if self._cap is None or not self._cap.isOpened():
            self._error = "Keine Kamera gefunden"
            return

        self._is_running = True

        # Kurz warten damit CSI-Kamera sich initialisiert
        # (verhindert grüne Frames direkt nach dem Öffnen)
        time.sleep(0.5)

        green_frame_count = 0  # Zähler für aufeinanderfolgende grüne Frames

        while not self._stop_event.is_set():
            ret, frame = self._cap.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Grüne/defekte Frames verwerfen
            if _check_green_frame(frame):
                green_frame_count += 1
                if green_frame_count > 30:
                    # Nach 30 grünen Frames: Kamera neu initialisieren
                    print("Grüne Frames erkannt – versuche Kamera neu zu öffnen...")
                    self._cap.release()
                    time.sleep(1.0)
                    self._cap = self._open_camera()
                    green_frame_count = 0
                    if self._cap is None or not self._cap.isOpened():
                        self._error = "Kamera nach Neustart nicht verfügbar"
                        break
                time.sleep(0.01)
                continue

            green_frame_count = 0  # Zurücksetzen bei gültigem Frame

            # FPS berechnen
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = now

            # Alten Frame verwerfen, neuen einreihen
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass

        if self._cap:
            self._cap.release()
        self._is_running = False

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """Versucht CSI-Kamera, fällt auf USB-Kamera zurück."""
        if self.use_csi:
            print("Versuche CSI-Kamera (GStreamer)...")
            pipeline = build_gstreamer_pipeline(
                sensor_id=self.sensor_id,
                capture_width=1280,
                capture_height=720,
                display_width=640,
                display_height=480,
                framerate=30,
                flip_method=0,
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print("CSI-Kamera geöffnet")
                return cap
            print("CSI-Kamera nicht verfügbar, versuche USB-Kamera...")

        # Fallback: USB-Kamera
        print("Versuche USB-Kamera (Index 0)...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Buffer minimieren = keine grünen/alten Frames
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            print("USB-Kamera geöffnet")
            return cap

        print("Keine Kamera gefunden")
        return None

    def get_frame(self) -> Optional[np.ndarray]:
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    def get_fps(self) -> float:
        return self._fps

    def is_running(self) -> bool:
        return self._is_running

    def get_error(self) -> Optional[str]:
        return self._error

    def stop(self):
        self._stop_event.set()
