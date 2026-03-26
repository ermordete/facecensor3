#!/usr/bin/python3
"""
camera.py - Kamera-Handling Modul
===================================
Separater Thread für Kamera-Capture, damit UI und Video-Processing
sich nicht gegenseitig blockieren.

Performance-Optimierungen:
- Dedizierter Capture-Thread (kein Blockieren des UI-Threads)
- Queue mit maximaler Größe 1 → immer neuester Frame, keine Latenz
- Skalierung für Face Detection getrennt vom Display-Frame
- GStreamer-Pipeline für Jetson CSI-Kamera
"""

import cv2
import threading
import queue
import time
import numpy as np
from typing import Optional, Tuple


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
    Baut die GStreamer-Pipeline für Jetson Nano CSI-Kamera.
    
    Warum GStreamer?
    - Nutzt Jetson-Hardware-Decoder (nvarguscamerasrc) direkt
    - Wesentlich geringere CPU-Last als software-seitige Dekodierung
    - Höherer Durchsatz bei gleicher Qualität
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, "
        f"height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )


class CameraThread(threading.Thread):
    """
    Kamera-Capture in einem eigenen Thread.
    
    Entkoppelt Kamera-I/O vollständig vom Verarbeitungs- und UI-Thread.
    Hält immer den aktuellsten Frame bereit (Queue maxsize=1).
    """
    
    def __init__(self, use_csi: bool = True, sensor_id: int = 0):
        super().__init__(daemon=True)  # Daemon: stirbt mit dem Hauptprozess
        self.use_csi = use_csi
        self.sensor_id = sensor_id
        
        # Queue mit maxsize=1: alten Frame verwerfen, wenn kein Abnehmer da
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        
        self._stop_event = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._error: Optional[str] = None
        
        # FPS-Tracking
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()
    
    def run(self):
        """Haupt-Capture-Loop im separaten Thread."""
        self._cap = self._open_camera()
        
        if self._cap is None or not self._cap.isOpened():
            self._error = "Keine Kamera gefunden"
            return
        
        self._is_running = True
        
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            
            if not ret or frame is None:
                # Kurz warten und retry
                time.sleep(0.01)
                continue
            
            # FPS berechnen
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_timer = now
            
            # Alten Frame verwerfen falls Queue voll (non-blocking)
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Neuen Frame einreihen (non-blocking)
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass
        
        # Cleanup
        if self._cap:
            self._cap.release()
        self._is_running = False
    
    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """Versucht CSI-Kamera, fällt auf USB-Kamera zurück."""
        if self.use_csi:
            print("📹 Versuche CSI-Kamera (GStreamer)...")
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
                print("✅ CSI-Kamera geöffnet!")
                return cap
            else:
                print("⚠️  CSI-Kamera nicht verfügbar, versuche USB-Kamera...")
        
        # Fallback: USB-Kamera
        print("📹 Versuche USB-Kamera (Index 0)...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Buffer minimieren für geringste Latenz
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            print("✅ USB-Kamera geöffnet!")
            return cap
        
        print("❌ Keine Kamera gefunden!")
        return None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Gibt den aktuellsten Frame zurück (non-blocking).
        Gibt None zurück wenn kein neuer Frame verfügbar.
        """
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
        """Stoppt den Capture-Thread sauber."""
        self._stop_event.set()
