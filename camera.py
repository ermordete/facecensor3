#!/usr/bin/python3
"""
camera.py - Kamera-Handling Modul
===================================
Nutzt jetson.utils.videoSource("csi://0") für die CSI-Kamera —
genau wie der funktionierende Jetson-Beispielcode.

Warum jetson.utils statt OpenCV/GStreamer direkt?
- jetson.utils ist die native Jetson-Bibliothek von NVIDIA
- Öffnet CSI-Kamera zuverlässig über csi://0
- Kein manuelles GStreamer-Pipeline-Basteln nötig
- Fällt automatisch auf USB (/dev/video0) zurück wenn CSI fehlt

Frames werden von CUDA-Format (jetson.utils) in numpy/BGR konvertiert
damit der Rest der App (OpenCV, BlurProcessor) normal weiterarbeitet.
"""

import threading
import queue
import time
import numpy as np
from typing import Optional


class CameraThread(threading.Thread):
    """
    Kamera-Capture in einem eigenen Thread.
    Hält immer den aktuellsten Frame bereit (Queue maxsize=1).
    Nutzt jetson.utils.videoSource für zuverlässigen CSI-Zugriff.
    """

    def __init__(self, use_csi: bool = True, sensor_id: int = 0):
        super().__init__(daemon=True)
        self.use_csi   = use_csi
        self.sensor_id = sensor_id

        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event  = threading.Event()
        self._is_running  = False
        self._error: Optional[str] = None

        # FPS-Tracking
        self._fps         = 0.0
        self._frame_count = 0
        self._fps_timer   = time.time()

    def run(self):
        """Haupt-Capture-Loop."""
        camera = self._open_camera()

        if camera is None:
            self._error = "Keine Kamera gefunden"
            return

        self._is_running = True

        while not self._stop_event.is_set():
            try:
                # Frame aufnehmen (jetson.utils Format)
                img = camera.Capture()

                if img is None:
                    time.sleep(0.01)
                    continue

                # Von jetson CUDA-Image zu numpy BGR konvertieren
                # jetson.utils liefert RGBA → wir brauchen BGR für OpenCV
                import jetson.utils as ju
                frame = ju.cudaToNumpy(img)          # RGBA numpy array
                frame = frame[:, :, :3].copy()       # Alpha-Kanal entfernen → RGB
                frame = frame[:, :, ::-1].copy()     # RGB → BGR für OpenCV

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

            except Exception as e:
                print(f"Kamera Fehler: {e}")
                time.sleep(0.05)
                continue

        self._is_running = False

    def _open_camera(self):
        """
        Öffnet Kamera mit jetson.utils.videoSource.
        CSI: csi://0  (wie im funktionierenden Beispielcode)
        USB: /dev/video0
        """
        try:
            import jetson.utils as ju

            if self.use_csi:
                print("Öffne CSI-Kamera (csi://0) ...")
                try:
                    # Genau wie im funktionierenden Jetson-Beispielcode
                    camera = ju.videoSource("csi://0")
                    # Kurz testen ob ein Frame kommt
                    test_img = camera.Capture()
                    if test_img is not None:
                        print("CSI-Kamera geöffnet")
                        return camera
                    else:
                        print("CSI-Kamera liefert keinen Frame, versuche USB...")
                except Exception as e:
                    print(f"CSI-Kamera Fehler: {e}, versuche USB...")

            # Fallback: USB-Kamera
            print("Öffne USB-Kamera (/dev/video0) ...")
            camera = ju.videoSource("/dev/video0")
            test_img = camera.Capture()
            if test_img is not None:
                print("USB-Kamera geöffnet")
                return camera

            print("Keine Kamera gefunden")
            return None

        except ImportError:
            # jetson.utils nicht verfügbar → OpenCV Fallback
            print("jetson.utils nicht verfügbar, versuche OpenCV...")
            return self._open_camera_opencv()

    def _open_camera_opencv(self):
        """
        OpenCV Fallback wenn jetson.utils nicht installiert ist.
        Gibt ein OpenCV VideoCapture-kompatibles Objekt zurück.
        """
        import cv2

        # Wrapper damit OpenCV VideoCapture wie jetson.utils aussieht
        class OpenCVCameraWrapper:
            def __init__(self, cap):
                self._cap = cap

            def Capture(self):
                ret, frame = self._cap.read()
                if not ret:
                    return None
                # Gib ein Objekt zurück das wie jetson CUDA-Image aussieht
                # aber eigentlich schon ein numpy BGR Array ist
                return _OpenCVFrameWrapper(frame)

            def release(self):
                self._cap.release()

        class _OpenCVFrameWrapper:
            """Wrapper damit OpenCV-Frame wie jetson CUDA-Image behandelt wird."""
            def __init__(self, frame):
                self._frame = frame

        # Patch: run() muss wissen ob es jetson oder OpenCV ist
        # Einfachste Lösung: direkt separaten OpenCV-Loop nutzen
        self._use_opencv_fallback = True

        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            print("OpenCV USB-Kamera geöffnet")
            return OpenCVCameraWrapper(cap)

        return None

    def run(self):
        """Haupt-Capture-Loop — unterstützt jetson.utils und OpenCV."""
        self._use_opencv_fallback = False
        camera = self._open_camera()

        if camera is None:
            self._error = "Keine Kamera gefunden"
            return

        self._is_running = True

        # Unterscheide zwischen jetson.utils und OpenCV Fallback
        use_jetson = not getattr(self, '_use_opencv_fallback', False)

        while not self._stop_event.is_set():
            try:
                img = camera.Capture()

                if img is None:
                    time.sleep(0.01)
                    continue

                if use_jetson:
                    # jetson.utils: CUDA Image → numpy BGR
                    import jetson.utils as ju
                    frame = ju.cudaToNumpy(img)      # RGBA
                    frame = frame[:, :, :3].copy()   # → RGB
                    frame = frame[:, :, ::-1].copy() # → BGR
                else:
                    # OpenCV Wrapper: Frame ist bereits BGR numpy
                    frame = img._frame

                # FPS berechnen
                self._frame_count += 1
                now = time.time()
                elapsed = now - self._fps_timer
                if elapsed >= 1.0:
                    self._fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._fps_timer = now

                # Frame einreihen
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

            except Exception as e:
                print(f"Kamera Fehler: {e}")
                time.sleep(0.05)

        self._is_running = False

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
