#!/usr/bin/python3
"""
recorder.py - Aufnahme-Modul
==============================
Video-Aufnahme mit OpenCV VideoWriter.
"""

import cv2
import numpy as np
import os
from datetime import datetime
from typing import Optional


class Recorder:
    """Verwaltet Video-Aufnahme und Screenshots."""
    
    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self._writer: Optional[cv2.VideoWriter] = None
        self._is_recording = False
        self._current_file = ""
        self._frame_count = 0
    
    def start_recording(self, frame_shape, fps: float = 30.0):
        """Startet eine neue Videoaufnahme."""
        if self._is_recording:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"recording_{timestamp}.avi")
        
        h, w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
        
        if self._writer.isOpened():
            self._is_recording = True
            self._current_file = filename
            self._frame_count = 0
            print(f"🎥 Aufnahme gestartet: {filename}")
        else:
            print("❌ VideoWriter konnte nicht geöffnet werden!")
            self._writer = None
    
    def stop_recording(self) -> str:
        """Stoppt die laufende Aufnahme. Gibt Dateiname zurück."""
        if not self._is_recording:
            return ""
        
        self._is_recording = False
        if self._writer:
            self._writer.release()
            self._writer = None
        
        print(f"✅ Aufnahme gespeichert: {self._current_file} ({self._frame_count} Frames)")
        return self._current_file
    
    def write_frame(self, frame: np.ndarray):
        """Schreibt einen Frame in die laufende Aufnahme."""
        if self._is_recording and self._writer:
            self._writer.write(frame)
            self._frame_count += 1
    
    def save_screenshot(self, frame: np.ndarray) -> str:
        """Speichert einen Screenshot. Gibt Dateiname zurück."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"screenshot_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot: {filename}")
        return filename
    
    @property
    def is_recording(self) -> bool:
        return self._is_recording
    
    @property
    def current_file(self) -> str:
        return self._current_file
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
