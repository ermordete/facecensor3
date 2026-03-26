#!/usr/bin/python3
"""
effects.py – Gesichts-Blur
============================
Stark vereinfacht: nur noch Gesichts-Blur (Gaussian Blur).
Alle anderen Effekte, Presets, Emojis vollständig entfernt.
"""

import cv2
import numpy as np
from typing import List, Tuple


class BlurProcessor:
    """
    Wendet Gaussian Blur auf erkannte Gesichtsregionen an.
    Stärke 1–100 steuert die Blur-Intensität.
    """

    def __init__(self):
        self.strength: int = 50  # Default-Stärke

    def apply(
        self,
        frame: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Blur auf alle übergebenen Gesichts-Regionen anwenden."""
        for (x, y, w, h) in faces:
            frame = self._blur_face(frame, x, y, w, h)
        return frame

    def _blur_face(
        self,
        frame: np.ndarray,
        x: int, y: int, w: int, h: int,
    ) -> np.ndarray:
        """Blur auf eine einzelne Gesichtsregion."""
        fh, fw = frame.shape[:2]

        # Koordinaten auf Frame-Grenzen begrenzen
        x = max(0, x)
        y = max(0, y)
        w = min(fw - x, w)
        h = min(fh - y, h)

        if w <= 0 or h <= 0:
            return frame

        region = frame[y:y + h, x:x + w]
        if region.size == 0:
            return frame

        # Kernel-Größe aus Stärke berechnen (immer ungerade, min 5, max 101)
        k = int(5 + (self.strength / 100.0) * 96)
        k = max(5, min(101, k))
        if k % 2 == 0:
            k += 1

        blurred = cv2.GaussianBlur(region, (k, k), 0)
        frame[y:y + h, x:x + w] = blurred
        return frame

    def set_strength(self, value: int):
        """Stärke setzen (1–100)."""
        self.strength = max(1, min(100, value))
