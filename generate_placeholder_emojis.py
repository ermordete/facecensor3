#!/usr/bin/python3
"""
generate_placeholder_emojis.py
================================
Erstellt einfache Platzhalter-Emoji-Bilder für den Emoji-Overlay-Effekt.
Wird einmalig ausgeführt wenn keine echten PNG-Emojis vorhanden sind.

Ausführen: python3 generate_placeholder_emojis.py
"""

import cv2
import numpy as np
import os

os.makedirs("assets", exist_ok=True)

EMOJIS = {
    "sunglasses": {"color": (50,  50,  200), "label": "SG"},
    "laugh":      {"color": (50,  200, 200), "label": "HA"},
    "ghost":      {"color": (200, 200, 200), "label": "GH"},
    "robot":      {"color": (100, 100, 180), "label": "RB"},
    "cat":        {"color": (100, 180, 255), "label": "CA"},
    "fire":       {"color": (50,  100, 255), "label": "FI"},
    "heart":      {"color": (50,  50,  255), "label": "HT"},
    "star":       {"color": (50,  200, 255), "label": "ST"},
}

SIZE = 128

for key, info in EMOJIS.items():
    img = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    
    # Hintergrund-Kreis
    color_bgra = (*info["color"], 255)
    cv2.circle(img, (SIZE // 2, SIZE // 2), SIZE // 2 - 4, color_bgra, -1)
    
    # Rand
    cv2.circle(img, (SIZE // 2, SIZE // 2), SIZE // 2 - 4, (255, 255, 255, 180), 3)
    
    # Text-Label
    cv2.putText(
        img, info["label"],
        (SIZE // 2 - 18, SIZE // 2 + 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255, 255), 2, cv2.LINE_AA
    )
    
    path = f"assets/emoji_{key}.png"
    cv2.imwrite(path, img)
    print(f"✅ {path}")

print("\n🎉 Platzhalter-Emojis erstellt!")
print("Ersetze die PNG-Dateien in assets/ durch echte Emoji-PNGs für bessere Optik.")
