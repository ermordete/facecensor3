#!/usr/bin/python3
"""
FaceCensor Pro - Jetson Nano Edition
=====================================
Professionelles Face-Anonymisierungstool für Content Creator.
Einstiegspunkt der Anwendung.
"""

import sys
import os

# Stelle sicher, dass das Verzeichnis im Pfad ist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui import MainWindow


def main():
    # High-DPI Support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName("FaceCensor Pro")
    app.setOrganizationName("CreatorTools")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
