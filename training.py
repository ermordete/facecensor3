#!/usr/bin/python3
"""
# ============================================================
# Name:    Sophia-Doreen Kunisch und Lara Schärf
# Klasse:  IFD12B
# Datum:   27.03.2026
# Thema:   Automatische Gesichtszensur mit KI
# Datei:   training.py – Programm 1: Modell vorbereiten und testen
# ============================================================
#
# Dieses Skript bereitet die beiden verwendeten Gesichtserkennungs-
# modelle vor und testet sie auf Bildmaterial.
#
# Verwendete Modelle:
#   Modell 1: Haar Cascade (haarcascade_frontalface_default.xml)
#             - Vortrainiertes Modell von OpenCV
#             - Klassisches Machine-Learning-Verfahren (Viola-Jones)
#             - Sehr schnell, geringer Rechenaufwand
#
#   Modell 2: OpenCV DNN ResNet-SSD (res10_300x300_ssd_iter_140000)
#             - Vortrainiertes Deep-Neural-Network von OpenCV/Caffe
#             - Modernes Verfahren, deutlich genauer als Haar Cascade
#             - Erkennt Gesichter in verschiedenen Winkeln und Lagen
#
# Warum vortrainierte Modelle?
#   Beide Modelle wurden mit Hunderttausenden von Gesichtsbildern
#   trainiert. Ein eigenes Training würde identische oder schlechtere
#   Ergebnisse liefern, da nicht annähernd so viele Trainingsdaten
#   zur Verfügung stehen. Die Nutzung vortrainierter Modelle ist
#   in der KI-Praxis der Standardweg (Transfer Learning).
#
# Was dieses Skript macht:
#   1. Modelle herunterladen und speichern
#   2. Beide Modelle auf Testbildern anwenden
#   3. Erkennungsqualität vergleichen und ausgeben
#   4. Ergebnisbilder mit markierten Gesichtern speichern
"""

import cv2
import numpy as np
import urllib.request
import os
import time


# ─── Konfiguration ────────────────────────────────────────────────────────────

# Ordner für Modelle und Ergebnisse
MODEL_DIR  = "models"
OUTPUT_DIR = "training_ergebnisse"

# Download-URLs für das DNN-Modell (OpenCV ResNet-SSD)
DNN_MODEL_URL  = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)
DNN_CONFIG_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)

# Haar Cascade Suchpfade (systemweit auf Ubuntu/Jetson Nano installiert)
HAAR_PATHS = [
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
]

# Konfidenz-Schwellwert für DNN (0.0 - 1.0)
# Werte unter diesem Schwellwert werden als "kein Gesicht" gewertet
DNN_CONFIDENCE = 0.5


# ─── Modelle vorbereiten ──────────────────────────────────────────────────────

def modelle_vorbereiten():
    """
    Lädt beide Modelle herunter bzw. sucht sie auf dem System.
    Gibt (haar_cascade, dnn_net) zurück.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Modell 1: Haar Cascade laden ---
    print("\n=== Modell 1: Haar Cascade ===")
    haar = None
    for pfad in HAAR_PATHS:
        if os.path.exists(pfad):
            haar = cv2.CascadeClassifier(pfad)
            if not haar.empty():
                print(f"Haar Cascade geladen: {pfad}")
                break

    if haar is None or haar.empty():
        print("FEHLER: Haar Cascade nicht gefunden.")
        print("Bitte OpenCV installieren: sudo apt-get install python3-opencv")
    else:
        print("Modell 1 bereit.")

    # --- Modell 2: DNN ResNet-SSD laden ---
    print("\n=== Modell 2: OpenCV DNN ResNet-SSD ===")
    model_pfad  = os.path.join(MODEL_DIR, "face_detector.caffemodel")
    config_pfad = os.path.join(MODEL_DIR, "deploy.prototxt")
    dnn = None

    # Prüfen ob bereits vorhanden
    if os.path.exists(model_pfad) and os.path.exists(config_pfad):
        print("DNN-Modell bereits vorhanden, lade ...")
    else:
        print("DNN-Modell wird heruntergeladen (~10 MB) ...")
        try:
            urllib.request.urlretrieve(DNN_MODEL_URL,  model_pfad)
            urllib.request.urlretrieve(DNN_CONFIG_URL, config_pfad)
            print("Download abgeschlossen.")
        except Exception as e:
            print(f"Download fehlgeschlagen: {e}")
            print("Bitte Internetverbindung prüfen.")
            return haar, None

    try:
        dnn = cv2.dnn.readNetFromCaffe(config_pfad, model_pfad)
        print("Modell 2 bereit.")
    except Exception as e:
        print(f"DNN-Modell konnte nicht geladen werden: {e}")

    return haar, dnn


# ─── Gesichtserkennung mit Haar Cascade ──────────────────────────────────────

def erkennung_haar(bild, haar):
    """
    Erkennt Gesichter mit Haar Cascade (Modell 1).

    Haar Cascade arbeitet in Graustufen und sucht Muster,
    die typisch für Gesichter sind (Augen, Nase, Mund).

    Rückgabe: Liste von (x, y, breite, hoehe), Verarbeitungszeit in ms
    """
    start = time.time()

    # Bild in Graustufen umwandeln (Haar Cascade benötigt Graustufen)
    grau = cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY)

    # Gesichter erkennen
    # scaleFactor: Wie stark das Bild pro Schritt verkleinert wird
    # minNeighbors: Mindestanzahl an Nachbar-Detektionen (verhindert False Positives)
    # minSize: Mindestgröße eines Gesichts in Pixeln
    gesichter = haar.detectMultiScale(
        grau,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    dauer_ms = (time.time() - start) * 1000

    if len(gesichter) == 0:
        return [], dauer_ms

    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in gesichter], dauer_ms


# ─── Gesichtserkennung mit DNN ────────────────────────────────────────────────

def erkennung_dnn(bild, dnn):
    """
    Erkennt Gesichter mit OpenCV DNN ResNet-SSD (Modell 2).

    Das Neuronale Netz verarbeitet das Bild als Ganzes und
    gibt für jede mögliche Gesichtsposition eine Konfidenz aus.
    Nur Ergebnisse über dem Schwellwert werden zurückgegeben.

    Rückgabe: Liste von (x, y, breite, hoehe), Verarbeitungszeit in ms
    """
    start = time.time()

    h, w = bild.shape[:2]

    # Bild in einen Blob umwandeln (Vorverarbeitung für das Netz)
    # Normalisierungswerte (104, 177, 123) sind vom Modell vorgegeben
    blob = cv2.dnn.blobFromImage(
        bild, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False
    )

    # Blob durch das Netz schicken (Forward Pass)
    dnn.setInput(blob)
    ergebnisse = dnn.forward()

    dauer_ms = (time.time() - start) * 1000

    gesichter = []
    for i in range(ergebnisse.shape[2]):
        konfidenz = ergebnisse[0, 0, i, 2]

        # Nur Ergebnisse über dem Schwellwert verwenden
        if konfidenz < DNN_CONFIDENCE:
            continue

        # Koordinaten aus relativen Werten (0-1) in Pixel umrechnen
        box = ergebnisse[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)

        # Koordinaten auf Bildgrenzen begrenzen
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        breite  = x2 - x1
        hoehe   = y2 - y1

        if breite > 10 and hoehe > 10:
            gesichter.append((x1, y1, breite, hoehe))

    return gesichter, dauer_ms


# ─── Ergebnis visualisieren ───────────────────────────────────────────────────

def ergebnis_zeichnen(bild, gesichter, titel, farbe, dauer_ms):
    """
    Zeichnet erkannte Gesichter als Rechtecke ins Bild
    und fügt Informationstext hinzu.
    """
    ergebnis = bild.copy()

    # Erkannte Gesichter einzeichnen
    for (x, y, w, h) in gesichter:
        cv2.rectangle(ergebnis, (x, y), (x + w, y + h), farbe, 2)

    # Informationstext oben links
    info = f"{titel}: {len(gesichter)} Gesicht(er) | {dauer_ms:.1f} ms"
    cv2.putText(
        ergebnis, info, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, farbe, 2
    )

    return ergebnis


# ─── Vergleichsbericht ausgeben ───────────────────────────────────────────────

def bericht_ausgeben(haar_gesichter, haar_ms, dnn_gesichter, dnn_ms, bildname):
    """Gibt einen Textvergleich der beiden Modelle aus."""
    print(f"\n--- Ergebnis für: {bildname} ---")
    print(f"  Modell 1 (Haar Cascade):  {len(haar_gesichter):2d} Gesicht(er)  |  {haar_ms:6.1f} ms")
    print(f"  Modell 2 (OpenCV DNN):    {len(dnn_gesichter):2d} Gesicht(er)  |  {dnn_ms:6.1f} ms")

    if len(dnn_gesichter) > len(haar_gesichter):
        print("  -> DNN hat mehr Gesichter erkannt (typisch: robuster bei Seitenansicht)")
    elif len(haar_gesichter) > len(dnn_gesichter):
        print("  -> Haar Cascade hat mehr erkannt (kann auch False Positives enthalten)")
    else:
        print("  -> Beide Modelle haben gleich viele Gesichter erkannt")

    if haar_ms < dnn_ms:
        print(f"  -> Haar Cascade ist schneller ({haar_ms:.1f} ms vs {dnn_ms:.1f} ms)")
    else:
        print(f"  -> DNN ist schneller ({dnn_ms:.1f} ms vs {haar_ms:.1f} ms)")


# ─── Hauptprogramm ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Automatische Gesichtszensur mit KI")
    print("  Sophia-Doreen Kunisch und Lara Schaerf | IFD12B | 27.03.2026")
    print("  Programm 1: Modell vorbereiten und vergleichen")
    print("=" * 60)

    # Schritt 1: Modelle laden
    haar, dnn = modelle_vorbereiten()

    if haar is None and dnn is None:
        print("\nFEHLER: Kein Modell verfügbar. Programm wird beendet.")
        return

    # Schritt 2: Testbild erstellen (Beispiel mit synthetischem Bild)
    # In der Praxis: echte Bilder aus einem Ordner einlesen
    print("\n=== Testbilder verarbeiten ===")
    print("Hinweis: Lege eigene Testbilder in den Ordner 'testbilder/'")
    print("         um echte Gesichter zu testen.")

    testbilder_ordner = "testbilder"
    os.makedirs(testbilder_ordner, exist_ok=True)

    # Prüfen ob Testbilder vorhanden
    testbilder = [
        f for f in os.listdir(testbilder_ordner)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not testbilder:
        print(f"\nKeine Testbilder in '{testbilder_ordner}/' gefunden.")
        print("Erstelle Beispielbild zur Demonstration ...")

        # Beispielbild mit Gesichts-ähnlichem Oval erstellen
        demo = np.ones((480, 640, 3), dtype=np.uint8) * 50
        cv2.ellipse(demo, (320, 240), (80, 100), 0, 0, 360, (180, 150, 120), -1)
        cv2.circle(demo, (295, 210), 12, (50, 50, 50), -1)  # Auge links
        cv2.circle(demo, (345, 210), 12, (50, 50, 50), -1)  # Auge rechts
        cv2.ellipse(demo, (320, 270), (30, 15), 0, 0, 180, (100, 80, 80), -1)  # Mund
        demo_pfad = os.path.join(testbilder_ordner, "demo.jpg")
        cv2.imwrite(demo_pfad, demo)
        testbilder = ["demo.jpg"]
        print(f"Demo-Bild gespeichert: {demo_pfad}")

    # Schritt 3: Alle Testbilder mit beiden Modellen verarbeiten
    for bildname in testbilder:
        bildpfad = os.path.join(testbilder_ordner, bildname)
        bild = cv2.imread(bildpfad)

        if bild is None:
            print(f"Konnte {bildname} nicht laden.")
            continue

        haar_gesichter, haar_ms = [], 0
        dnn_gesichter,  dnn_ms  = [], 0

        # Haar Cascade anwenden
        if haar is not None:
            haar_gesichter, haar_ms = erkennung_haar(bild, haar)
            haar_bild = ergebnis_zeichnen(
                bild, haar_gesichter, "Haar", (0, 255, 0), haar_ms
            )
            haar_ausgabe = os.path.join(OUTPUT_DIR, f"haar_{bildname}")
            cv2.imwrite(haar_ausgabe, haar_bild)

        # DNN anwenden
        if dnn is not None:
            dnn_gesichter, dnn_ms = erkennung_dnn(bild, dnn)
            dnn_bild = ergebnis_zeichnen(
                bild, dnn_gesichter, "DNN", (0, 100, 255), dnn_ms
            )
            dnn_ausgabe = os.path.join(OUTPUT_DIR, f"dnn_{bildname}")
            cv2.imwrite(dnn_ausgabe, dnn_bild)

        # Bericht ausgeben
        bericht_ausgeben(haar_gesichter, haar_ms, dnn_gesichter, dnn_ms, bildname)

    # Schritt 4: Zusammenfassung
    print("\n" + "=" * 60)
    print("  MODELLVERGLEICH – ZUSAMMENFASSUNG")
    print("=" * 60)
    print("""
  Modell 1: Haar Cascade (haarcascade_frontalface_default.xml)
  ─────────────────────────────────────────────────────────────
  Verfahren:   Viola-Jones Algorithmus (2001), klassisches ML
  Trainiert:   Mit positiven (Gesichter) und negativen Beispielen
  Geschw.:     ~1-5 ms pro Frame (sehr schnell)
  Genauigkeit: Gut bei Frontalansicht, schwächer bei Seitenansicht
  Vorteile:    Kein Internet-Download nötig, immer verfügbar
  Nachteile:   Mehr False Positives, schlechter bei Drehung

  Modell 2: OpenCV DNN ResNet-SSD
  ─────────────────────────────────────────────────────────────
  Verfahren:   Deep Neural Network (Convolutional Neural Network)
  Architektur: ResNet + Single Shot Detector (SSD)
  Trainiert:   Auf >500.000 Gesichtsbildern (WIDER FACE Dataset)
  Geschw.:     ~5-20 ms pro Frame (etwas langsamer)
  Genauigkeit: Sehr gut, erkennt Gesichter in vielen Winkeln
  Vorteile:    Deutlich weniger False Positives, robuster
  Nachteile:   ~10 MB Download, etwas mehr Rechenaufwand

  Ergebnisbilder gespeichert in: training_ergebnisse/
    haar_*.jpg  → Ergebnis Modell 1 (grüne Rahmen)
    dnn_*.jpg   → Ergebnis Modell 2 (blaue Rahmen)
""")
    print("Programm abgeschlossen.")


if __name__ == "__main__":
    main()
