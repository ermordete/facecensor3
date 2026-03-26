#!/bin/bash
# download_models.sh
# Lädt alle benötigten Modelle manuell herunter
# Ausführen: bash download_models.sh

mkdir -p models

echo "=== Lade DNN Face Detector (~10 MB) ==="
wget -c -O models/face_detector.caffemodel \
  "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

wget -c -O models/deploy.prototxt \
  "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"

echo "=== Lade MediaPipe Hand Landmarker (~9 MB) ==="
wget -c -O models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

echo ""
echo "=== Fertig. Prüfe Dateien: ==="
ls -lh models/
