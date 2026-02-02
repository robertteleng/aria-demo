# ARIA Demo

Demo simple para gafas Meta Aria: detección de objetos + profundidad + eye tracking + feedback audio/visual.

## Estructura

```
aria-demo/
├── demo.py          # Script principal
├── observer.py      # Captura de frames (Aria/Webcam/Video)
├── detector.py      # YOLO + Depth Anything (CUDA streams)
├── dashboard.py     # Visualización OpenCV
├── audio.py         # Feedback auditivo (beeps + TTS)
└── README.md
```

## Uso

```bash
# Modo interactivo (menú)
python demo.py

# Conectar con gafas Aria
python demo.py --aria

# Usar webcam
python demo.py --webcam

# Usar video
python demo.py --video /ruta/al/video.mp4

# Opciones adicionales
python demo.py --webcam --no-audio   # Sin audio
python demo.py --aria --no-depth     # Sin profundidad
```

## Controles

- `q` - Salir
- `s` - Scan de escena (anuncia todos los objetos)

## Dashboard (Gradio)

Interfaz web accesible en `http://localhost:7860`

```
┌──────────────────┬─────────────┐
│   RGB + Boxes    │  Depth Map  │
│   + Gaze Point   │ (pseudocolor)│
├──────────────────┼─────────────┤
│   Eye Tracking   │   Status    │
│   + Detections   │   + FPS     │
└──────────────────┴─────────────┘
     [Start] [Stop] [Scan Scene]
```

## Dependencias

```bash
pip install ultralytics opencv-python torch sounddevice numpy gradio
# Para TTS en Linux:
pip install pyttsx3
```

## Características

- **Detección**: YOLO11 (descarga automática desde Ultralytics)
- **Profundidad**: Depth Anything V2 con CUDA
- **Eye Tracking**: Detección de pupila + punto de mirada
- **Audio**: Beeps direccionales + TTS
- **CUDA Streams**: YOLO + Depth ejecutándose en paralelo
