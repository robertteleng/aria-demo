#!/usr/bin/env python3
"""
ARIA Demo - Demo simple para gafas Meta Aria.

Uso:
    python demo.py              # Modo interactivo
    python demo.py --aria       # Conectar con gafas Aria
    python demo.py --webcam     # Usar webcam
    python demo.py --video PATH # Usar video
"""

import argparse
import sys
import time
from typing import Optional, Tuple

import numpy as np

# Componentes locales
from observer import AriaDemoObserver, MockObserver
from detector import ParallelDetector, Detection
from dashboard import Dashboard, create_gradio_app, SimpleDashboard
from audio import AudioFeedback


class AriaDemo:
    """Controlador principal de la demo."""

    def __init__(
        self,
        mode: str = "webcam",
        video_path: Optional[str] = None,
        enable_audio: bool = True,
        enable_depth: bool = True
    ):
        self.mode = mode
        self._running = False
        self._frame_count = 0
        self._fps_start = time.time()
        self._fps = 0.0

        # Inicializar observer
        print(f"[DEMO] Iniciando en modo: {mode}")
        if mode == "aria":
            try:
                self.observer = AriaDemoObserver()
            except Exception as e:
                print(f"[ERROR] No se pudo conectar con Aria: {e}")
                print("[INFO] Cambiando a modo webcam...")
                self.observer = MockObserver(source="webcam")
        elif mode == "video":
            self.observer = MockObserver(source="video", video_path=video_path)
        else:
            self.observer = MockObserver(source="webcam")

        # Inicializar detector
        self.detector = ParallelDetector(enable_depth=enable_depth)

        # Dashboard
        self.dashboard = Dashboard()

        # Audio
        self.audio = AudioFeedback() if enable_audio else None

        # Estado actual
        self._current_detections = []
        self._current_depth = None
        self._current_gaze = None

        print("[DEMO] Componentes inicializados")

    def process_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Procesa un frame y retorna outputs para Gradio."""
        rgb_frame = self.observer.get_frame("rgb")
        eye_frame = self.observer.get_frame("eye")

        if rgb_frame is None:
            # Frame vacío
            empty = np.zeros((480, 640, 3), dtype=np.uint8)
            return empty, empty, empty, "Esperando frames..."

        # Detectar
        detections, depth_map = self.detector.process(rgb_frame)
        self._current_detections = detections
        self._current_depth = depth_map

        # Gaze
        gaze_point = None
        if eye_frame is not None:
            gaze_point = self.detector.estimate_gaze(eye_frame)
            self._current_gaze = gaze_point

        # Audio
        if self.audio and detections:
            self.audio.announce(detections, depth_map)

        # FPS
        self._frame_count += 1
        if self._frame_count % 10 == 0:
            elapsed = time.time() - self._fps_start
            self._fps = self._frame_count / elapsed if elapsed > 0 else 0

        # Renderizar
        return self.dashboard.render(
            rgb_frame, depth_map, eye_frame,
            detections, gaze_point, self._fps
        )

    def scan_scene(self):
        """Escanea y anuncia todos los objetos."""
        if self.audio:
            self.audio.scan_scene(self._current_detections)

    def run_gradio(self, share: bool = False):
        """Ejecuta con interfaz Gradio."""
        app = create_gradio_app(
            process_callback=self.process_frame,
            scan_callback=self.scan_scene
        )
        app.launch(share=share)

    def run_opencv(self):
        """Ejecuta con interfaz OpenCV (fallback)."""
        simple_dash = SimpleDashboard()
        print("[DEMO] Presiona 'q' para salir, 's' para scan")

        try:
            while True:
                rgb_frame = self.observer.get_frame("rgb")
                eye_frame = self.observer.get_frame("eye")

                if rgb_frame is None:
                    time.sleep(0.1)
                    continue

                # Procesar
                detections, depth_map = self.detector.process(rgb_frame)
                gaze_point = None
                if eye_frame is not None:
                    gaze_point = self.detector.estimate_gaze(eye_frame)

                # Audio
                if self.audio:
                    self.audio.announce(detections, depth_map)

                # FPS
                self._frame_count += 1
                if self._frame_count % 10 == 0:
                    elapsed = time.time() - self._fps_start
                    self._fps = self._frame_count / elapsed

                # Mostrar
                key = simple_dash.show(
                    rgb_frame, depth_map, eye_frame,
                    detections, gaze_point, self._fps
                )

                if key == ord('q'):
                    break
                elif key == ord('s') and self.audio:
                    self.audio.scan_scene(detections)

        except KeyboardInterrupt:
            print("\n[DEMO] Interrumpido")

        finally:
            self.cleanup()

    def cleanup(self):
        """Limpia recursos."""
        print(f"[DEMO] Frames procesados: {self._frame_count}")
        print(f"[DEMO] FPS promedio: {self._fps:.1f}")
        self.observer.stop()
        if self.audio:
            self.audio.cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="ARIA Demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--aria", action="store_true", help="Usar gafas Aria")
    group.add_argument("--webcam", action="store_true", help="Usar webcam")
    group.add_argument("--video", type=str, help="Usar archivo de video")
    parser.add_argument("--no-audio", action="store_true", help="Desactivar audio")
    parser.add_argument("--no-depth", action="store_true", help="Desactivar profundidad")
    parser.add_argument("--opencv", action="store_true", help="Usar OpenCV en vez de Gradio")
    parser.add_argument("--share", action="store_true", help="Compartir Gradio públicamente")
    return parser.parse_args()


def select_mode_interactive():
    """Menú interactivo."""
    print("\n" + "=" * 50)
    print("  ARIA DEMO")
    print("=" * 50)
    print("\n  [1] Gafas Aria")
    print("  [2] Webcam")
    print("  [3] Video")
    print("  [q] Salir\n")

    while True:
        choice = input("  Opción: ").strip().lower()
        if choice == "1":
            return "aria", None
        elif choice == "2":
            return "webcam", None
        elif choice == "3":
            path = input("  Ruta del video: ").strip()
            return "video", path
        elif choice == "q":
            sys.exit(0)


def main():
    args = parse_args()

    # Determinar modo
    if args.aria:
        mode, video_path = "aria", None
    elif args.webcam:
        mode, video_path = "webcam", None
    elif args.video:
        mode, video_path = "video", args.video
    else:
        mode, video_path = select_mode_interactive()

    # Crear demo
    demo = AriaDemo(
        mode=mode,
        video_path=video_path,
        enable_audio=not args.no_audio,
        enable_depth=not args.no_depth
    )

    # Ejecutar
    try:
        if args.opencv:
            demo.run_opencv()
        else:
            demo.run_gradio(share=args.share)
    except ImportError as e:
        print(f"[WARN] Gradio no disponible: {e}")
        print("[INFO] Usando OpenCV...")
        demo.run_opencv()


if __name__ == "__main__":
    main()
