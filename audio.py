"""
Audio feedback: beeps espaciales + TTS.

KISS: Beep direccional + voz que dice qué hay y dónde.
"""

import platform
import subprocess
import threading
import time
from typing import List, Optional

import numpy as np

from detector import Detection


class AudioFeedback:
    """Feedback auditivo con beeps y TTS."""

    def __init__(self):
        self._cooldowns = {}  # {object_name: last_announce_time}
        self._cooldown_seconds = 2.0  # Mínimo entre anuncios del mismo objeto
        self._last_any_announce = 0
        self._min_between_any = 0.5  # Mínimo entre cualquier anuncio

        # Configurar TTS
        self._tts_lock = threading.Lock()
        self._platform = platform.system()

        # Intentar cargar sounddevice para beeps
        try:
            import sounddevice as sd
            self._sd = sd
            self._sample_rate = 44100
        except ImportError:
            print("[AUDIO] sounddevice no disponible, beeps desactivados")
            self._sd = None

        # Intentar cargar pyttsx3 para TTS (Linux/Windows)
        self._tts_engine = None
        if self._platform != "Darwin":  # No macOS
            try:
                import pyttsx3
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty('rate', 180)
            except Exception:
                print("[AUDIO] pyttsx3 no disponible")

        print(f"[AUDIO] Inicializado (platform={self._platform})")

    def announce(self, detections: List[Detection], depth_map=None):
        """
        Anuncia detecciones relevantes.

        Solo anuncia el objeto más cercano si pasó el cooldown.
        """
        if not detections:
            return

        now = time.time()

        # Verificar cooldown global
        if now - self._last_any_announce < self._min_between_any:
            return

        # Buscar objeto más relevante (más cercano)
        for det in detections:
            # Verificar cooldown específico del objeto
            key = f"{det.name}_{det.zone}"
            last_time = self._cooldowns.get(key, 0)

            if now - last_time >= self._cooldown_seconds:
                # Anunciar este objeto
                self._announce_detection(det)
                self._cooldowns[key] = now
                self._last_any_announce = now
                break  # Solo uno por frame

    def _announce_detection(self, det: Detection):
        """Anuncia una detección específica."""
        # Beep direccional
        self._play_beep(det.zone, det.distance)

        # TTS con mensaje
        message = self._format_message(det)
        self._speak(message)

    def _format_message(self, det: Detection) -> str:
        """Formatea mensaje para TTS."""
        # Traducir zona
        zone_es = {
            "left": "izquierda",
            "center": "centro",
            "right": "derecha"
        }.get(det.zone, det.zone)

        # Traducir distancia
        dist_es = {
            "very_close": "muy cerca",
            "close": "cerca",
            "medium": "medio",
            "far": "lejos"
        }.get(det.distance, "")

        # Traducir objetos comunes
        name_es = {
            "person": "Persona",
            "chair": "Silla",
            "table": "Mesa",
            "door": "Puerta",
            "car": "Coche",
            "bottle": "Botella",
            "cup": "Taza",
            "laptop": "Portátil",
            "cell phone": "Móvil",
            "book": "Libro",
            "backpack": "Mochila",
            "umbrella": "Paraguas",
            "handbag": "Bolso",
            "suitcase": "Maleta",
            "tv": "Televisor",
            "keyboard": "Teclado",
            "mouse": "Ratón",
        }.get(det.name, det.name.capitalize())

        if dist_es:
            return f"{name_es} a tu {zone_es}, {dist_es}"
        else:
            return f"{name_es} a tu {zone_es}"

    def _play_beep(self, zone: str, distance: str):
        """Reproduce beep direccional."""
        if self._sd is None:
            return

        try:
            # Frecuencia según distancia
            freq = {
                "very_close": 1200,
                "close": 1000,
                "medium": 800,
                "far": 600
            }.get(distance, 800)

            # Volumen según distancia
            volume = {
                "very_close": 0.8,
                "close": 0.6,
                "medium": 0.4,
                "far": 0.2
            }.get(distance, 0.5)

            # Pan estéreo según zona
            if zone == "left":
                left_vol, right_vol = volume, volume * 0.2
            elif zone == "right":
                left_vol, right_vol = volume * 0.2, volume
            else:  # center
                left_vol, right_vol = volume, volume

            # Generar beep corto (100ms)
            duration = 0.1
            t = np.linspace(0, duration, int(self._sample_rate * duration), False)
            tone = np.sin(2 * np.pi * freq * t)

            # Crear estéreo con pan
            stereo = np.column_stack([tone * left_vol, tone * right_vol])

            # Reproducir sin bloquear
            self._sd.play(stereo.astype(np.float32), self._sample_rate)

        except Exception as e:
            print(f"[AUDIO ERROR] Beep: {e}")

    def _speak(self, text: str):
        """Reproduce texto con TTS."""
        def _tts():
            with self._tts_lock:
                try:
                    if self._platform == "Darwin":
                        # macOS: usar comando 'say' (más rápido)
                        subprocess.run(
                            ["say", "-r", "200", text],
                            capture_output=True,
                            timeout=5
                        )
                    elif self._tts_engine:
                        # Linux/Windows: usar pyttsx3
                        self._tts_engine.say(text)
                        self._tts_engine.runAndWait()
                except Exception as e:
                    print(f"[AUDIO ERROR] TTS: {e}")

        # Ejecutar en hilo separado para no bloquear
        threading.Thread(target=_tts, daemon=True).start()

    def scan_scene(self, detections: List[Detection]):
        """Anuncia todos los objetos detectados (resumen de escena)."""
        if not detections:
            self._speak("No hay objetos detectados")
            return

        # Agrupar por zona
        by_zone = {"left": [], "center": [], "right": []}
        for det in detections:
            by_zone[det.zone].append(det.name)

        # Construir mensaje
        parts = []
        if by_zone["left"]:
            items = ", ".join(by_zone["left"][:3])  # Max 3 por zona
            parts.append(f"A tu izquierda: {items}")
        if by_zone["center"]:
            items = ", ".join(by_zone["center"][:3])
            parts.append(f"Al centro: {items}")
        if by_zone["right"]:
            items = ", ".join(by_zone["right"][:3])
            parts.append(f"A tu derecha: {items}")

        message = ". ".join(parts) if parts else "Escena vacía"
        self._speak(message)

    def cleanup(self):
        """Limpia recursos."""
        if self._tts_engine:
            try:
                self._tts_engine.stop()
            except:
                pass
