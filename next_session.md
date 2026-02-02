# Next Session - Testing Prompt

## Objetivo
Verificar que todos los componentes de ARIA Demo funcionan correctamente.

---

## Checklist de Testing

### 1. Server MJPEG

```bash
cd /home/robert/Projects/aria/aria-demo
source .venv/bin/activate
python main.py test_video.mp4
```

**Verificar:**
- [ ] Servidor arranca en http://localhost:5000
- [ ] Video RGB muestra detecciones con bounding boxes
- [ ] Mapa de profundidad se renderiza correctamente
- [ ] FPS estable entre 15-20
- [ ] Panel de status actualiza en tiempo real
- [ ] Objetos cercanos tienen colores rojos/naranjas
- [ ] Objetos lejanos tienen colores verdes

### 2. Audio Espacial

**Verificar:**
- [ ] Beeps suenan cuando hay objetos cercanos
- [ ] Beeps se escuchan en el lado correcto (izq/der)
- [ ] Objetos "very_close" generan beeps más fuertes
- [ ] TTS anuncia objetos peligrosos no vistos

### 3. Webcam (si disponible)

```bash
python main.py
```

**Verificar:**
- [ ] Webcam se activa correctamente
- [ ] Detecciones en tiempo real

---

## Posibles Problemas

### PortAudio not found
```bash
sudo apt-get install -y libportaudio2 portaudio19-dev
```

### CUDA out of memory
- Reiniciar el proceso
- Cerrar otras aplicaciones que usen GPU

### Video corre muy rápido
- Verificar que observer.py tiene frame rate limiting
- `self._frame_interval = 1.0 / self._target_fps`

### No hay audio
- Verificar que sounddevice está instalado
- Verificar que hay dispositivo de audio activo

---

## Comandos Útiles

```bash
# Ver uso de GPU
nvidia-smi

# Ver logs del servidor
python main.py test_video.mp4 2>&1 | tee server.log
```

---

## PRÓXIMA FEATURE: VLM + Voz

Integrar FastVLM de aria-scene para descripciones detalladas + control por voz.

### Arquitectura

```
┌─────────────────────────────────────────────────────┐
│              aria-demo (real-time ~20 FPS)          │
├─────────────────────────────────────────────────────┤
│  YOLO → Detections → Beeps (continuo)               │
│  Depth → Distance                                   │
│  Gaze → is_gazed                                    │
├─────────────────────────────────────────────────────┤
│  [Voz: "describe" / Botón Scan / Tecla 's']         │
│            ↓                                        │
│  FastVLM-0.5B → Descripción rica → TTS              │
│            (~400ms, on-demand)                      │
└─────────────────────────────────────────────────────┘
```

### Archivos a Crear/Modificar

| Archivo | Acción | Descripción |
|---------|--------|-------------|
| `scene_describer.py` | CREAR | Wrapper FastVLM (lazy load) |
| `voice_control.py` | CREAR | Whisper/Vosk para comandos de voz |
| `detector.py` | MODIFICAR | Añadir `describe_scene()` |
| `main.py` | MODIFICAR | Endpoint `/scan`, botón UI |
| `audio.py` | MODIFICAR | Método para TTS largo |

### Comandos de Voz

| Comando | Acción |
|---------|--------|
| "describe" / "scan" | Descripción VLM de la escena |
| "stop" | Pausar alertas |
| "resume" | Reanudar alertas |
| "help" | Listar comandos |

### Dependencias Adicionales

```bash
# Whisper para reconocimiento de voz
pip install openai-whisper
# o Vosk (offline, más ligero)
pip install vosk
```

### VRAM Estimado

| Modelo | VRAM |
|--------|------|
| YOLO26s | ~0.5GB |
| Depth Anything V2 | ~0.8GB |
| Meta Gaze | ~0.2GB |
| FastVLM-0.5B | ~1.2GB |
| **Total** | **~2.7GB** |

RTX 3090 (24GB) → OK

---

## Otras Mejoras Futuras

1. **Grabación**: Añadir botón para grabar sesión
2. **Configuración**: Archivo config.yaml para ajustar parámetros
3. **Aria Real**: Probar con gafas físicas cuando estén disponibles
4. **Haptic Feedback**: Integrar con dispositivos hápticos
5. **Multi-idioma**: TTS en diferentes idiomas
