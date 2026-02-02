# Next Session - Testing Prompt

## Objetivo
Verificar que todos los componentes de ARIA Demo funcionan correctamente.

---

## Checklist de Testing

### 1. Server MJPEG (Principal)

```bash
cd /home/robert/Projects/aria/aria-demo
source .venv/bin/activate
python server.py test_video.mp4
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

### 3. Demo con Gradio

```bash
python demo.py --video test_video.mp4
```

**Verificar:**
- [ ] Gradio abre en localhost:7860
- [ ] Botones Start/Stop funcionan
- [ ] Botón "Scan Scene" anuncia objetos
- [ ] Todos los paneles muestran contenido

### 4. Demo con OpenCV

```bash
python demo.py --video test_video.mp4 --opencv
```

**Verificar:**
- [ ] Ventana OpenCV se abre
- [ ] Tecla 'q' cierra la aplicación
- [ ] Tecla 's' hace scan de escena

### 5. Webcam (si disponible)

```bash
python demo.py --webcam
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
python server.py test_video.mp4 2>&1 | tee server.log

# Probar solo detección (sin audio)
python demo.py --video test_video.mp4 --no-audio

# Probar sin profundidad (más rápido)
python demo.py --video test_video.mp4 --no-depth
```

---

## Mejoras Futuras (Ideas)

1. **Grabación**: Añadir botón para grabar sesión
2. **Configuración**: Archivo config.yaml para ajustar parámetros
3. **Aria Real**: Probar con gafas físicas cuando estén disponibles
4. **Haptic Feedback**: Integrar con dispositivos hápticos
5. **Multi-idioma**: TTS en diferentes idiomas
