# AI Agents - Project Instructions
# This file is read by multiple AI coding assistants (Claude Code, Cursor, Copilot, etc.)

## Git Commits
- NEVER add "Co-Authored-By" lines to commit messages
- Write commit messages in English
- Follow conventional commits format (feat:, fix:, refactor:, docs:, chore:)
- Keep commit messages concise (1-2 lines)

## Communication
- Speak to the user in Spanish
- Be concise, pragmatic. No filler text

## Project Structure
```
docker/                  # All Docker files
  Dockerfile.base        # OpenCV+CUDA+NVDEC base (~20 min build, solo 1 vez)
  Dockerfile.app         # App layer sobre base (~3 min)
  Dockerfile.tensorrt    # Legacy todo-en-uno (no usar)
  Dockerfile.jetson      # ARM64 Jetson Orin
  docker-compose.yml     # MAIN WAY TO RUN THE APP
  docker-compose.jetson.yml
  docker-build.sh        # Helper: base/app/all/run/dev
src/                     # Python source code
scripts/                 # Utility scripts (export_tensorrt.py, etc.)
tests/                   # Test files
models/                  # TensorRT engines, ONNX, weights (mounted volume)
data/                    # Videos, datasets (mounted volume)
docs/                    # Documentation (DOCKER.md, REALSENSE.md)
```

## Docker - IMPORTANT
- **ALWAYS use docker-compose to run**: `docker compose -f docker/docker-compose.yml up`
- **NEVER suggest raw `docker run` commands** — compose handles GPU, ports, volumes, audio, USB
- Build workflow:
  1. Base (solo 1 vez): `./docker/docker-build.sh base` (auto-detecta GPU)
  2. App: `./docker/docker-build.sh app`
  3. Run: `docker compose -f docker/docker-compose.yml up`
- Dev mode (code changes sin rebuild): compose already mounts `src/`, `run.py`, `scripts/` as read-only
- Images: `aria-base:opencv-nvdec` (base) → `aria-demo:tensorrt` (app)
- compose mounts: USB, webcam, PulseAudio, data/, models/, source code
- Required env vars are in compose: `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`
- Port 5000: dashboard web (already in compose)
- See `docs/DOCKER.md` for full documentation

## GPU Architecture
- Build compiles OpenCV CUDA only for the host GPU (auto-detected)
- Manual override: `CUDA_ARCH_BIN="12.0" ./docker/docker-build.sh base`
- RTX 20xx=7.5, 30xx=8.6, 40xx=8.9, 50xx=12.0

## Runtime
- Multi-process: Main (no CUDA) + Detector (CUDA) + TTS (CUDA)
- Main process at port 5000 serves MJPEG stream + dashboard
- Sources: `webcam`, `realsense`, video file path, aria glasses
- Modes: `indoor`, `outdoor`, `all`
- `--no-tts` flag for development without TTS

## Language
- Code comments and docs: Spanish or English (follow existing file convention)
- Commit messages: English
