# Dimljus — Video LoRA Training Toolkit

A purpose-built toolkit for video LoRA training on diffusion transformer models (Wan 2.1/2.2 T2V/I2V). Built by [Alvdansen Labs](https://github.com/alvdansen).

**Status**: Phase 1 complete (Video Ingestion & Captioning). 190 tests passing.

## What Dimljus Does Today

Dimljus solves the **#1 barrier to entry** in video LoRA training: getting from raw footage to clean, properly formatted training clips. These tools are **fully standalone** — they produce standard output formats that work with musubi-tuner, ai-toolkit, or any other trainer.

### Video Pipeline

```bash
# Scan a folder of pre-cut clips — get a validation report
python -m dimljus.video scan "C:\path\to\clips"

# Ingest a long video — scene detect, split, normalize
python -m dimljus.video ingest "C:\path\to\video.mp4" --output "C:\output"

# Normalize clips to training specs (16fps, 480p, 4n+1 frames)
python -m dimljus.video normalize "C:\clips" --output "C:\normalized"
```

### VLM Captioning

```bash
# Caption clips using Gemini API
python -m dimljus.video caption "C:\clips" --provider gemini

# Caption clips using Replicate API (e.g. Gemini via Replicate)
python -m dimljus.video caption "C:\clips" --provider replicate

# Audit existing captions against VLM output
python -m dimljus.video audit "C:\clips" --provider gemini
```

### Data Config

```yaml
# dimljus_data.yaml — minimum viable config
dataset:
  path: ./video_clips

# Everything else defaults to Wan training priors:
# 16 fps, 480p, auto frame count (4n+1), sidecar .txt captions
```

## Installation

```bash
# Core (config schema + validation)
pip install -e .

# With video tools (scene detection, ffmpeg integration)
pip install -e ".[video]"

# With captioning (Gemini + Replicate backends)
pip install -e ".[caption]"

# Everything
pip install -e ".[all]"

# Development
pip install -e ".[dev]"
```

**System requirements**: ffmpeg and ffprobe must be available. Install via `winget install ffmpeg` (Windows) or your system package manager.

## Project Structure

```
dimljus/
  config/          Phase 0 — Data config schema (Pydantic v2, YAML)
  video/           Phase 1 — Video probe, validate, scene detect, split/normalize, CLI
  caption/         Phase 1 — VLM captioning (Gemini, Replicate), prompts, batch orchestrator
```

## What's Next

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Data Config Schema | Complete |
| 1 | Video Ingestion & Captioning | Complete |
| 2 | Caption Refinement | Next |
| 3 | Image Extraction & Processing | Planned |
| 4 | Dataset Validation & Organization | Planned |
| 5 | Training Config Schema | Planned |
| 6 | Latent Pre-Encoding | Planned |
| 7 | Training Infrastructure | Planned |
| 8 | Wan Model Implementations | Planned |
| 9 | Differential MoE Training | Planned |
| 10 | Control Signal Expansion | Planned |

## Philosophy

- **Curator-first, not math-first.** Datasets and aesthetic judgment drive technical decisions.
- **Video-native.** Videos are temporally coherent sequences, not stacked images.
- **Standalone tools.** Data preparation works with ANY trainer, not just Dimljus.
- **Control signal routing.** Every input (caption, reference image, depth map) is a first-class control signal with its own preparation, validation, and weighting.
- **Validate before you compute.** Catch problems before burning GPU time.

## Documentation

- [Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md) — How video LoRA training works, pipeline design
- [Training Methodology](docs/TRAINING_METHODOLOGY.md) — Differential MoE, dataset standards
- [Control Signal Architecture](docs/CONTROL_SIGNAL_ARCHITECTURE.md) — Signal routing design
- [Model Landscape](docs/MODEL_LANDSCAPE.md) — Ecosystem analysis, trainer gaps
- [Research Log](docs/RESEARCH_LOG.md) — Chronological findings and decisions

## License

Apache-2.0
