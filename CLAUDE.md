# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the application (Flask backend on :5000 + Gradio UI on :7860)
python main.py

# Download BLIP models locally (~800MB, run once before first use)
python download_blip_models.py

# Run all tests
pytest

# Run a single test file
pytest tests/test_model_manager.py

# Run a single test function
pytest tests/test_model_manager.py::TestModelManager::test_initialization -v

# Run tests with output
pytest -v -s
```

## Architecture

**Data flow**: User → Gradio UI (`app/frontend/gradio_app.py`) → HTTP calls → Flask API (`app/backend/services/flask_app.py`) → `ModelManager` (`app/backend/utils/model_manager.py`) orchestrates three AI subsystems:

1. **Whisper** (HuggingFace `openai/whisper-tiny`) — speech-to-text, loaded eagerly at startup
2. **Vision** — three-tier backend selection (priority order, determined at init):
   - `deepseek+blip`: DeepSeek text API + local BLIP models (VQA + captioning). BLIP extracts visual features, DeepSeek crafts the natural language response. BLIP is lazy-loaded on first image request.
   - `gemini`: Google Gemini API with native image+text support
   - `blip`: Pure local BLIP (no API key needed, limited quality)
3. **Kokoro TTS** (`hexgrad/Kokoro-82M`) — text-to-speech with 8 languages, ~63 voices. Pipelines are cached per language code (first letter of voice ID = lang code). Assets auto-downloaded from HuggingFace on first run.

**Startup flow** (`main.py`):
- Clears `app/uploads/` directory
- Registers cleanup on exit
- Starts Flask in a daemon thread (port 5000, debug off, reloader off)
- Launches Gradio in the main thread (port 7860)

**Gradio UI** loads voice metadata directly from `kokoro_voices.py` (avoids startup race condition with the Flask API) and communicates with the backend exclusively via HTTP.

## Configuration

**All configuration is read from `.env` only** — the app uses `dotenv_values()`, not `os.environ`. System environment variables are ignored.

Key `.env` variables:
- `DEEPSEEK_API_KEY` — if set (and not a placeholder), enables DeepSeek+BLIP hybrid mode (highest priority)
- `GOOGLE_API_KEY` — if set and DeepSeek is not, enables Gemini mode
- `FLASK_SECRET_KEY` — Flask session secret
- `UPLOAD_FOLDER` — defaults to `app/uploads`
- `STT_MODEL`, `GEMINI_MODEL`, `KOKORO_ASSETS_DIR`, `KOKORO_REPO_ID` — optional overrides

`main.py` also hardcodes proxy bypass logic (clears all proxy env vars, sets `HF_ENDPOINT` to hf-mirror.com) — this is intentional for direct network access in certain environments.

## Key Design Decisions

- **BLIP models are lazy-loaded** (`_ensure_blip_loaded()`) — they only load on the first image request, not at startup. If loading fails, the vision backend degrades gracefully (text-only chat for DeepSeek, error message for pure BLIP).
- **Kokoro pipeline caching** (`_get_kokoro_pipeline()`) — reuses the model object across pipelines to avoid redundant HuggingFace downloads when switching languages.
- **TTS chunking**: Texts under 150 chars are processed as a single chunk. Longer texts are split by sentence boundaries and processed in parallel with `ThreadPoolExecutor(max_workers=2)`, with fallback to sequential processing.
- **Voice ID convention**: The first letter of the voice ID encodes the language (`a`=American English, `b`=British, `j`=Japanese, `z`=Mandarin, `e`=Spanish, `f`=French, `h`=Hindi, `i`=Italian, `p`=Portuguese).
- **Whisper requires exactly 3000 frames** of input features — audio is padded or truncated accordingly after resampling to 16kHz.

## Testing

Tests are in `tests/` and use pytest with `unittest.mock.patch`. Test fixtures in `conftest.py` generate synthetic test images and audio files. Tests do **not** load real ML models — they mock `transformers`, `google.generativeai`, `KPipeline`, and `soundfile` at the import/patch level.

- `test_model_manager.py` — mocks all ML dependencies; tests init, transcription, vision, TTS, voice retrieval
- `test_flask_api.py` — uses Flask test client with a mocked `ModelManager` injected via `monkeypatch`
- `test_gradio_interface.py` — mocks `gr.Blocks` and `requests`; tests UI event handlers in isolation

Test data is stored in `tests/test_data/` and auto-cleaned after the session.
