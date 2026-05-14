---
trigger: always_on
---

# AGENTS.md - System Instructions for Video Search Project

## 🏗️ Project Architecture & Modularity
- **Modular Codebase**: All code must be decoupled into the following specific files:
    - `video_utils.py`: For OpenCV/PyAV frame extraction and sampling logic.
    - `model_utils.py`: For SigLIP/CLIP model loading and embedding generation.
    - `db_handler.py`: For LanceDB/Vector store interactions.
    - `app.py`: For the Streamlit user interface and search logic.
- **Entry Points**: Do not write heavy logic in `main.py`. Use it only as a clean entry point to call modular functions.

## 🐍 Coding Standards
- **Style Guide**: Always follow **PEP 8** strictly for naming conventions, spacing, and structure.
- **Documentation**: Every function must have a clear docstring explaining inputs, outputs, and the specific AIML logic used.
- **Type Hinting**: Use Python type hints (e.g., `def process(video: str) -> list:`) for all signatures.

## 🛡️ Security & Execution Guardrails
- **Terminal Permissions**: **Do not auto-execute any terminal commands** (e.g., `pip install`, `rm -rf`) without explicit user confirmation in the chat.
- **Environment**: Use a virtual environment. Suggest the exact `pip install` commands needed rather than running them.
- **Data Safety**: Do not delete any video files or generated `/data` folders without asking first.

## 🚀 Performance Requirements (Variphi Special)
- **Memory Efficiency**: Always use **Generators** or "Yield" patterns when processing video frames to handle 30-minute segments without RAM spikes.
- **Latency**: Ensure all vector search operations are optimized for sub-second retrieval.