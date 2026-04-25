---
title: AI Loan Approval System
emoji: 🏦
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# AI Loan Approval System
## Overview
A three-level loan decision engine built for Hugging Face Spaces:
- Level 1: 9 deterministic sub-agents including bank-KYC coverage in fraud
- Level 2: 3 head agents (Powered by OpenRouter LLMs with deterministic fallback)
- Level 3: Dynamic weighted final decision engine

The app provides:
- Gradio UI on port 7860 (Hugging Face ready Glass theme)
- FastAPI API on port 8000

## Current Status
- Transitioned to OpenRouter APIs for free model execution.
- Added dynamic model selection from the UI.
- Removed API cost-optimizations (training happens on every step now).
- Cleaned up directory structure into a concise, flat repository.

## Repository Layout
```text
.
├── app.py
├── inference.py
├── requirements.txt
├── openenv.yaml
├── .env                 # local only (ignored)
├── .env.example         # template to copy
├── .gitignore
├── env/
├── agents/
├── brain/
├── logs/
│   └── .gitkeep
└── docs/
    ├── IMPLEMENTATION_GUIDE.md
    └── archive/
```

## Setup
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
Create local secrets file from template:
```bash
cp .env.example .env
```

Set your OpenRouter keys in `.env` for the three distinct agents:
```env
NEMOTRON_API_KEY=sk-or-v1-...
GLM_API_KEY=sk-or-v1-...
MINIMAX_API_KEY=sk-or-v1-...
```

If any key is empty, the system runs in DEMO mode with deterministic fallbacks.

Additional runtime controls are available in `.env`:
- `RL_TRACE_ENABLED`, `RL_TRACE_PATH`, `RL_TRACE_FLUSH_EVERY`
- `GRPO_TRAIN_ENABLED`, `GRPO_TRAIN_EVERY_STEPS`, `GRPO_MIN_TRACE_BEFORE_TRAIN`
- `REWARD_FLOOR`, `UNCERTAINTY_PENALTY_ENABLED`, `UNCERTAINTY_PENALTY_WEIGHT`

### 3. Run app
```bash
python app.py
```
- UI: http://localhost:7860
- API: http://localhost:8000

### 4. Run inference benchmark
```bash
python inference.py
```

## API Endpoints
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`

## Validation Commands
```bash
python test_offline.py
python inference.py
```

## Documentation
- Main implementation doc: `docs/IMPLEMENTATION_GUIDE.md`
- Historical docs: `docs/archive/`
