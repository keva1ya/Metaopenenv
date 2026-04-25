# AI Loan Approval System

## Overview
A three-level loan decision engine built for Hugging Face Spaces:
- Level 1: 8 deterministic sub-agents (risk, fraud, policy)
- Level 2: 3 head agents (Powered by OpenRouter LLMs with deterministic fallback)
- Level 3: Big Brain DQN final decision

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
