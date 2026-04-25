# AI Loan Approval System

## Overview
A three-level loan decision engine:
- Level 1: 8 deterministic sub-agents (risk, fraud, policy)
- Level 2: 3 head agents (LLM with deterministic fallback)
- Level 3: Big Brain DQN final decision

The app provides:
- Gradio UI on port 7860
- FastAPI API on port 8000

## Current Status
- Phase 1 fixes: completed
- Phase 2 fixes: completed and validated
- Repository cleanup: completed

## Repository Layout
```text
loan_approval_openenv/
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

Set your key in `.env`:
```env
OPENAI_API_KEY=sk-...
```

If key is empty, the system runs in DEMO mode with deterministic fallbacks.

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

## Notes on Cleanliness
- Runtime logs and model artifacts are ignored
- Python caches and virtualenvs are removed from repo
- Legacy planning docs are archived in `docs/archive/`

## Documentation
- Main implementation doc: `docs/IMPLEMENTATION_GUIDE.md`
- Historical docs: `docs/archive/`
