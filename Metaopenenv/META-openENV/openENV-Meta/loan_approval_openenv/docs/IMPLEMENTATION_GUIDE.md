# Implementation Guide

## Scope
This document is the canonical implementation and maintenance guide for this repository after cleanup and Phase 2 completion.

## What Was Completed
### Phase 1
- Head-agent prompt newline formatting fixed
- Mutable default in voting result fixed
- API action validation improved

### Phase 2
- Council recommendation caching added per episode
- Hard-mode step path now supports shaped reward integration
- Dissent logging now records all dissenters in log storage
- App flow optimized to reduce unnecessary council calls in manual mode
- Inference flow updated to pass precomputed council recommendation where applicable

## Repository Cleanup Performed
- Removed transient/runtime clutter:
  - `__pycache__/`
  - `*.pyc`
  - `.venv/`
  - `loan_approval_openenv.egg-info/`
  - stale JSON runtime logs
- Standardized structure:
  - Main docs consolidated under `docs/`
  - Historical planning docs moved to `docs/archive/`
  - `logs/` retained with `.gitkeep`

## Current Folder Structure
```text
loan_approval_openenv/
├── app.py
├── inference.py
├── requirements.txt
├── openenv.yaml
├── .env
├── .env.example
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

## Operational Behavior
### Environment keys
- Active key used by application: `OPENAI_API_KEY`
- Loaded via `python-dotenv` in app and inference scripts

### Decision flow
1. Generate applicant
2. Sub-agent evaluations
3. Head-agent council vote
4. Voting aggregation and dissent handling
5. Big Brain decision and reward path

### Fallback mode
If `OPENAI_API_KEY` is missing, head agents run deterministic fallback logic.

## Validation Checklist
Use after any code change:
1. `python test_offline.py`
2. `python inference.py`
3. `python app.py` and manual UI smoke test

## Maintenance Rules
- Keep runtime artifacts out of git
- Keep `logs/` folder but not mutable log payloads
- Keep only one canonical guide (`docs/IMPLEMENTATION_GUIDE.md`)
- Archive old planning documents under `docs/archive/`

## Archived Documents
The following were preserved in `docs/archive/`:
- DOCUMENTATION_INDEX.md
- IMPLEMENTATION_ROADMAP.md
- IMPLEMENTATION_SUMMARY.md
- PHASE_1_EXECUTION_LOG.md
- PHASE_2_DETAILED_GUIDE.md
- QUICK_REFERENCE.md
