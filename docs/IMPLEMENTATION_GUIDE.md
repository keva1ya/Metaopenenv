# Implementation Guide

## Scope
This document is the canonical implementation and maintenance guide for this repository after the OpenRouter integration and repository cleanup.

## What Was Completed
### Core Upgrades (OpenRouter & UI)
- Migrated Head Agents (Risk, Fraud, Policy) from OpenAI to OpenRouter (`https://openrouter.ai/api/v1`).
- Added dynamic UI components (Advanced Settings) to allow changing models on-the-fly.
- Removed API-cost limiters to ensure continuous RL agent training on Hugging Face Spaces.
- Upgraded the UI with a modern `Glass` theme suitable for Hugging Face Spaces.

### Previous Phases
- Head-agent prompt newline formatting fixed
- Mutable default in voting result fixed
- API action validation improved
- Hard-mode step path now supports shaped reward integration
- Dissent logging records all dissenters in log storage

## Repository Cleanup Performed
- Removed highly nested directory layers (`Metaopenenv/Metaopenenv/META-openENV/openENV-Meta/loan_approval_openenv/`).
- Removed transient/runtime clutter:
  - `__pycache__/`
  - `*.pyc`
  - `.venv/`
  - `loan_approval_openenv.egg-info/`
  - stale JSON runtime logs
- Standardized structure:
  - Main docs consolidated under `docs/`
  - Historical planning docs moved to `docs/archive/`

## Operational Behavior
### Environment keys
- Active keys used by application: `NEMOTRON_API_KEY`, `GLM_API_KEY`, `MINIMAX_API_KEY`
- Loaded via `python-dotenv` in app and inference scripts.

### Decision flow
1. Generate applicant
2. Sub-agent evaluations
3. Head-agent council vote
4. Voting aggregation and dissent handling
5. Big Brain decision and reward path

### Fallback mode
If any of the three keys are missing, head agents run deterministic fallback logic.

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
