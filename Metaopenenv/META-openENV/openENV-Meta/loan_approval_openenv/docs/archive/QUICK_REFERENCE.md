# ⚡ QUICK REFERENCE CHECKLIST — Implementation Roadmap

**Print this out or keep it handy while implementing!**

---

## 🎯 Current Status

- [x] **Phase 1**: ✅ COMPLETE (3 bugs fixed, validated)
- [ ] **Phase 2**: 🔧 READY TO START (3 improvements to implement)
- [ ] **Phase 3**: 📅 OPTIONAL (tests, monitoring, CI/CD)

---

## 📋 Phase 1 Checklist — COMPLETE ✅

### Fix 1.1: Newline Formatting (3 files)
- [x] `agents/council/risk_agent.py` line 84: `'chr(10)'` → `'\n'`
- [x] `agents/council/fraud_agent.py` line 68: `'chr(10)'` → `'\n'`
- [x] `agents/council/policy_agent.py` line 74: `'chr(10)'` → `'\n'`

### Fix 1.2: Mutable Default (1 file)
- [x] `agents/voting.py` line 14: Add `Field` import
- [x] `agents/voting.py` line 44: `= []` → `= Field(default_factory=list)`

### Fix 1.3: API Validation (1 file)
- [x] `app.py` api_step: Add try/except for LoanAction parsing
- [x] Return 400 on invalid action

### Validation
- [x] `python test_offline.py` — PASSED ✅
- [x] State vector shape: (28,) ✅

---

## 🔧 Phase 2 Checklist — READY TO START

### Pre-Implementation
- [ ] Read [PHASE_2_DETAILED_GUIDE.md](PHASE_2_DETAILED_GUIDE.md) (estimated 15 min)
- [ ] Git commit current state (optional but recommended)
- [ ] Clear 2-3 hours for focused work

### Fix 2.1: Council Deduplication (HIGHEST IMPACT)
**Effort**: 60 min | **Impact**: 50% cost reduction

**Step 2.1.1: Add Debug Logging**
- [ ] `env/loan_env.py`: Add logging to `get_council_recommendation()`
- [ ] Run `python app.py` and click "Next Applicant"
- [ ] Verify: See 2 "[AUDIT] Council called" messages per click

**Step 2.1.2: Implement Cache**
- [ ] `env/loan_env.py.__init__()`: Add `_cached_council_result`, `_cached_applicant_id`
- [ ] `env/loan_env.py.get_council_recommendation()`: Add cache hit/miss logic
- [ ] `env/loan_env.py.reset()`: Invalidate cache

**Step 2.1.3: Refactor State Builder**
- [ ] `brain/state_builder.py.build_state()`: Accept optional `council_rec` parameter
- [ ] Use provided council_rec if available, else fetch from env

**Step 2.1.4: Update App Flow**
- [ ] `app.py.process_next_applicant()`: Pass council_rec to state_builder
- [ ] Ensure single council call per applicant

**Validation**
- [ ] Console shows "[CACHE HIT]" after first applicant load
- [ ] `python test_offline.py` — PASSED
- [ ] Latency benchmark: `python benchmark_latency.py`
- [ ] Expected: ~0.8s per applicant (was ~1.5s)

---

### Fix 2.2: Complete Dissent Logging (MEDIUM PRIORITY)
**Effort**: 20 min | **Impact**: Better audit trail

**Step 2.2.1: Modify Dissent Logic**
- [ ] `agents/voting.py.run_vote()`: Find dissent logging section
- [ ] Change: Log ALL dissenters, not just first one
- [ ] Use loop: `for dissenter in dissenters: dissent_log.add_entry(...)`

**Validation**
- [ ] Run several episodes
- [ ] Check `logs/dissent_log.json`
- [ ] Verify: Multiple dissent entries if multiple agents disagreed

---

### Fix 2.3: Reward Alignment (MEDIUM PRIORITY)
**Effort**: 45 min | **Impact**: Consistent RL signal

**Step 2.3.1: Update Reward Shaper**
- [ ] `brain/reward_shaper.py.calculate_shaped_reward()`: Add council_rec parameter
- [ ] Add component: `council_alignment = 1.0 if action == council_rec else -1.0`
- [ ] Blend weights: Hard mode (0.4 alignment, 0.3 grader, 0.3 risk)

**Step 2.3.2: Update Env Step**
- [ ] `env/loan_env.py.step()`: Pass council_rec to reward_shaper
- [ ] Pass is_hard_mode flag

**Validation**
- [ ] Run 10 hard-mode episodes
- [ ] When action matches council: reward > 0.2 (expected)
- [ ] When action differs: reward < 0.2 (expected)

---

### Phase 2 Final Validation
- [ ] All 3 fixes implemented
- [ ] `python test_offline.py` — PASSED
- [ ] `python app.py` — No 500 errors, cache visible in console
- [ ] Check `logs/dissent_log.json` — All dissenters logged
- [ ] Reward alignment test — Passed 8/10 episodes
- [ ] Latency < 0.8s per applicant

---

## 📅 Phase 3 Checklist — OPTIONAL

### Test Suite
- [ ] Create `tests/test_sub_agents.py` — Unit tests for all 8 sub-agents
- [ ] Create `tests/test_integration.py` — Multi-tier workflow tests
- [ ] Create `tests/test_voting.py` — Voting engine tests

### Monitoring Dashboard
- [ ] Add Gradio panels: Approval rate trend, dissent rate, reward trend
- [ ] Add metrics: Average latency, LLM cost estimate
- [ ] Real-time updates on every applicant

### CI/CD
- [ ] Add GitHub Actions workflow
- [ ] Run tests on every commit
- [ ] Lint and format checks

### Documentation
- [ ] Write CHANGELOG.md for v0.1
- [ ] Create troubleshooting guide
- [ ] Update README.md with new features

---

## 📂 Key Files Reference

| Purpose | File | Status |
|---------|------|--------|
| **Overall Roadmap** | [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) | Created |
| **Phase 1 Log** | [PHASE_1_EXECUTION_LOG.md](PHASE_1_EXECUTION_LOG.md) | ✅ Complete |
| **Phase 2 Guide** | [PHASE_2_DETAILED_GUIDE.md](PHASE_2_DETAILED_GUIDE.md) | Created |
| **This Checklist** | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | You are here |

---

## ⏱️ Time Estimates

| Phase | Estimated | Actual | Status |
|-------|-----------|--------|--------|
| **P1** | 30-45m | ~15m | ✅ DONE |
| **P2.1** | 60m | - | 🔧 Ready |
| **P2.2** | 20m | - | 🔧 Ready |
| **P2.3** | 45m | - | 🔧 Ready |
| **P3** | 1-2h | - | 📅 Optional |

---

## 🔍 Testing Commands

```bash
# Quick smoke test (all phases)
python test_offline.py

# Latency benchmark (after Fix 2.1)
python benchmark_latency.py

# Interactive UI (visual inspection)
python app.py
# Then open: http://localhost:7860

# Check dissent log (after Fix 2.2)
python -c "import json; entries = [json.loads(l) for l in open('logs/dissent_log.json')]; print(f'{len(entries)} dissent entries')"

# Reward alignment test (after Fix 2.3)
# See detailed guide for validation script

# View latest dissent entries
tail -5 logs/dissent_log.json
```

---

## 🚨 Troubleshooting Quick Guide

| Issue | Check | Solution |
|-------|-------|----------|
| `test_offline.py` fails | Recent file edits | Verify syntax in modified files |
| Latency not improved | Cache working? | Console should show "[CACHE HIT]" |
| Dissent not logged | Fix 2.2 applied? | Verify loop in run_vote() |
| Reward always negative | Council_rec passed? | Check env.step() calls reward_shaper |
| API returns 500 | Invalid action? | Should return 400 now after Fix 1.3 |

---

## ✅ Sign-Off Checklist (Before Deploying)

- [ ] Phase 1 validated (test_offline.py passes)
- [ ] Phase 2 implemented (all 3 fixes complete)
- [ ] Phase 2 validated (latency < 0.8s, dissent logged, reward aligned)
- [ ] No 500 errors in API
- [ ] Approval rates stable (no unexpected changes)
- [ ] Dissent log shows realistic patterns
- [ ] Documentation updated

---

## 📞 Quick Help

1. **"What do I do first?"**  
   → Read PHASE_2_DETAILED_GUIDE.md

2. **"How long will this take?"**  
   → Phase 2: ~2-3 hours total. Fix 2.1 is longest (60 min).

3. **"Is Phase 2 safe?"**  
   → Medium risk overall. Has detailed validation steps. All changes are backwards-compatible.

4. **"Can I rollback?"**  
   → Yes. Each fix has rollback instructions in its detailed guide.

5. **"Do I need Phase 3?"**  
   → No. Phase 3 is optional polish. Phases 1-2 bring system to production-ready.

---

## 🎯 Success Criteria Summary

### Phase 1 (✅ DONE)
- [x] 3 bugs fixed
- [x] test_offline.py passes
- [x] No new errors introduced

### Phase 2 (IN PROGRESS)
- [ ] Council cached (cache hits visible)
- [ ] Latency < 0.8s per applicant
- [ ] All dissenters logged
- [ ] Reward aligned with council
- [ ] Stable approval rates

### Phase 3 (OPTIONAL)
- [ ] Test suite created
- [ ] Monitoring dashboard added
- [ ] CI/CD pipeline configured

---

## 📊 Progress Tracking

**Date Started**: 2026-04-25  
**Phase 1 Completion**: 2026-04-25 ✅  
**Phase 2 Estimated Start**: [INSERT DATE]  
**Phase 2 Estimated Completion**: [INSERT DATE + 3 hours]  
**Phase 3 Estimated Start**: [INSERT DATE]  

---

**Last Updated**: 2026-04-25  
**Status**: Phase 1 ✅ COMPLETE | Phase 2 🔧 READY | Phase 3 📅 OPTIONAL

**Next Action**: Read PHASE_2_DETAILED_GUIDE.md and begin Fix 2.1 (Council Deduplication)

---

## 🚀 Ready to Start Phase 2?

When you're ready, follow these steps:

1. **Read** the guide: `cat PHASE_2_DETAILED_GUIDE.md`
2. **Implement** Fix 2.1 (60 min) — Council Deduplication
3. **Validate** with latency benchmark
4. **Implement** Fix 2.2 (20 min) — Dissent Logging
5. **Implement** Fix 2.3 (45 min) — Reward Alignment
6. **Validate** entire Phase 2 with test suite

Estimated total time: 2-3 hours.

Good luck! 🎯
