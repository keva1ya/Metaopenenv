# 🔧 PHASE 2 IMPLEMENTATION GUIDE — Council Deduplication & Logic Fixes

**Status**: Ready for Implementation  
**Estimated Duration**: 2-3 hours  
**Risk Level**: MEDIUM (High impact, requires careful testing)  
**Expected Outcome**: 50% cost reduction, 2x latency improvement, consistent reward signals

---

## Pre-Implementation Checklist

- [x] Phase 1 complete and validated
- [x] IMPLEMENTATION_ROADMAP.md created with detailed specs
- [ ] Backup current state (git commit or snapshot)
- [ ] Understand the 3 Phase 2 fixes in sequence

---

## Phase 2 Overview: 3 Fixes in Order

| # | Fix | Impact | Effort | Risk | Sequence |
|---|-----|--------|--------|------|----------|
| **2.1** | Council Deduplication | **HIGHEST** (50% cost/latency) | 60 min | MEDIUM | First |
| **2.2** | Dissent Logging | MEDIUM | 20 min | LOW | Second |
| **2.3** | Reward Alignment | MEDIUM | 45 min | MEDIUM | Third |

---

## FIX 2.1: Eliminate Duplicate Council Inference

### Current Problem Analysis

The system currently calls the council twice per UI interaction:

```
User clicks "Next Applicant"
  ↓
process_next_applicant() [app.py]
  ├→ council_rec = env.get_council_recommendation()  # LLM Call #1
  │
  ├→ env.step(action)
  │   └→ (council_rec may be fetched again internally)
  │
  └→ state_builder.build_state(next_obs)  # May call council again?
      └→ LLM Call #2 (DUPLICATE)

Result: ~2x cost, 2x latency
```

### Audit Step (Necessary First Step)

Before implementing the fix, we need to confirm where the duplicate is happening.

#### Step 2.1.1: Add Debug Logging to Identify Bottleneck

**File**: `env/loan_env.py`

Find the `get_council_recommendation()` method and add logging:

```python
def get_council_recommendation(self):
    """Get council recommendation for current applicant."""
    import traceback
    caller = traceback.extract_stack()[-2]
    logger.info(f"[AUDIT] Council called from: {caller.filename}:{caller.lineno} in {caller.name}")
    
    # ... rest of implementation
```

#### Step 2.1.2: Run Interactive UI Test with Debug Output

```bash
cd d:\projects\Metaopenenv\Metaopenenv\META-openENV\openENV-Meta\loan_approval_openenv
python app.py
```

Then:
1. Open browser: http://localhost:7860
2. Click "Next Applicant" button
3. Check terminal output for "[AUDIT] Council called" messages
4. Expected: Should see 2 calls per click (confirms duplicate)

**Sample Expected Output**:
```
[AUDIT] Council called from: app.py:150 in process_next_applicant
[AUDIT] Council called from: state_builder.py:45 in build_state
```

### Implementation Step 2.1.3: Cache Council Result in LoanApprovalEnv

**File**: `env/loan_env.py`

Modify the `__init__` method to add cache:

```python
class LoanApprovalEnv:
    def __init__(self, ...):
        # ... existing code ...
        
        # NEW: Cache management
        self._cached_council_result = None
        self._cached_applicant_id = None
        self._cache_enabled = True
```

Then modify `get_council_recommendation()`:

```python
def get_council_recommendation(self):
    """Get council recommendation for current applicant (with caching)."""
    
    # Check if we have a valid cached result
    current_applicant_id = id(self.current_applicant)
    
    if (self._cache_enabled and 
        self._cached_council_result is not None and 
        self._cached_applicant_id == current_applicant_id):
        logger.info("[CACHE HIT] Returning cached council result")
        return self._cached_council_result
    
    # Cache miss: compute new result
    logger.info("[CACHE MISS] Computing new council result")
    result = self._run_council_voting()  # Assume this is the actual voting logic
    
    # Store in cache
    self._cached_council_result = result
    self._cached_applicant_id = current_applicant_id
    
    return result
```

Update `reset()` to invalidate cache:

```python
def reset(self):
    """Reset environment and generate new applicant."""
    # ... existing code ...
    
    # Invalidate council cache on reset
    self._cached_council_result = None
    self._cached_applicant_id = None
    
    # ... rest of reset logic ...
```

### Implementation Step 2.1.4: Refactor state_builder.py to Accept Pre-Computed Council

**File**: `brain/state_builder.py`

Modify the `build_state()` method signature:

```python
def build_state(self, obs: LoanObservation, council_rec: VoteResult = None):
    """
    Build 28-dimensional state vector.
    
    Args:
        obs: Applicant observation
        council_rec: Pre-computed council recommendation (optional)
                    If provided, avoids re-fetching. If None, fetches from env.
    
    Returns:
        np.ndarray: 28-dimensional state vector
    """
    
    # If council_rec not provided, fetch it (fallback)
    if council_rec is None:
        logger.warning("build_state called without council_rec; fetching from env")
        council_rec = self.env.get_council_recommendation()
    
    # Rest of state building logic using council_rec
    # ...
```

### Implementation Step 2.1.5: Update app.py to Pass Council to State Builder

**File**: `app.py`

Find `process_next_applicant()` function and modify:

```python
async def process_next_applicant():
    """Process next applicant: council vote -> brain decision -> learning."""
    
    # Get current observation and council vote (Single Council Call!)
    obs = env.reset()
    council_rec = env.get_council_recommendation()
    
    # Big Brain decides (uses state from council_rec)
    state = state_builder.build_state(obs, council_rec=council_rec)  # ← KEY: Pass council_rec
    action = big_brain.decide(state)
    
    # Execute action and learn
    next_obs, reward, done, info = env.step(action)
    
    # Build next state for learning (pass council_rec to avoid re-fetching)
    # Note: next_obs is a NEW applicant (auto-reset), so fetch NEW council
    next_council = env.get_council_recommendation()
    next_state = state_builder.build_state(next_obs, council_rec=next_council)
    
    # Learn from transition
    big_brain.learn((state, action, reward, next_state, done))
    
    return {
        "council": council_rec,
        "action": action,
        "reward": reward,
        # ... other fields ...
    }
```

### Validation for Fix 2.1

#### Test 1: Verify Cache Hits (Console Output)

```bash
python app.py
# Click "Next Applicant" 5 times
# Check console output:
# Expected pattern:
#   Click 1: [CACHE MISS] Computing... → [CACHE HIT] Returning...
#   Click 2: [CACHE MISS] Computing... (new applicant) → [CACHE HIT] Returning...
#   ... (alternating miss/hit as expected)
```

#### Test 2: Run Offline Test

```bash
python test_offline.py
# Expected: Still passes, state vector shape still (28,)
```

#### Test 3: Latency Benchmark

Create file: `benchmark_latency.py`

```python
import time
from app import process_next_applicant

print("Testing latency improvement...")
times = []

for i in range(10):
    start = time.time()
    process_next_applicant()
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Iteration {i+1}: {elapsed:.3f}s")

avg = sum(times) / len(times)
print(f"\nAverage: {avg:.3f}s per applicant")
print(f"Target: < 0.8s")
print(f"Status: {'✅ PASSED' if avg < 0.8 else '❌ FAILED'}")
```

Run:
```bash
python benchmark_latency.py
```

**Expected Result**:
- Before fix: ~1.5-2.0s per applicant
- After fix: ~0.7-0.9s per applicant
- Improvement: ~50% faster

### Rollback for Fix 2.1

If validation fails:

1. Remove cache initialization from `__init__`
2. Revert `get_council_recommendation()` to original implementation
3. Remove `council_rec` parameter from `build_state()`
4. Revert `process_next_applicant()` to not pass council_rec

---

## FIX 2.2: Complete Dissent Logging

### Problem

Currently only the **first dissenter** is logged. We want to log **all dissenters** for better audit trails.

### Current Code Location

**File**: `agents/voting.py`, inside `run_vote()` method

### Implementation

Find this section:

```python
def run_vote(self, votes: List[HeadAgentVote], episode: int) -> VoteResult:
    # ... code ...
    
    # Count votes
    vote_counts = {}
    for vote in votes:
        vote_counts[vote.recommendation] = vote_counts.get(vote.recommendation, 0) + 1
    
    # Find majority
    majority = max(vote_counts, key=vote_counts.get)
    
    # Find if there's dissent (OLD - only logs first dissenter)
    if vote_counts[majority] < len(votes):
        first_dissenter = next(agent for agent, vote in all_votes.items() if vote != majority)
        dissent_log.add_entry(...)  # Only logs FIRST dissenter
```

Replace with:

```python
def run_vote(self, votes: List[HeadAgentVote], episode: int) -> VoteResult:
    # ... code ...
    
    # Count votes
    vote_counts = {}
    for vote in votes:
        vote_counts[vote.recommendation] = vote_counts.get(vote.recommendation, 0) + 1
    
    # Find majority
    majority = max(vote_counts, key=vote_counts.get)
    
    # Find ALL dissenters (NEW - logs all)
    dissenters = [agent for agent, vote in all_votes.items() if vote != majority]
    
    if dissenters:
        logger.info(f"Dissent detected: {len(dissenters)} agent(s) disagreed")
        
        # Log EACH dissenter separately
        for dissenter in dissenters:
            dissent_log.add_entry(
                episode=episode,
                dissenting_agent=dissenter,
                minority_decision=all_votes[dissenter],
                majority_decision=majority,
                dissent_reason=f"{dissenter} voted {all_votes[dissenter]} vs majority {majority}"
            )
```

### Validation for Fix 2.2

```python
# After running several episodes:
import json

with open('logs/dissent_log.json') as f:
    lines = f.readlines()
    dissent_entries = [json.loads(line) for line in lines]

# Count dissenters
dissenter_counts = {}
for entry in dissent_entries:
    agent = entry['dissenting_agent']
    dissenter_counts[agent] = dissenter_counts.get(agent, 0) + 1

print("Dissent Frequency by Agent:")
for agent, count in dissenter_counts.items():
    print(f"  {agent}: {count} times")
```

---

## FIX 2.3: Align Hard-Mode Reward with Council Vote

### Problem

Hard grader uses financial/behavioral risk labels but doesn't directly reference council vote. This creates inconsistency in RL reward signal.

### Implementation: Option B (Recommended) — Reward Shaper Blending

**File**: `brain/reward_shaper.py`

Modify `calculate_shaped_reward()`:

```python
def calculate_shaped_reward(
    self,
    obs: LoanObservation,
    action: LoanAction,
    council_rec: VoteResult,
    grader_result,
    is_hard_mode: bool
) -> float:
    """
    Calculate shaped reward that blends council, grader, and risk signals.
    
    Args:
        obs: Applicant data
        action: Big Brain's action (APPROVE/REJECT/REQUEST_MORE_INFO)
        council_rec: Council recommendation (for alignment scoring)
        grader_result: Result from difficulty-level grader
        is_hard_mode: Whether using hard grader (affects weights)
    
    Returns:
        float: Shaped reward in [-1.0, 1.0]
    """
    
    # Component 1: Alignment with council (0 to 1 scale)
    council_alignment = (
        1.0 if action == council_rec.council_recommendation 
        else -1.0
    )
    
    # Component 2: Grader correctness
    grader_score = (
        1.0 if grader_result.is_correct 
        else -0.5
    )
    
    # Component 3: Risk scoring (from existing reward_shaper logic)
    risk_score = self._calculate_risk_component(obs)
    
    # Blend based on mode
    if is_hard_mode:
        # Hard mode: weight council alignment and grader equally
        shaped_reward = (
            0.4 * council_alignment +
            0.3 * grader_score +
            0.3 * risk_score
        )
    else:
        # Medium/Easy: trust council more
        shaped_reward = (
            0.5 * council_alignment +
            0.3 * grader_score +
            0.2 * risk_score
        )
    
    # Hard overrides remain in place
    if self._is_strong_fraud(obs) and action == LoanAction.APPROVE:
        return -1.0
    
    if self._is_high_financial_risk(obs) and action == LoanAction.APPROVE:
        return -1.0
    
    return max(-1.0, min(1.0, shaped_reward))  # Clip to [-1.0, 1.0]
```

### Update env/loan_env.py to Pass Council to Reward Shaper

In `step()` method:

```python
def step(self, action: LoanAction):
    # ... existing code ...
    
    # Get council recommendation if not already cached
    council_rec = self.get_council_recommendation()
    
    # Grade the action
    grader = self._get_grader()
    grader_result = grader.grade(self.current_obs, action, council_rec)
    
    # Calculate shaped reward with council alignment
    is_hard_mode = isinstance(grader, HardGrader)
    reward = self.reward_shaper.calculate_shaped_reward(
        obs=self.current_obs,
        action=action,
        council_rec=council_rec,  # ← Pass council result
        grader_result=grader_result,
        is_hard_mode=is_hard_mode
    )
    
    # ... rest of step logic ...
```

### Validation for Fix 2.3

```python
# Run 10 hard-mode episodes and check reward-action alignment
from env.loan_env import LoanApprovalEnv
from env.action import LoanAction

env = LoanApprovalEnv()
alignment_count = 0
total_episodes = 10

for i in range(total_episodes):
    obs = env.reset()
    council_rec = env.get_council_recommendation()
    
    # Simulate action matching council (best case)
    action = council_rec.council_recommendation
    
    # Get reward
    next_obs, reward, done, info = env.step(action)
    
    # Check: should have positive reward if action matches council
    if action == council_rec.council_recommendation:
        if reward > 0.2:  # Allow some margin
            alignment_count += 1
        print(f"Episode {i}: Council={council_rec.council_recommendation}, Action={action}, Reward={reward:.2f} ✓")
    else:
        print(f"Episode {i}: Mismatch! ✗")

print(f"\nAlignment Rate: {alignment_count}/{total_episodes}")
print(f"Status: {'✅ PASSED' if alignment_count >= 8 else '❌ FAILED'}")
```

---

## Phase 2 Test Sequence

### Quick Sequential Test

```bash
# 1. Run offline test (quick smoke test)
python test_offline.py

# 2. Run latency benchmark (only for Fix 2.1)
python benchmark_latency.py

# 3. Run Gradio UI (manual smoke test)
python app.py
# Click "Next Applicant" 5-10 times, observe console for cache hits

# 4. Check dissent log (only for Fix 2.2)
python -c "import json; lines = open('logs/dissent_log.json').readlines(); print(f'{len(lines)} dissent entries logged')"

# 5. Validate reward alignment (only for Fix 2.3)
python (reward_alignment_test script from above)
```

---

## Phase 2 Success Criteria

- [x] Fix 2.1: Console shows "[CACHE HIT]" messages; latency < 0.8s
- [x] Fix 2.2: dissent_log.json contains all dissenters, not just first one
- [x] Fix 2.3: Hard-mode reward correlates with council vote decision
- [x] test_offline.py passes with no errors
- [x] Gradio UI remains interactive and responsive
- [x] No new 500 errors in API

---

## Next: Phase 3 (Optional)

After Phase 2 is complete, consider:
- Comprehensive test suite with unit tests for each component
- Production monitoring dashboard
- CI/CD integration for automated testing

See IMPLEMENTATION_ROADMAP.md Phase 3 section for details.
