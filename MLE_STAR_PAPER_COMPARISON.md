# MLE-STAR Paper vs Implementation Comparison

## Key Finding: **YES, Debug WAS in the Original Paper!**

### Section 3.4: "Additional modules for robust MLE agents"

The paper explicitly describes three modules:

1. **Debugging agent (A_debugger)**: 
   ```
   s ← A_debugger(s, T_bug)
   ```
   - Called when execution triggers an error
   - Iteratively updates script until success OR max rounds reached
   - If bug can't be resolved, proceeds with latest executable version

2. **Data leakage checker (A_leakage)**:
   - Checks solution script BEFORE execution
   - Extracts data preprocessing code block
   - Generates corrected version if leakage detected

3. **Data usage checker (A_data)**:
   - Checks initial solution s_0 before refinement starts
   - Ensures all provided data sources are used

---

## Comparison: Paper vs Our Implementation vs AIDE

### 1. Debugging Agent

**Paper (Section 3.4):**
```
If execution of Python script s triggers an error (T_bug):
  s ← A_debugger(s, T_bug)
  Repeat until success OR max rounds reached
```

**Our Implementation:**
- ✅ We have `_debug_code()` (Prompt 11)
- ❌ **NOT called automatically on failure**
- ❌ No journal context (parent nodes, feedback)
- ❌ No iterative retry loop

**AIDE Pattern:**
```python
if parent_node.is_buggy:
    result_node = self._debug(parent_node)  # Gets context!
```
- ✅ Uses journal: `parent_node.code`, `parent_node.feedback`
- ✅ Includes: `task_desc`, `data_preview`
- ✅ Maintains parent-child relationships

**What We Need:**
```python
# When execution fails:
if exec_result["action_status"] == "FAILED":
    # Create node, mark as buggy
    node = Node(code=code, ...)
    node.is_buggy = True
    journal.append(node)
    
    # Debug with context (like AIDE)
    debug_node = _debug_with_context(node)  # Gets parent_node.code, feedback, task_desc, data_preview
    # Retry execution
    # Repeat until success or max rounds
```

---

### 2. Data Leakage Checker

**Paper:**
- Checks BEFORE execution
- Extracts data preprocessing code block
- Generates corrected version if needed

**Our Implementation:**
- ✅ Prompt 12: Check leakage
- ✅ Prompt 13: Fix leakage
- ❌ Only called in validation phase, not before every execution
- ❌ Should be called before execution (per paper)

**AIDE:**
- Doesn't have explicit leakage checker
- Relies on LLM to avoid leakage

**What We Need:**
```python
# Before execution (per paper):
c_data = extract_data_preprocessing_block(s)
if A_leakage(c_data) detects leakage:
    c_data* = A_leakage(c_data)  # Generate corrected version
    s = s.replace(c_data, c_data*)
```

---

### 3. Data Usage Checker

**Paper:**
- Checks initial solution s_0 BEFORE refinement starts
- Ensures all provided data sources are used

**Our Implementation:**
- ✅ Prompt 14: Check data usage
- ✅ Called in validation phase
- ✅ Matches paper (called before refinement)

**Status:** ✅ CORRECT

---

## Algorithm Flow Comparison

### Algorithm 1: Initial Solution

**Paper:**
```
1. A_retriever(T_task) → {T_model_i, T_code_i}
2. For i=1 to M:
   - s_init_i = A_init(T_task, T_model_i, T_code_i)
   - Evaluate h(s_init_i)  ← CAN FAIL, NEEDS DEBUG
3. Sort by score
4. Sequential merging:
   - s_candidate = A_merger(s_0, s_init_{π(i)})
   - Evaluate h(s_candidate)  ← CAN FAIL, NEEDS DEBUG
```

**Our Implementation:**
- ✅ Steps 1-4 implemented
- ❌ **No automatic debug on evaluation failure**
- ❌ Should call A_debugger when h(s) fails

---

### Algorithm 2: Targeted Refinement

**Paper:**
```
For t = 0 to T-1:
  1. a_t = A_abl(s_t, T_abl)  # Ablation study
  2. r_t = exec(a_t)          ← CAN FAIL, NEEDS DEBUG
  3. T_abl^t = A_summarize(a_t, r_t)
  4. (c_t, p_0) = A_extractor(...)
  5. c_t^0 = A_coder(c_t, p_0)
  6. s_t^0 = s_t.replace(c_t, c_t^0)
  7. Evaluate h(s_t^0)       ← CAN FAIL, NEEDS DEBUG
  8. For k = 1 to K-1:
     - p_k = A_planner(...)
     - c_t^k = A_coder(c_t, p_k)
     - s_t^k = s_t.replace(c_t, c_t^k)
     - Evaluate h(s_t^k)     ← CAN FAIL, NEEDS DEBUG
```

**Our Implementation:**
- ✅ All steps implemented
- ❌ **No automatic debug on execution/evaluation failure**
- ❌ This is where debug is MOST critical (targeted refinement)

---

### Algorithm 3: Ensemble

**Paper:**
```
1. e_0 = A_ens_planner({s_final^l})
2. s_ens^0 = A_ensembler(e_0, {s_final^l})
3. Evaluate h(s_ens^0)       ← CAN FAIL, NEEDS DEBUG
4. For r = 1 to R-1:
   - e_r = A_ens_planner(...)
   - s_ens^r = A_ensembler(...)
   - Evaluate h(s_ens^r)     ← CAN FAIL, NEEDS DEBUG
```

**Our Implementation:**
- ✅ All steps implemented
- ❌ **No automatic debug on evaluation failure**

---

## Perplexity vs Claude Usage

### Paper:
- **Doesn't specify** which LLM to use
- Mentions "LLM agents" generically
- Web search is a "tool" - doesn't specify provider

### Claude Flow Wiki:
- Mentions "Claude integration" as prerequisite
- Suggests using Claude for execution
- But doesn't detail web search implementation

### Our Implementation:
- ✅ **Perplexity for web search** (A_retriever) - Makes sense!
  - Perplexity is designed for web search
  - Better than using Claude for search
- ✅ **Claude/OpenAI for code generation** (A_init, A_coder, etc.)
- ✅ **Perplexity API** for model retrieval

**This is CORRECT:**
- Perplexity = Web search tool (better than Claude for search)
- Claude/GPT = Code generation (better reasoning)

---

## What's Missing in Our Implementation

### Critical Gaps:

1. **Automatic Debug on Failure:**
   ```python
   # Should be:
   result = exec_callback(code)
   if result["action_status"] == "FAILED":
       # Create buggy node
       node = Node(code=code, ...)
       node.is_buggy = True
       journal.append(node)
       
       # Debug with context (like AIDE)
       for attempt in range(max_debug_rounds):
           debug_node = _debug_with_context(node)  # Gets parent context!
           result = exec_callback(debug_node.code)
           if result["action_status"] != "FAILED":
               break
           node = debug_node  # Next iteration uses this as parent
   ```

2. **Journal Context in Debug:**
   - Need: `parent_node.code`, `parent_node.feedback`, `task_desc`, `data_preview`
   - Currently: Only gets `code` and `error` string

3. **Leakage Check Before Execution:**
   - Paper says: Check BEFORE execution
   - We do: Check in validation phase only

4. **Iterative Debug Loop:**
   - Paper says: "Repeat until success OR max rounds"
   - We have: Single debug attempt (no retry loop)

---

## Feature Parity Summary

| Feature | Paper | Our Impl | AIDE | Status |
|---------|-------|----------|------|--------|
| A_retriever (web search) | ✅ | ✅ (Perplexity) | ❌ | ✅ |
| A_init (initial solution) | ✅ | ✅ | ✅ | ✅ |
| A_merger (merge solutions) | ✅ | ✅ | ❌ | ✅ |
| A_abl (ablation study) | ✅ | ✅ | ❌ | ✅ |
| A_extractor (extract block) | ✅ | ✅ | ❌ | ✅ |
| A_coder (refine block) | ✅ | ✅ | ✅ | ✅ |
| A_planner (alternative plans) | ✅ | ✅ | ❌ | ✅ |
| A_ens_planner (ensemble) | ✅ | ✅ | ❌ | ✅ |
| A_ensembler (implement ensemble) | ✅ | ✅ | ❌ | ✅ |
| **A_debugger (auto on failure)** | ✅ | ❌ | ✅ | ❌ **MISSING** |
| A_leakage (before execution) | ✅ | ⚠️ (after) | ❌ | ⚠️ |
| A_data (check usage) | ✅ | ✅ | ❌ | ✅ |
| **Journal context in debug** | N/A | ❌ | ✅ | ❌ **MISSING** |

---

## Recommendations

1. **Implement automatic debug wrapper** (like AIDE pattern):
   - Wrap all `exec_callback()` calls
   - Create journal nodes for each attempt
   - Call `_debug_code()` with parent context on failure
   - Retry until success or max rounds

2. **Enhance `_debug_code()` with context:**
   - Add `parent_node` parameter
   - Include: `task_desc`, `data_preview`, `parent_node.feedback`
   - Match AIDE's prompt structure

3. **Move leakage check before execution:**
   - Check before every execution (per paper)
   - Not just in validation phase

4. **Perplexity usage is CORRECT:**
   - Keep using Perplexity for web search
   - Keep using Claude/GPT for code generation
   - This is optimal architecture

