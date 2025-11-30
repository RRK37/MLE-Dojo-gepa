# MLE-STAR Implementation Analysis

## Key Findings

### 1. **Debug Should Be Called on Failure** ✅

**From PaperMethod.md:**
- Prompt 11 is explicitly for debugging: "Debug code with error"
- The algorithms show "Evaluate h(s) using D" - when this fails, we should debug

**From AIDE Pattern:**
- AIDE checks `parent_node.is_buggy` and calls `_debug(parent_node)`
- Debug is a **separate step** that uses journal context

**Current Issue:**
- We have `_debug_code()` but it's not being called automatically
- We're not using journal to track failed attempts
- We're not maintaining parent-child relationships

### 2. **Targeted Refinement (Algorithm 2) is Where Debug Should Work**

**Algorithm 2 Flow (from paper):**
```
4: for t = 0 to T − 1 do
5:     a_t = A_abl(s_t, T_abl)           # Generate ablation
6:     r_t = exec(a_t)                   # Execute ablation ← CAN FAIL
7:     T_abl^t = A_summarize(a_t, r_t)   # Summarize results
8:     (c_t, p_0) = A_extractor(...)     # Extract code block
9:     c_t^0 = A_coder(c_t, p_0)         # Refine code block
10:    s_t^0 = s_t.replace(c_t, c_t^0)   # Replace in solution
11:    Evaluate h(s_t^0) using D         # Evaluate ← CAN FAIL
```

**What Should Happen:**
- When `exec(a_t)` fails → Debug the ablation code
- When `Evaluate h(s_t^0)` fails → Debug the refined solution
- Debug should use context from previous attempts (via journal)

### 3. **Journal/Context is Critical (Like AIDE)**

**AIDE's Debug Pattern:**
```python
def _debug(self, parent_node: Node) -> Node:
    prompt = {
        "Task description": self.task_desc,
        "Previous (buggy) implementation": parent_node.code,
        "Execution output": parent_node.feedback,  # ← Context!
        "Data Overview": self.data_preview,       # ← Context!
    }
    # Creates child node with parent=parent_node
    return Node(..., parent=parent_node, node_type="debug")
```

**What We Need:**
1. ✅ Track failed nodes in journal (we do this)
2. ❌ Pass parent node context to debug (we don't)
3. ❌ Include task_desc and data_preview in debug prompt (we don't)
4. ❌ Maintain parent-child relationships for debug chain (we don't)

### 4. **Feature Parity with Paper Algorithms**

**Algorithm 1: Initial Solution**
- ✅ Retrieve models (A_retriever)
- ✅ Generate solutions (A_init)
- ✅ Evaluate h(s_init_i) ← **NEEDS DEBUG ON FAILURE**
- ✅ Sequential merging (A_merger)
- ✅ Evaluate h(s_candidate) ← **NEEDS DEBUG ON FAILURE**

**Algorithm 2: Targeted Refinement**
- ✅ Ablation study (A_abl)
- ✅ Execute ablation ← **NEEDS DEBUG ON FAILURE**
- ✅ Summarize (A_summarize)
- ✅ Extract code block (A_extractor)
- ✅ Refine code block (A_coder)
- ✅ Evaluate h(s_t^0) ← **NEEDS DEBUG ON FAILURE**
- ✅ Inner loop with alternative plans

**Algorithm 3: Ensemble**
- ✅ Ensemble planner (A_ens_planner)
- ✅ Implement ensemble (A_ensembler)
- ✅ Evaluate h(s_ens^0) ← **NEEDS DEBUG ON FAILURE**

**Validation & Debugging (Prompts 11-14)**
- ✅ Prompt 11: Debug code with error
- ✅ Prompt 12: Check data leakage
- ✅ Prompt 13: Fix data leakage
- ✅ Prompt 14: Check data usage

### 5. **What's Missing**

1. **Automatic Debug on Failure:**
   - Need `_execute_with_debug()` wrapper
   - Should create journal nodes for each attempt
   - Should maintain parent-child relationships

2. **Context in Debug:**
   - Task description
   - Data preview
   - Previous execution feedback
   - Parent node code/feedback

3. **Journal Integration:**
   - Track all execution attempts
   - Mark nodes as buggy
   - Use journal to find parent nodes for debug

## Recommended Implementation

### Pattern: AIDE-Style Debug with Journal

```python
def _execute_with_debug(self, code: str, exec_callback, parent_node=None):
    """Execute with automatic debug on failure, using journal."""
    # Create initial node
    node = Node(code=code, parent=parent_node, node_type="draft")
    result = exec_callback(code)
    self.parse_exec_result(node, result)
    self.journal.append(node)
    
    # If failed, debug (like AIDE)
    if node.is_buggy:
        debug_node = self._debug_code_with_context(node)
        result = exec_callback(debug_node.code)
        self.parse_exec_result(debug_node, result)
        self.journal.append(debug_node)
        return result, debug_node.code, debug_node
    
    return result, code, node

def _debug_code_with_context(self, parent_node: Node) -> Node:
    """Debug using journal context (like AIDE)."""
    prompt = {
        "Task description": self.task_desc,
        "Previous (buggy) implementation": parent_node.code,
        "Execution output": parent_node.feedback,
    }
    if self.data_preview:
        prompt["Data Overview"] = self.data_preview
    
    # Use Prompt 11
    response = self._safe_query_llm(prompts.prompt_11_debug(...))
    fixed_code = extract_code(response)
    
    return Node(
        code=fixed_code,
        parent=parent_node,
        node_type="debug",
        instruction_prompt="Debugging failed code"
    )
```

This matches:
- ✅ AIDE's pattern
- ✅ Paper's Prompt 11
- ✅ Journal-based tracking
- ✅ Context-aware debugging

