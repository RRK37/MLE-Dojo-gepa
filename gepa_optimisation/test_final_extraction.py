"""
Test the complete CV extraction flow with the updated observation structure
that includes raw_result.
"""

import re

def extract_cv_score_from_output(obs: dict, execution_succeeded: bool) -> float:
    """Mimics the updated adapter method."""
    if not execution_succeeded:
        return 0.0
        
    import re
    
    stdout_text = None
    
    # Method 1: Check raw_result (NEW - most direct path to stdout)
    if "raw_result" in obs and isinstance(obs["raw_result"], dict):
        raw_result = obs["raw_result"]
        if "execution" in raw_result and isinstance(raw_result["execution"], dict):
            exec_result = raw_result["execution"]
            # Check both 'output' (from utils.py) and 'stdout' (from sandbox.py)
            stdout_text = exec_result.get("output") or exec_result.get("stdout")
            if stdout_text:
                print(f"[Test] ✓ Found stdout in raw_result->execution->output/stdout")
    
    if not stdout_text:
        print(f"[Test] Could not find stdout in observation structure")
        return 0.0
    
    print(f"[Test] Stdout length: {len(stdout_text)}, preview: {stdout_text[:100]}")
    
    # Parse CV score patterns
    patterns = [
        r'cross[-\s]?validated?\s+(?:accuracy|score)[:\s]+([0-9.]+)',
        r'(?:cv|validation)\s+(?:accuracy|score)[:\s]+([0-9.]+)',
        r'(?:test|val)\s+(?:accuracy|score)[:\s]+([0-9.]+)',
        r'accuracy[:\s]+([0-9.]+)',
        r'score[:\s]+([0-9.]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, stdout_text, re.IGNORECASE)
        if match:
            cv_score = float(match.group(1))
            if cv_score > 1.0:
                cv_score = cv_score / 100.0
            if 0.0 <= cv_score <= 1.0:
                print(f"[Test] ✓ Extracted CV score: {cv_score:.4f} using pattern: {pattern}")
                return cv_score
    
    print(f"[Test] No CV score pattern matched")
    return 0.0


# Test with the NEW observation structure that includes raw_result
print("="*70)
print("Test: Observation with raw_result (NEW structure from env.py)")
print("="*70)

obs = {
    "action_status": "FAILED",  # No submission created
    "execution_status": "SUCCESS",  # But code executed successfully
    "feedback": {
        "base": {
            "feedback_status": "SUCCESS",
            "feedback": "=== Code Execution Results ===\nExecution successful..."  # Processed text
        }
    },
    "raw_result": {  # NEW: Direct access to raw execution results
        "status": "FAILED",  # Overall status (no submission)
        "execution": {
            "status": "SUCCESS",
            "output": "Cross-validated accuracy: 0.8283\n",  # THE STDOUT WE NEED!
            "error": "",
            "execution_time": "2.64s"
        },
        "submission": None
    },
    "current_raw_score": 0.0,
    "current_position_score": 0.0
}

score = extract_cv_score_from_output(obs, execution_succeeded=True)
print(f"\n✅ Final Result: {score}")
print(f"   Should be used as reward since no submission was created")
print()

# Test 2: When execution fails
print("="*70)
print("Test 2: Failed execution (should return 0.0)")
print("="*70)

obs2 = {
    "action_status": "FAILED",
    "execution_status": "FAILED",  # Execution failed
    "raw_result": {
        "execution": {
            "status": "FAILED",
            "output": "",
            "error": "ValueError: Cannot center sparse matrices"
        }
    }
}

score2 = extract_cv_score_from_output(obs2, execution_succeeded=False)
print(f"\n✅ Final Result: {score2} (correctly returns 0.0 for failed execution)")
