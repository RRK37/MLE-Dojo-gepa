"""
Test the actual observation structure that comes from env.step()
to verify our CV extraction logic works with the real data format.
"""

import re

def extract_cv_score_from_output(obs: dict, execution_succeeded: bool) -> float:
    """
    Extract cross-validation score from execution stdout.
    This mimics the actual method in adapter.py
    """
    if not execution_succeeded:
        return 0.0
        
    # The observation structure from env.step() contains feedback with execution results
    # We need to check multiple possible locations for the stdout
    stdout_text = None
    
    # Method 1: Check if feedback is directly in obs (most common case)
    if "feedback" in obs and isinstance(obs["feedback"], dict):
        feedback = obs["feedback"]
        
        # Check for nested feedback structure with modes
        for feedback_mode, feedback_data in feedback.items():
            if not isinstance(feedback_data, dict):
                continue
            
            # Try raw_results path first (from feedback manager)
            if "raw_results" in feedback_data:
                raw_results = feedback_data["raw_results"]
                if isinstance(raw_results, dict) and "execution" in raw_results:
                    exec_result = raw_results["execution"]
                    if "stdout" in exec_result:
                        stdout_text = exec_result["stdout"]
                        break
                    elif "output" in exec_result:
                        stdout_text = exec_result["output"]
                        break
        
        # If not found in nested structure, check direct execution key
        if not stdout_text and "execution" in feedback:
            exec_result = feedback["execution"]
            if isinstance(exec_result, dict):
                stdout_text = exec_result.get("stdout") or exec_result.get("output")
    
    # Method 2: Check if obs itself has execution key (alternative structure)
    if not stdout_text and "execution" in obs:
        exec_result = obs["execution"]
        if isinstance(exec_result, dict):
            stdout_text = exec_result.get("stdout") or exec_result.get("output")
    
    if not stdout_text:
        print(f"[Test] Could not find stdout in observation structure")
        print(f"[Test] Obs keys: {list(obs.keys())}")
        if "feedback" in obs:
            print(f"[Test] Feedback type: {type(obs['feedback'])}")
            if isinstance(obs["feedback"], dict):
                print(f"[Test] Feedback keys: {list(obs['feedback'].keys())}")
        return 0.0
    
    print(f"[Test] Found stdout, searching for CV score (length: {len(stdout_text)})")
    print(f"[Test] Stdout preview: {stdout_text[:200]}")
    
    # Parse CV score patterns from stdout
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
            # Sanity check: scores should be between 0 and 1 (or 0-100 if percentage)
            if cv_score > 1.0:
                cv_score = cv_score / 100.0  # Convert percentage to decimal
            if 0.0 <= cv_score <= 1.0:
                print(f"[Test] ✓ Extracted CV score: {cv_score:.4f} using pattern: {pattern}")
                return cv_score
    
    print(f"[Test] No CV score pattern matched in stdout")
    return 0.0


# Test different observation structures based on what env.step() might return

print("="*70)
print("Test 1: Observation with feedback->execution->output (interface.py structure)")
print("="*70)
obs1 = {
    "feedback": {
        "execution": {
            "status": "SUCCESS",
            "output": "Cross-validated accuracy: 0.8283\n",
            "error": "",
            "execution_time": "2.64s"
        },
        "submission": None
    },
    "action_status": "FAILED",  # No submission
    "execution_status": "SUCCESS"
}
score1 = extract_cv_score_from_output(obs1, execution_succeeded=True)
print(f"\nResult: {score1}\n")

print("="*70)
print("Test 2: Observation with nested feedback structure (feedback manager)")
print("="*70)
obs2 = {
    "feedback": {
        "base": {
            "interface_mode": "execute_code",
            "raw_results": {
                "execution": {
                    "status": "SUCCESS",
                    "stdout": "Cross-validated accuracy: 0.8216\n",
                    "stderr": "",
                    "execution_time": 2.64
                }
            }
        }
    },
    "action_status": "FAILED",
    "execution_status": "SUCCESS"
}
score2 = extract_cv_score_from_output(obs2, execution_succeeded=True)
print(f"\nResult: {score2}\n")

print("="*70)
print("Test 3: Observation with execution at root level")
print("="*70)
obs3 = {
    "execution": {
        "status": "SUCCESS",
        "output": "5-fold cross-validation score: 0.8500\n",
        "error": ""
    },
    "action_status": "SUCCESS"
}
score3 = extract_cv_score_from_output(obs3, execution_succeeded=True)
print(f"\nResult: {score3}\n")

print("="*70)
print("Test 4: Failed execution (should return 0.0)")
print("="*70)
obs4 = {
    "feedback": {
        "execution": {
            "status": "FAILED",
            "output": "",
            "error": "ValueError: Something went wrong"
        }
    }
}
score4 = extract_cv_score_from_output(obs4, execution_succeeded=False)
print(f"\nResult: {score4}\n")
