"""
Test script to verify CV score extraction logic works correctly.
"""

import re

def extract_cv_score_from_stdout(stdout_text: str) -> float:
    """Test version of the CV score extraction."""
    if not stdout_text:
        return 0.0
    
    print(f"Searching for CV score in stdout (length: {len(stdout_text)})")
    print(f"Stdout preview: {stdout_text[:200]}")
    
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
                print(f"✓ Extracted CV score: {cv_score:.4f} using pattern: {pattern}")
                return cv_score
    
    print(f"No CV score pattern matched")
    return 0.0


# Test with actual output from the logs
test_cases = [
    "Cross-validated accuracy: 0.8283",
    "Cross-validated accuracy: 0.8216",
    "5-fold cross-validation accuracy: 0.8137",
    "CV score: 0.7892",
    "Validation accuracy: 0.8500",
    "Test score: 0.9123",
]

print("="*60)
print("Testing CV Score Extraction")
print("="*60)

for i, test_input in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test_input}")
    print("-" * 40)
    score = extract_cv_score_from_stdout(test_input)
    print(f"Result: {score}")
    print()
