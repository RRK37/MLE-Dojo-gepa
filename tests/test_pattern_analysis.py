"""
Test script for pattern analysis system.
Verifies that pattern extraction, clustering, and insight generation work correctly.
"""

import sys
sys.path.insert(0, '/home/rklotins/src/MLE-Dojo-gepa')

from mledojo.agent.aide.pattern_analysis import (
    CodeFeatureExtractor,
    SolutionAnalyzer,
    PatternDiscovery,
    CorrelationAnalyzer,
    InsightGenerator,
    TemporalPatternAnalyzer
)

# Sample solution codes for testing
sample_solutions = [
    {
        "code": """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
train = pd.read_csv('./input/train.csv')
X = train.drop('target', axis=1)
y = train['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict
predictions = model.predict_proba(X_val)[:, 1]
""",
        "score": 0.72,
        "node_type": "draft"
    },
    {
        "code": """
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder

# Load data
train = pd.read_csv('./input/train.csv')
X = train.drop('target', axis=1)
y = train['target']

# Encode categorical features
for col in X.select_dtypes(include='object'):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=200)
scores = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc')

# Train final model
model.fit(X, y)
""",
        "score": 0.85,
        "node_type": "improve"
    },
    {
        "code": """
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
train = pd.read_csv('./input/train.csv')
X = train.drop('target', axis=1)
y = train['target']

# Feature engineering
X['feature_sum'] = X.sum(axis=1)
X['feature_mean'] = X.mean(axis=1)
X['feature_std'] = X.std(axis=1)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter tuning
param_grid = {
    'num_leaves': [31, 50],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [100, 200]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = lgb.LGBMClassifier()
grid = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc')
grid.fit(X_scaled, y)
""",
        "score": 0.89,
        "node_type": "improve"
    },
    {
        "code": """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Load data
train = pd.read_csv('./input/train.csv')
X = train.drop('target', axis=1)
y = train['target']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Train simple model
model = LogisticRegression()
model.fit(X_imputed, y)
""",
        "score": 0.65,
        "node_type": "draft"
    },
]

def test_pattern_analysis():
    print("=" * 80)
    print("Testing Pattern Analysis System")
    print("=" * 80)
    
    # Initialize components
    print("\n1. Initializing components...")
    analyzer = SolutionAnalyzer()
    discovery = PatternDiscovery()
    correlator = CorrelationAnalyzer()
    generator = InsightGenerator()
    temporal = TemporalPatternAnalyzer()
    print("✓ Components initialized")
    
    # Analyze solutions
    print("\n2. Analyzing solutions...")
    analyzed_solutions = []
    for i, sol in enumerate(sample_solutions):
        analysis = analyzer.analyze_solution(
            code=sol["code"],
            score=sol["score"],
            execution_time=0.0,
            node_type=sol["node_type"]
        )
        analyzed_solutions.append(analysis)
        print(f"✓ Solution {i+1}: Score={sol['score']:.2f}, Features={len(analysis['features'])}")
        print(f"  Models: {', '.join(analysis['features']['model_types'])}")
        print(f"  Preprocessing: {', '.join(analysis['features']['preprocessing'])}")
        print(f"  Validation: {', '.join(analysis['features']['validation'])}")
    
    # Cluster solutions
    print("\n3. Discovering patterns (clustering)...")
    clusters = discovery.cluster_solutions(analyzed_solutions, n_clusters=2)
    print(f"✓ Found {len(clusters)} clusters")
    for cluster_id, cluster_info in clusters.items():
        print(f"\n  Cluster {cluster_id}:")
        print(f"    Solutions: {len(cluster_info['solutions'])}")
        print(f"    Avg Score: {cluster_info['avg_score']:.3f}")
        print(f"    Std Dev: {cluster_info['std_score']:.3f}")
        print(f"    Common features: {list(cluster_info['common_features'].keys())[:5]}")
    
    # Analyze correlations
    print("\n4. Analyzing correlations...")
    patterns = correlator.analyze_clusters(clusters)
    print(f"✓ High-performing patterns: {len(patterns['high_performing_patterns'])}")
    print(f"✓ Low-performing patterns: {len(patterns['low_performing_patterns'])}")
    print(f"✓ Risky patterns: {len(patterns['risky_patterns'])}")
    print(f"✓ Stable patterns: {len(patterns['stable_patterns'])}")
    
    # Generate insights
    print("\n5. Generating insights...")
    insights = generator.generate_insights(patterns)
    guidance = generator.generate_guidance(patterns)
    print(f"✓ Generated insights ({len(insights.split('**'))-1} sections)")
    print("\nInsights Preview:")
    print("-" * 80)
    print(insights[:500] + "..." if len(insights) > 500 else insights)
    print("-" * 80)
    print(f"\n✓ Generated {len(guidance)} guidance recommendations")
    for i, g in enumerate(guidance[:3], 1):
        print(f"  {i}. {g}")
    
    # Temporal analysis
    print("\n6. Analyzing temporal patterns...")
    scores = [s['score'] for s in analyzed_solutions]
    features_list = [s['features'] for s in analyzed_solutions]
    temporal_insights = temporal.analyze_evolution(scores, features_list)
    print("✓ Temporal analysis complete")
    if temporal_insights:
        print("\nTemporal Insights:")
        print("-" * 80)
        print(temporal_insights)
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    try:
        test_pattern_analysis()
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
