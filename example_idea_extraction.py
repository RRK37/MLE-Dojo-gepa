"""
Example script demonstrating the Idea Extraction System

This shows how abstract ideas are extracted from solutions and stored
in a knowledge base for later retrieval and reuse.
"""

from mledojo.agent.aide.idea_extraction import (
    IdeaExtractor,
    AbstractIdea,
    CodeStructureAnalyzer,
    NaturalLanguageIdeaExtractor
)
from mledojo.agent.aide.journal import Node

# Example solutions (simplified)
example_solutions = [
    {
        'plan': "Use XGBoost with feature engineering including polynomial features and target encoding to improve predictions",
        'code': """
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold

# Feature engineering
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Target encoding
target_means = train.groupby('category')['target'].mean()
train['target_encoded'] = train['category'].map(target_means)

# XGBoost with cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = xgb.XGBRegressor(n_estimators=1000, max_depth=6)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_poly)):
    model.fit(X_poly[train_idx], y[train_idx])
    preds = model.predict(X_poly[val_idx])
""",
        'score': 0.85
    },
    {
        'plan': "Implement ensemble of LightGBM and RandomForest with weighted averaging based on validation performance",
        'code': """
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

# Train LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=500)
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_val)

# Train RandomForest
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)

# Weighted ensemble
ensemble_preds = 0.6 * lgb_preds + 0.4 * rf_preds
""",
        'score': 0.88
    },
    {
        'plan': "Use stacking ensemble with neural network as meta-learner, combining tree-based base models",
        'code': """
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor

# Base models
base_models = [
    ('xgb', xgb.XGBRegressor(n_estimators=500)),
    ('lgb', lgb.LGBMRegressor(n_estimators=500)),
]

# Meta learner
meta_learner = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu')

# Stacking
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5
)
stacking_model.fit(X_train, y_train)
""",
        'score': 0.91
    },
    {
        'plan': "Apply extensive feature engineering with datetime features, aggregations, and interactions before using CatBoost",
        'code': """
import catboost as cb
import pandas as pd

# Datetime features
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month
df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek

# Aggregation features
agg_features = df.groupby('category').agg({
    'value': ['mean', 'std', 'min', 'max'],
    'count': 'sum'
}).reset_index()

# Interaction features
df['feature1_x_feature2'] = df['feature1'] * df['feature2']
df['feature1_div_feature3'] = df['feature1'] / (df['feature3'] + 1)

# CatBoost
model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.05)
model.fit(X_train, y_train, cat_features=['category'])
""",
        'score': 0.87
    }
]


def demo_code_pattern_extraction():
    """Demonstrate code pattern extraction"""
    print("=" * 80)
    print("DEMO 1: Code Pattern Extraction")
    print("=" * 80)
    
    analyzer = CodeStructureAnalyzer()
    
    for i, solution in enumerate(example_solutions, 1):
        print(f"\n--- Solution {i} ---")
        print(f"Plan: {solution['plan'][:80]}...")
        
        patterns = analyzer.extract_patterns(solution['code'])
        
        print("\nExtracted Patterns:")
        if patterns.get('ensemble', {}).get('is_ensemble'):
            print(f"  ✓ Ensemble: {patterns['ensemble']['ensemble_types']}")
        
        if patterns.get('feature_engineering', {}).get('has_feature_engineering'):
            print(f"  ✓ Feature Engineering: {patterns['feature_engineering']['techniques']}")
        
        if patterns.get('validation', {}).get('has_cross_validation'):
            print(f"  ✓ Cross-Validation: {patterns['validation']['validation_strategies']}")
        
        if patterns['architecture']['architectures']:
            print(f"  ✓ Model Architecture: {patterns['architecture']['architectures']}")


def demo_nl_idea_extraction():
    """Demonstrate natural language idea extraction"""
    print("\n" + "=" * 80)
    print("DEMO 2: Natural Language Idea Extraction")
    print("=" * 80)
    
    extractor = NaturalLanguageIdeaExtractor()
    
    for i, solution in enumerate(example_solutions, 1):
        print(f"\n--- Solution {i} ---")
        print(f"Plan: {solution['plan']}")
        
        ideas = extractor.extract_ideas_from_plan(solution['plan'], solution['code'])
        
        print("\nExtracted Ideas:")
        for idea in ideas:
            print(f"  • {idea['concept']} (category: {idea['category']}, confidence: {idea['confidence']:.2f})")


def demo_full_idea_extraction():
    """Demonstrate full idea extraction pipeline"""
    print("\n" + "=" * 80)
    print("DEMO 3: Full Idea Extraction & Knowledge Base")
    print("=" * 80)
    
    # Create nodes from example solutions
    nodes = []
    for i, solution in enumerate(example_solutions):
        node = Node(
            code=solution['code'],
            plan=solution['plan'],
            step=i
        )
        node.is_buggy = False
        node.metric = type('Metric', (), {'value': solution['score']})()
        nodes.append(node)
    
    # Extract ideas
    extractor = IdeaExtractor()
    print("\nExtracting ideas from all solutions...")
    ideas = extractor.extract_ideas_from_journal(nodes)
    
    print(f"\nExtracted {len(ideas)} unique ideas!")
    
    # Show statistics
    stats = extractor.knowledge_base.get_statistics()
    print(f"\nKnowledge Base Statistics:")
    print(f"  Total ideas: {stats['total_ideas']}")
    print(f"  Categories: {stats['categories']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
    print(f"  Avg success rate: {stats['avg_success_rate']:.2%}")
    
    # Show top ideas
    print("\n" + "-" * 80)
    print("Top Ideas by Category:")
    print("-" * 80)
    
    for category in ['approach', 'technique', 'pattern']:
        top_ideas = extractor.get_top_ideas(category=category, top_k=3)
        if top_ideas:
            print(f"\n{category.upper()}:")
            for i, idea in enumerate(top_ideas, 1):
                print(f"  {i}. {idea.concept}")
                print(f"     Success: {idea.success_rate:.0%} | Frequency: {idea.frequency} | Confidence: {idea.confidence:.2f}")


def demo_semantic_search():
    """Demonstrate semantic search for ideas"""
    print("\n" + "=" * 80)
    print("DEMO 4: Semantic Search for Ideas")
    print("=" * 80)
    
    # Create and populate knowledge base
    nodes = [Node(code=s['code'], plan=s['plan'], step=i) for i, s in enumerate(example_solutions)]
    for node, solution in zip(nodes, example_solutions):
        node.is_buggy = False
        node.metric = type('Metric', (), {'value': solution['score']})()
    
    extractor = IdeaExtractor()
    extractor.extract_ideas_from_journal(nodes)
    
    # Search queries
    queries = [
        "How to combine multiple models?",
        "Feature engineering techniques",
        "Cross-validation strategies"
    ]
    
    print("\nSearching for relevant ideas:")
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = extractor.knowledge_base.semantic_search(query, top_k=3)
        
        if results:
            for idea, similarity in results:
                print(f"  • {idea.concept} (similarity: {similarity:.3f})")
        else:
            print("  (Semantic search requires sentence-transformers library)")


def demo_idea_report():
    """Generate a full insight report"""
    print("\n" + "=" * 80)
    print("DEMO 5: Automated Insight Report")
    print("=" * 80)
    
    # Create and populate knowledge base
    nodes = [Node(code=s['code'], plan=s['plan'], step=i) for i, s in enumerate(example_solutions)]
    for node, solution in zip(nodes, example_solutions):
        node.is_buggy = False
        node.metric = type('Metric', (), {'value': solution['score']})()
    
    extractor = IdeaExtractor()
    extractor.extract_ideas_from_journal(nodes)
    
    # Generate report
    report = extractor.generate_insight_report()
    print("\n" + report)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("IDEA EXTRACTION SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("\nThis demonstrates how the system extracts abstract ideas from solutions,")
    print("stores them in a knowledge base, and enables intelligent retrieval.\n")
    
    # Run all demos
    demo_code_pattern_extraction()
    demo_nl_idea_extraction()
    demo_full_idea_extraction()
    demo_semantic_search()
    demo_idea_report()
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nThe idea extraction system can:")
    print("  1. Extract patterns from code structure (ensemble, validation, etc.)")
    print("  2. Extract concepts from natural language plans")
    print("  3. Discover themes across multiple solutions using topic modeling")
    print("  4. Store ideas in a searchable knowledge base")
    print("  5. Track success rates and score improvements for each idea")
    print("  6. Enable semantic search for relevant ideas")
    print("  7. Generate automated insight reports")
    print("\nThese ideas can be injected back into prompts to guide future solutions!")
