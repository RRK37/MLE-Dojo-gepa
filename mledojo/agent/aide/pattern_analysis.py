"""
Pattern analysis system for learning from solution execution patterns.
Automatically discovers what works and what doesn't across solutions.
"""

import numpy as np
import logging
from typing import Any, Dict, List, Tuple
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from mledojo.agent.aide.journal import Node

logger = logging.getLogger("aide.pattern_analysis")


class CodeFeatureExtractor:
    """Extracts features from code for pattern analysis."""
    
    def detect_model_type(self, code: str) -> List[str]:
        """Detect which ML models are used."""
        models = []
        model_patterns = {
            'xgboost': ['xgb.', 'XGBClassifier', 'XGBRegressor'],
            'lightgbm': ['lgb.', 'LGBMClassifier', 'LGBMRegressor'],
            'random_forest': ['RandomForestClassifier', 'RandomForestRegressor'],
            'neural_network': ['nn.', 'Sequential', 'torch.nn', 'keras'],
            'linear': ['LogisticRegression', 'LinearRegression', 'Ridge', 'Lasso'],
            'svm': ['SVC', 'SVR', 'SVM'],
            'knn': ['KNeighbors'],
            'naive_bayes': ['GaussianNB', 'MultinomialNB'],
            'catboost': ['CatBoost'],
            'gradient_boosting': ['GradientBoosting'],
        }
        
        for model_name, patterns in model_patterns.items():
            if any(pattern in code for pattern in patterns):
                models.append(model_name)
        
        return models if models else ['unknown']
    
    def extract_preprocessing(self, code: str) -> List[str]:
        """Extract preprocessing techniques used."""
        techniques = []
        
        preprocessing_patterns = {
            'scaling': ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'normalize'],
            'encoding': ['OneHotEncoder', 'LabelEncoder', 'get_dummies', 'OrdinalEncoder'],
            'imputation': ['SimpleImputer', 'fillna', 'dropna', 'interpolate'],
            'feature_selection': ['SelectKBest', 'RFE', 'SelectFromModel'],
            'dimensionality_reduction': ['PCA', 'TSNE', 'UMAP', 'TruncatedSVD'],
            'outlier_removal': ['IsolationForest', 'LOF', 'clip', 'quantile'],
            'augmentation': ['augment', 'transform', 'ImageDataGenerator'],
        }
        
        for technique, patterns in preprocessing_patterns.items():
            if any(pattern in code for pattern in patterns):
                techniques.append(technique)
        
        return techniques
    
    def detect_validation_strategy(self, code: str) -> str:
        """Detect validation strategy used."""
        if 'KFold' in code or 'StratifiedKFold' in code:
            return 'k-fold'
        elif 'train_test_split' in code:
            return 'train-test-split'
        elif 'cross_val_score' in code:
            return 'cross-validation'
        else:
            return 'unknown'
    
    def detect_hyperparameter_method(self, code: str) -> str:
        """Detect hyperparameter tuning method."""
        if 'GridSearchCV' in code:
            return 'grid_search'
        elif 'RandomizedSearchCV' in code:
            return 'random_search'
        elif 'Optuna' in code or 'optuna' in code:
            return 'bayesian_optimization'
        elif 'hyperopt' in code:
            return 'hyperopt'
        else:
            return 'fixed_params'
    
    def extract_keywords(self, plan: str) -> List[str]:
        """Extract important keywords from natural language plan."""
        important_words = [
            'feature engineering', 'ensemble', 'cross-validation', 'hyperparameter',
            'augmentation', 'regularization', 'boosting', 'stacking', 'optimization',
            'neural network', 'deep learning', 'transfer learning', 'pretrained',
            'tuning', 'validation', 'overfitting', 'underfitting'
        ]
        
        found_keywords = []
        plan_lower = plan.lower()
        for keyword in important_words:
            if keyword in plan_lower:
                found_keywords.append(keyword)
        
        return found_keywords


class SolutionAnalyzer:
    """Analyzes solutions to extract patterns and correlations."""
    
    def __init__(self):
        self.feature_extractor = CodeFeatureExtractor()
    
    def analyze_solution(self, node: Node) -> Dict[str, Any]:
        """Extract all features from a single solution."""
        features = {}
        
        # 1. Extract code patterns
        features['model_type'] = self.feature_extractor.detect_model_type(node.code)
        features['preprocessing'] = self.feature_extractor.extract_preprocessing(node.code)
        features['validation_strategy'] = self.feature_extractor.detect_validation_strategy(node.code)
        features['hyperparameter_method'] = self.feature_extractor.detect_hyperparameter_method(node.code)
        
        # 2. Extract plan semantics
        features['plan_keywords'] = self.feature_extractor.extract_keywords(node.plan)
        
        # 3. Code complexity
        features['code_length'] = len(node.code.split('\n'))
        features['num_imports'] = node.code.count('import')
        features['has_ensemble'] = any(word in node.code.lower() for word in ['voting', 'stacking', 'blend'])
        
        # 4. Performance
        features['score'] = node.metric.value if hasattr(node, 'metric') else 0.0
        features['is_buggy'] = node.is_buggy
        features['node_type'] = node.node_type
        
        return features
    
    def features_to_vector(self, features: Dict) -> np.ndarray:
        """Convert feature dict to numeric vector for clustering."""
        vector = []
        
        # Model type (one-hot for common models)
        all_models = ['xgboost', 'lightgbm', 'random_forest', 'neural_network', 'linear']
        for model in all_models:
            vector.append(1 if model in features['model_type'] else 0)
        
        # Preprocessing techniques (count)
        vector.append(len(features['preprocessing']))
        
        # Has ensemble
        vector.append(1 if features['has_ensemble'] else 0)
        
        # Validation strategy
        vector.append(1 if features['validation_strategy'] == 'k-fold' else 0)
        
        # Hyperparameter method
        vector.append(1 if features['hyperparameter_method'] != 'fixed_params' else 0)
        
        # Complexity (normalized)
        vector.append(min(features['code_length'] / 1000.0, 1.0))
        
        # Number of keywords
        vector.append(len(features['plan_keywords']) / 10.0)
        
        return np.array(vector)


class PatternDiscovery:
    """Discovers patterns across solutions."""
    
    def __init__(self, n_clusters: int = 5):
        self.analyzer = SolutionAnalyzer()
        self.n_clusters = n_clusters
        self.clusterer = None
    
    def cluster_solutions(self, solutions: List[Node]) -> Dict[int, List[Node]]:
        """Cluster solutions by similarity."""
        if len(solutions) < self.n_clusters:
            # Not enough solutions, return single cluster
            return {0: solutions}
        
        # Extract features from all solutions
        features_list = []
        valid_solutions = []
        
        for node in solutions:
            try:
                features = self.analyzer.analyze_solution(node)
                vec = self.analyzer.features_to_vector(features)
                features_list.append(vec)
                valid_solutions.append(node)
            except Exception as e:
                logger.warning(f"Failed to extract features from node {node.id}: {e}")
                continue
        
        if len(valid_solutions) < 2:
            return {0: valid_solutions}
        
        # Cluster
        X = np.array(features_list)
        n_clusters = min(self.n_clusters, len(valid_solutions))
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.clusterer.fit_predict(X)
        
        # Group by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(valid_solutions[idx])
        
        return clusters


class CorrelationAnalyzer:
    """Analyzes correlations between patterns and scores."""
    
    def __init__(self):
        self.analyzer = SolutionAnalyzer()
    
    def analyze_clusters(self, clusters: Dict[int, List[Node]]) -> Dict[str, Any]:
        """Analyze each cluster to find success patterns."""
        insights = {
            'high_performing_patterns': [],
            'low_performing_patterns': [],
            'risky_patterns': [],
            'stable_patterns': []
        }
        
        for cluster_id, nodes in clusters.items():
            if not nodes:
                continue
            
            analysis = self._analyze_cluster(nodes)
            
            # Categorize based on performance
            if analysis['avg_score'] > 0.6:
                insights['high_performing_patterns'].append(analysis)
            elif analysis['avg_score'] < 0.4:
                insights['low_performing_patterns'].append(analysis)
            
            if analysis['failure_rate'] > 0.4:
                insights['risky_patterns'].append(analysis)
            elif analysis['failure_rate'] < 0.2 and len(nodes) >= 2:
                insights['stable_patterns'].append(analysis)
        
        # Sort by performance
        insights['high_performing_patterns'].sort(key=lambda x: x['avg_score'], reverse=True)
        insights['low_performing_patterns'].sort(key=lambda x: x['avg_score'])
        
        return insights
    
    def _analyze_cluster(self, nodes: List[Node]) -> Dict[str, Any]:
        """Analyze a single cluster."""
        scores = [n.metric.value for n in nodes if hasattr(n, 'metric') and n.metric.value > 0]
        failures = [n.is_buggy for n in nodes]
        
        # Extract common features
        all_features = [self.analyzer.analyze_solution(n) for n in nodes]
        
        # Find most common characteristics
        model_types = [f['model_type'] for f in all_features]
        all_models = sum(model_types, [])
        most_common_model = max(set(all_models), key=all_models.count) if all_models else 'unknown'
        
        preprocessing_steps = [f['preprocessing'] for f in all_features]
        common_preprocessing = self._find_common_elements(preprocessing_steps)
        
        validation_strategies = [f['validation_strategy'] for f in all_features]
        most_common_validation = max(set(validation_strategies), key=validation_strategies.count)
        
        return {
            'cluster_size': len(nodes),
            'avg_score': np.mean(scores) if scores else 0.0,
            'std_score': np.std(scores) if scores else 0.0,
            'failure_rate': np.mean(failures),
            'most_common_model': most_common_model,
            'common_preprocessing': common_preprocessing,
            'validation_strategy': most_common_validation,
            'representative_plan': nodes[0].plan[:150] if nodes else "",
        }
    
    def _find_common_elements(self, lists: List[List[str]]) -> List[str]:
        """Find elements that appear in most lists."""
        if not lists:
            return []
        
        all_elements = sum(lists, [])
        if not all_elements:
            return []
        
        counts = Counter(all_elements)
        threshold = max(len(lists) * 0.3, 1)  # Appear in 30%+ of solutions
        return [elem for elem, count in counts.items() if count >= threshold]


class InsightGenerator:
    """Generates human-readable insights from analysis."""
    
    def generate_insights(self, correlation_analysis: Dict) -> str:
        """Generate textual insights to inject into prompts."""
        insights = []
        
        # High-performing patterns
        if correlation_analysis['high_performing_patterns']:
            insights.append("**High-Performing Patterns Discovered:**")
            for pattern in correlation_analysis['high_performing_patterns'][:3]:
                preprocessing_str = ', '.join(pattern['common_preprocessing']) if pattern['common_preprocessing'] else 'minimal preprocessing'
                insight = f"- Solutions using **{pattern['most_common_model']}** with {preprocessing_str} achieved average score of **{pattern['avg_score']:.3f}**"
                insights.append(insight)
        
        # Low-performing patterns
        if correlation_analysis['low_performing_patterns']:
            insights.append("\n**Patterns to Avoid:**")
            for pattern in correlation_analysis['low_performing_patterns'][:2]:
                preprocessing_str = ', '.join(pattern['common_preprocessing']) if pattern['common_preprocessing'] else 'minimal preprocessing'
                insight = f"- Solutions using {pattern['most_common_model']} with {preprocessing_str} only achieved {pattern['avg_score']:.3f}"
                insights.append(insight)
        
        # Risky patterns
        if correlation_analysis['risky_patterns']:
            insights.append("\n**Risky Approaches (High Failure Rate):**")
            for pattern in correlation_analysis['risky_patterns'][:2]:
                insight = f"- {pattern['most_common_model']}-based approaches failed {pattern['failure_rate']*100:.0f}% of the time"
                insights.append(insight)
        
        # Stable patterns
        if correlation_analysis['stable_patterns']:
            insights.append("\n**Reliable Approaches:**")
            for pattern in correlation_analysis['stable_patterns'][:2]:
                preprocessing_str = ', '.join(pattern['common_preprocessing']) if pattern['common_preprocessing'] else 'standard preprocessing'
                insight = f"- {pattern['most_common_model']} with {preprocessing_str} has {(1-pattern['failure_rate'])*100:.0f}% success rate"
                insights.append(insight)
        
        if not insights:
            return ""
        
        return "\n".join(insights)
    
    def generate_guidance(self, correlation_analysis: Dict) -> List[str]:
        """Generate specific guidance based on analysis."""
        guidance = []
        
        if correlation_analysis['high_performing_patterns']:
            best_pattern = correlation_analysis['high_performing_patterns'][0]
            guidance.append(f"Consider using {best_pattern['most_common_model']} as it has shown the best results so far")
        
        if correlation_analysis['risky_patterns']:
            guidance.append("Be cautious with risky approaches unless you have a good reason and can mitigate the failure risk")
        
        if correlation_analysis['stable_patterns']:
            guidance.append("Build upon reliable approaches as a foundation before trying more experimental techniques")
        
        guidance.append("Focus on approaches that have demonstrated success rather than repeating failed patterns")
        
        return guidance


class TemporalPatternAnalyzer:
    """Analyzes how patterns change over time."""
    
    def __init__(self):
        self.analyzer = SolutionAnalyzer()
    
    def analyze_evolution(self, nodes: List[Node]) -> Dict[str, Any]:
        """See how solution characteristics evolve."""
        if len(nodes) < 3:
            return {'insights': [], 'has_stagnation': False, 'has_low_diversity': False}
        
        nodes_by_step = sorted([n for n in nodes if hasattr(n, 'step')], key=lambda n: n.step)
        
        # Track metrics over time
        scores_over_time = []
        model_types_over_time = []
        
        for node in nodes_by_step:
            if hasattr(node, 'metric') and node.metric.value > 0:
                scores_over_time.append(node.metric.value)
            
            try:
                features = self.analyzer.analyze_solution(node)
                model_types_over_time.append(features['model_type'])
            except:
                continue
        
        insights = []
        has_stagnation = False
        has_low_diversity = False
        
        # Detect stagnation
        if len(scores_over_time) >= 5:
            recent_scores = scores_over_time[-5:]
            score_variance = np.var(recent_scores)
            if score_variance < 0.001:  # Very low variance
                insights.append("⚠️ Scores have stagnated - consider trying a radically different approach")
                has_stagnation = True
        
        # Detect over-reliance on one model type
        if len(model_types_over_time) >= 5:
            recent_models = model_types_over_time[-5:]
            unique_models = set(sum(recent_models, []))
            if len(unique_models) <= 1:
                insights.append("⚠️ All recent solutions use the same model type - consider diversifying")
                has_low_diversity = True
        
        return {
            'insights': insights,
            'scores_trend': scores_over_time,
            'has_stagnation': has_stagnation,
            'has_low_diversity': has_low_diversity
        }
