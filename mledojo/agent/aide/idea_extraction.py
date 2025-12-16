"""
Idea Extraction and Knowledge Management System for AIDE

Extracts abstract concepts, patterns, and themes from solution code and plans,
stores them in a structured knowledge base, and enables semantic retrieval.
"""

import re
import ast
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import DBSCAN
import networkx as nx

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

from mledojo.agent.aide.journal import Node

logger = logging.getLogger("aide.idea_extraction")


@dataclass
class AbstractIdea:
    """Represents an abstract idea extracted from solutions"""
    id: str
    concept: str  # High-level concept name
    description: str  # Detailed description
    category: str  # Category: approach/technique/pattern/heuristic
    keywords: List[str]
    confidence: float  # 0-1 confidence score
    
    # Evidence supporting this idea
    supporting_nodes: List[str] = field(default_factory=list)  # Node IDs
    code_snippets: List[str] = field(default_factory=list)
    plan_excerpts: List[str] = field(default_factory=list)
    
    # Contextual information
    success_rate: float = 0.0  # % of solutions using this that succeeded
    avg_score_improvement: float = 0.0
    applicable_task_types: Set[str] = field(default_factory=set)
    
    # Relationships
    related_ideas: List[str] = field(default_factory=list)  # IDs of related ideas
    prerequisite_ideas: List[str] = field(default_factory=list)
    
    # Temporal information
    first_seen: Optional[int] = None  # Step number
    last_seen: Optional[int] = None
    frequency: int = 0


class CodeStructureAnalyzer:
    """Analyzes code structure to extract abstract patterns"""
    
    def __init__(self):
        self.pattern_extractors = {
            'pipeline': self._extract_pipeline_pattern,
            'ensemble': self._extract_ensemble_pattern,
            'feature_engineering': self._extract_feature_engineering,
            'validation': self._extract_validation_pattern,
            'optimization': self._extract_optimization_pattern,
            'preprocessing': self._extract_preprocessing_pattern,
            'architecture': self._extract_architecture_pattern,
        }
    
    def extract_patterns(self, code: str) -> Dict[str, Any]:
        """Extract all patterns from code"""
        patterns = {}
        
        try:
            tree = ast.parse(code)
            
            for pattern_name, extractor in self.pattern_extractors.items():
                patterns[pattern_name] = extractor(tree, code)
        except SyntaxError:
            logger.warning("Failed to parse code for pattern extraction")
        
        return patterns
    
    def _extract_pipeline_pattern(self, tree: ast.AST, code: str) -> Dict:
        """Extract data processing pipeline patterns"""
        pipeline_indicators = {
            'sklearn_pipeline': ['Pipeline', 'make_pipeline'],
            'sequential_transforms': ['fit_transform', 'transform'],
            'chained_operations': ['.pipe(', 'chain'],
        }
        
        found_patterns = []
        for pattern_type, indicators in pipeline_indicators.items():
            if any(ind in code for ind in indicators):
                found_patterns.append(pattern_type)
        
        # Detect pipeline stages
        stages = []
        if 'StandardScaler' in code or 'MinMaxScaler' in code:
            stages.append('scaling')
        if 'OneHotEncoder' in code or 'LabelEncoder' in code:
            stages.append('encoding')
        if 'PCA' in code or 'TruncatedSVD' in code:
            stages.append('dimensionality_reduction')
        
        return {
            'has_pipeline': bool(found_patterns),
            'pipeline_types': found_patterns,
            'stages': stages,
            'stage_count': len(stages)
        }
    
    def _extract_ensemble_pattern(self, tree: ast.AST, code: str) -> Dict:
        """Extract ensemble method patterns"""
        ensemble_methods = {
            'voting': ['VotingClassifier', 'VotingRegressor'],
            'stacking': ['StackingClassifier', 'StackingRegressor', 'stacking'],
            'boosting': ['XGBoost', 'LightGBM', 'CatBoost', 'GradientBoosting'],
            'bagging': ['BaggingClassifier', 'BaggingRegressor', 'RandomForest'],
            'weighted_average': ['weighted average', 'ensemble_weights'],
        }
        
        found_methods = []
        for method_type, indicators in ensemble_methods.items():
            if any(ind.lower() in code.lower() for ind in indicators):
                found_methods.append(method_type)
        
        # Count number of models
        model_count = 0
        model_indicators = ['_model', 'clf_', 'reg_', 'estimator']
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if any(ind in node.id.lower() for ind in model_indicators):
                    model_count += 1
        
        return {
            'is_ensemble': bool(found_methods),
            'ensemble_types': found_methods,
            'estimated_model_count': model_count,
            'complexity': 'high' if len(found_methods) > 1 else 'medium' if found_methods else 'low'
        }
    
    def _extract_feature_engineering(self, tree: ast.AST, code: str) -> Dict:
        """Extract feature engineering patterns"""
        techniques = {
            'polynomial': ['PolynomialFeatures', '**2', '**3'],
            'interactions': ['interaction', '*', 'multiply'],
            'aggregations': ['groupby', 'agg', 'aggregate'],
            'binning': ['cut', 'qcut', 'bin'],
            'datetime': ['dt.', 'to_datetime', 'year', 'month', 'day'],
            'text': ['TfidfVectorizer', 'CountVectorizer', 'tokenize'],
            'target_encoding': ['target_encode', 'mean_encode'],
            'embeddings': ['embedding', 'Word2Vec', 'GloVe'],
        }
        
        found_techniques = []
        for technique, indicators in techniques.items():
            if any(ind in code for ind in indicators):
                found_techniques.append(technique)
        
        # Count created features
        feature_creation_patterns = [
            r"df\[\'.*?\'\]\s*=",  # New column assignment
            r"\.assign\(",          # Using assign
            r"feature_\w+",         # Feature naming convention
        ]
        
        created_features = sum(
            len(re.findall(pattern, code)) for pattern in feature_creation_patterns
        )
        
        return {
            'has_feature_engineering': bool(found_techniques),
            'techniques': found_techniques,
            'estimated_new_features': created_features,
            'sophistication': 'high' if len(found_techniques) > 3 else 'medium' if found_techniques else 'low'
        }
    
    def _extract_validation_pattern(self, tree: ast.AST, code: str) -> Dict:
        """Extract validation strategy patterns"""
        strategies = {
            'kfold': ['KFold', 'StratifiedKFold'],
            'time_series': ['TimeSeriesSplit'],
            'group': ['GroupKFold'],
            'stratified': ['Stratified'],
            'holdout': ['train_test_split'],
            'cross_val': ['cross_val_score', 'cross_validate'],
        }
        
        found_strategies = []
        for strategy, indicators in strategies.items():
            if any(ind in code for ind in indicators):
                found_strategies.append(strategy)
        
        # Detect number of folds
        n_folds = None
        fold_match = re.search(r'n_splits?\s*=\s*(\d+)', code)
        if fold_match:
            n_folds = int(fold_match.group(1))
        
        return {
            'validation_strategies': found_strategies,
            'n_folds': n_folds,
            'has_cross_validation': any(s in found_strategies for s in ['kfold', 'time_series', 'group', 'cross_val'])
        }
    
    def _extract_optimization_pattern(self, tree: ast.AST, code: str) -> Dict:
        """Extract hyperparameter optimization patterns"""
        methods = {
            'grid_search': ['GridSearchCV'],
            'random_search': ['RandomizedSearchCV'],
            'bayesian': ['BayesianOptimization', 'Optuna', 'optuna'],
            'hyperopt': ['hyperopt', 'fmin'],
            'genetic': ['TPOT', 'genetic'],
            'manual': ['param_grid', 'params ='],
        }
        
        found_methods = []
        for method, indicators in methods.items():
            if any(ind in code for ind in indicators):
                found_methods.append(method)
        
        # Count parameters being tuned
        param_patterns = [
            r'param_grid\s*=\s*\{',
            r'search_space\s*=\s*\{',
            r'trial\.suggest',
        ]
        
        has_tuning = any(re.search(pattern, code) for pattern in param_patterns)
        
        return {
            'optimization_methods': found_methods,
            'has_hyperparameter_tuning': has_tuning,
            'sophistication': 'high' if any(m in ['bayesian', 'genetic'] for m in found_methods) else 'medium' if found_methods else 'low'
        }
    
    def _extract_preprocessing_pattern(self, tree: ast.AST, code: str) -> Dict:
        """Extract preprocessing patterns"""
        techniques = {
            'scaling': ['StandardScaler', 'MinMaxScaler', 'RobustScaler'],
            'normalization': ['Normalizer', 'normalize'],
            'encoding': ['LabelEncoder', 'OneHotEncoder', 'OrdinalEncoder'],
            'imputation': ['SimpleImputer', 'fillna', 'dropna'],
            'outlier_removal': ['IsolationForest', 'clip', 'quantile'],
            'resampling': ['resample', 'SMOTE', 'RandomOverSampler'],
        }
        
        found = []
        for technique, indicators in techniques.items():
            if any(ind in code for ind in indicators):
                found.append(technique)
        
        return {
            'preprocessing_techniques': found,
            'preprocessing_complexity': len(found)
        }
    
    def _extract_architecture_pattern(self, tree: ast.AST, code: str) -> Dict:
        """Extract model architecture patterns"""
        architectures = {
            'deep_learning': ['nn.Module', 'Sequential', 'torch.nn', 'keras'],
            'tree_based': ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'DecisionTree'],
            'linear': ['LogisticRegression', 'LinearRegression', 'Ridge', 'Lasso'],
            'svm': ['SVC', 'SVR', 'SVM'],
            'naive_bayes': ['GaussianNB', 'MultinomialNB'],
            'knn': ['KNeighbors'],
            'transformer': ['Transformer', 'BERT', 'GPT', 'attention'],
            'cnn': ['Conv', 'CNN'],
            'rnn': ['LSTM', 'GRU', 'RNN'],
        }
        
        found = []
        for arch, indicators in architectures.items():
            if any(ind in code for ind in indicators):
                found.append(arch)
        
        return {
            'architectures': found,
            'model_family': found[0] if found else 'unknown',
            'complexity': 'high' if 'deep_learning' in found else 'medium' if 'tree_based' in found else 'low'
        }


class NaturalLanguageIdeaExtractor:
    """Extracts abstract ideas from natural language plans and descriptions"""
    
    def __init__(self):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.sentence_model = None
        
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Predefined concept templates
        self.concept_templates = {
            'strategy': [
                'approach', 'strategy', 'method', 'technique', 'way to',
                'solution', 'plan', 'idea'
            ],
            'improvement': [
                'improve', 'enhance', 'optimize', 'better', 'increase',
                'boost', 'refine', 'tune'
            ],
            'technique': [
                'use', 'apply', 'implement', 'employ', 'utilize',
                'leverage', 'adopt'
            ],
            'pattern': [
                'pattern', 'structure', 'design', 'architecture',
                'framework', 'pipeline'
            ],
        }
    
    def extract_ideas_from_plan(self, plan: str, code: str = "") -> List[Dict[str, Any]]:
        """Extract abstract ideas from natural language plan"""
        ideas = []
        
        # Extract key sentences
        sentences = self._split_into_sentences(plan)
        
        # Extract concepts using multiple methods
        ideas.extend(self._extract_strategy_concepts(sentences))
        ideas.extend(self._extract_technique_mentions(sentences))
        ideas.extend(self._extract_goal_oriented_ideas(sentences))
        
        # Enrich with code context if available
        if code:
            ideas = self._enrich_with_code_context(ideas, code)
        
        return ideas
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_strategy_concepts(self, sentences: List[str]) -> List[Dict]:
        """Extract high-level strategy concepts"""
        ideas = []
        
        strategy_patterns = [
            (r'(?:use|using|employ|apply)\s+(\w+(?:\s+\w+){0,3})', 'technique'),
            (r'(?:approach|strategy|method)\s+(?:is\s+to\s+)?(\w+(?:\s+\w+){0,3})', 'strategy'),
            (r'(?:improve|enhance|optimize)\s+(\w+(?:\s+\w+){0,3})', 'improvement'),
            (r'(?:by|through)\s+(\w+(?:\s+\w+){0,3})', 'method'),
        ]
        
        for sentence in sentences:
            for pattern, category in strategy_patterns:
                matches = re.findall(pattern, sentence.lower())
                for match in matches:
                    if len(match.split()) >= 2:  # At least 2 words
                        ideas.append({
                            'concept': match.strip(),
                            'category': category,
                            'source': sentence,
                            'confidence': 0.7
                        })
        
        return ideas
    
    def _extract_technique_mentions(self, sentences: List[str]) -> List[Dict]:
        """Extract specific technique mentions"""
        known_techniques = {
            'feature engineering': ['feature engineering', 'feature creation', 'new features'],
            'ensemble': ['ensemble', 'combine models', 'multiple models', 'stacking', 'voting'],
            'hyperparameter tuning': ['hyperparameter', 'tuning', 'optimization', 'grid search'],
            'cross-validation': ['cross validation', 'cv', 'k-fold', 'validation'],
            'regularization': ['regularization', 'l1', 'l2', 'dropout'],
            'data augmentation': ['augmentation', 'augment', 'synthetic data'],
            'transfer learning': ['transfer learning', 'pretrained', 'fine-tune'],
        }
        
        ideas = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for technique, indicators in known_techniques.items():
                if any(ind in sentence_lower for ind in indicators):
                    ideas.append({
                        'concept': technique,
                        'category': 'technique',
                        'source': sentence,
                        'confidence': 0.9
                    })
        
        return ideas
    
    def _extract_goal_oriented_ideas(self, sentences: List[str]) -> List[Dict]:
        """Extract goal-oriented ideas (what the solution aims to achieve)"""
        goal_patterns = [
            r'(?:aim|goal|objective|purpose)\s+(?:is\s+)?to\s+(\w+(?:\s+\w+){0,4})',
            r'(?:in order to|to)\s+(\w+(?:\s+\w+){0,3})',
            r'(?:achieve|reach|obtain)\s+(\w+(?:\s+\w+){0,3})',
        ]
        
        ideas = []
        for sentence in sentences:
            for pattern in goal_patterns:
                matches = re.findall(pattern, sentence.lower())
                for match in matches:
                    ideas.append({
                        'concept': match.strip(),
                        'category': 'goal',
                        'source': sentence,
                        'confidence': 0.6
                    })
        
        return ideas
    
    def _enrich_with_code_context(self, ideas: List[Dict], code: str) -> List[Dict]:
        """Enrich ideas with information from code"""
        # Extract code snippets that relate to each idea
        for idea in ideas:
            concept_words = idea['concept'].split()
            # Find code lines containing concept keywords
            code_lines = code.split('\n')
            relevant_lines = [
                line for line in code_lines
                if any(word.lower() in line.lower() for word in concept_words)
            ]
            idea['code_evidence'] = relevant_lines[:3]  # Keep top 3 lines
        
        return ideas


class TopicModeler:
    """Discovers latent topics/themes across multiple solutions"""
    
    def __init__(self, n_topics: int = 10):
        self.n_topics = n_topics
        self.lda_model = None
        self.nmf_model = None
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 3)
        )
    
    def fit_topics(self, documents: List[str]) -> Dict[str, Any]:
        """Fit topic models on collection of plans/descriptions"""
        if len(documents) < self.n_topics:
            logger.warning(f"Not enough documents ({len(documents)}) for {self.n_topics} topics")
            return {}
        
        # Vectorize documents
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        
        # Fit LDA (probabilistic topic model)
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=20
        )
        lda_topics = self.lda_model.fit_transform(doc_term_matrix)
        
        # Fit NMF (non-negative matrix factorization)
        self.nmf_model = NMF(
            n_components=self.n_topics,
            random_state=42,
            max_iter=200
        )
        nmf_topics = self.nmf_model.fit_transform(doc_term_matrix)
        
        # Extract topic keywords
        feature_names = self.vectorizer.get_feature_names_out()
        
        lda_topic_words = self._get_topic_words(self.lda_model, feature_names)
        nmf_topic_words = self._get_topic_words(self.nmf_model, feature_names)
        
        return {
            'lda_topics': lda_topic_words,
            'nmf_topics': nmf_topic_words,
            'n_documents': len(documents),
            'topic_distributions': {
                'lda': lda_topics.tolist(),
                'nmf': nmf_topics.tolist()
            }
        }
    
    def _get_topic_words(self, model, feature_names, n_words: int = 10) -> List[Dict]:
        """Extract top words for each topic"""
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_weights = [topic[i] for i in top_indices]
            
            # Infer topic label
            topic_label = self._infer_topic_label(top_words)
            
            topics.append({
                'id': topic_idx,
                'label': topic_label,
                'keywords': top_words,
                'weights': top_weights
            })
        
        return topics
    
    def _infer_topic_label(self, keywords: List[str]) -> str:
        """Infer a human-readable label for a topic based on keywords"""
        # Simple heuristic-based labeling
        keyword_str = ' '.join(keywords[:5]).lower()
        
        labels = {
            'feature': 'Feature Engineering',
            'model': 'Model Selection',
            'ensemble': 'Ensemble Methods',
            'validation': 'Validation Strategy',
            'optimization': 'Hyperparameter Optimization',
            'preprocessing': 'Data Preprocessing',
            'neural': 'Deep Learning',
            'tree': 'Tree-based Methods',
        }
        
        for key, label in labels.items():
            if key in keyword_str:
                return label
        
        return f"Topic: {keywords[0]}"
    
    def get_document_topics(self, document: str, top_k: int = 3) -> List[Dict]:
        """Get top topics for a single document"""
        if self.lda_model is None:
            return []
        
        doc_vec = self.vectorizer.transform([document])
        topic_dist = self.lda_model.transform(doc_vec)[0]
        
        top_topics = []
        for topic_idx in topic_dist.argsort()[-top_k:][::-1]:
            top_topics.append({
                'topic_id': topic_idx,
                'probability': float(topic_dist[topic_idx]),
                'label': self.lda_topic_words[topic_idx]['label'] if hasattr(self, 'lda_topic_words') else f"Topic {topic_idx}"
            })
        
        return top_topics


class IdeaKnowledgeBase:
    """Stores and retrieves abstract ideas with semantic search"""
    
    def __init__(self):
        self.ideas: Dict[str, AbstractIdea] = {}
        self.idea_counter = 0
        
        # Embedding model for semantic search
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.idea_embeddings = {}
        else:
            self.encoder = None
            self.idea_embeddings = None
        
        # Graph for idea relationships
        self.idea_graph = nx.DiGraph()
        
        # Inverted index for fast keyword lookup
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Category index
        self.category_index: Dict[str, Set[str]] = defaultdict(set)
    
    def add_idea(self, idea: AbstractIdea) -> str:
        """Add a new idea to the knowledge base"""
        if not idea.id:
            idea.id = f"idea_{self.idea_counter}"
            self.idea_counter += 1
        
        self.ideas[idea.id] = idea
        
        # Update indices
        for keyword in idea.keywords:
            self.keyword_index[keyword.lower()].add(idea.id)
        
        self.category_index[idea.category].add(idea.id)
        
        # Add to graph
        self.idea_graph.add_node(idea.id, **{
            'concept': idea.concept,
            'category': idea.category,
            'confidence': idea.confidence
        })
        
        # Compute embedding if available
        if self.encoder is not None:
            embedding = self.encoder.encode(
                f"{idea.concept} {idea.description}"
            )
            self.idea_embeddings[idea.id] = embedding
        
        logger.info(f"Added idea: {idea.concept} (ID: {idea.id})")
        return idea.id
    
    def link_ideas(self, idea_id1: str, idea_id2: str, relationship: str = "related"):
        """Create a relationship between two ideas"""
        if idea_id1 in self.ideas and idea_id2 in self.ideas:
            self.idea_graph.add_edge(idea_id1, idea_id2, relationship=relationship)
            self.ideas[idea_id1].related_ideas.append(idea_id2)
    
    def search_by_keyword(self, keyword: str, top_k: int = 5) -> List[AbstractIdea]:
        """Search ideas by keyword"""
        idea_ids = self.keyword_index.get(keyword.lower(), set())
        ideas = [self.ideas[id] for id in idea_ids]
        
        # Sort by confidence and success rate
        ideas.sort(key=lambda x: (x.success_rate, x.confidence), reverse=True)
        return ideas[:top_k]
    
    def search_by_category(self, category: str) -> List[AbstractIdea]:
        """Get all ideas in a category"""
        idea_ids = self.category_index.get(category, set())
        return [self.ideas[id] for id in idea_ids]
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[AbstractIdea, float]]:
        """Search ideas by semantic similarity"""
        if self.encoder is None or not self.idea_embeddings:
            logger.warning("Semantic search not available")
            return []
        
        # Encode query
        query_embedding = self.encoder.encode(query)
        
        # Compute similarities
        similarities = {}
        for idea_id, idea_embedding in self.idea_embeddings.items():
            similarity = np.dot(query_embedding, idea_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(idea_embedding)
            )
            similarities[idea_id] = similarity
        
        # Get top-k
        top_ids = sorted(similarities, key=similarities.get, reverse=True)[:top_k]
        results = [(self.ideas[id], similarities[id]) for id in top_ids]
        
        return results
    
    def get_related_ideas(self, idea_id: str, max_depth: int = 2) -> List[AbstractIdea]:
        """Get ideas related to a given idea"""
        if idea_id not in self.idea_graph:
            return []
        
        # BFS to find related ideas
        related_ids = set()
        queue = [(idea_id, 0)]
        visited = {idea_id}
        
        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            
            for neighbor in self.idea_graph.neighbors(current_id):
                if neighbor not in visited:
                    visited.add(neighbor)
                    related_ids.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        return [self.ideas[id] for id in related_ids]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        return {
            'total_ideas': len(self.ideas),
            'categories': dict(Counter(idea.category for idea in self.ideas.values())),
            'avg_confidence': np.mean([idea.confidence for idea in self.ideas.values()]) if self.ideas else 0,
            'avg_success_rate': np.mean([idea.success_rate for idea in self.ideas.values()]) if self.ideas else 0,
            'total_relationships': self.idea_graph.number_of_edges(),
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export knowledge base to dictionary"""
        return {
            'ideas': {
                id: {
                    'concept': idea.concept,
                    'description': idea.description,
                    'category': idea.category,
                    'keywords': idea.keywords,
                    'confidence': idea.confidence,
                    'success_rate': idea.success_rate,
                    'avg_score_improvement': idea.avg_score_improvement,
                }
                for id, idea in self.ideas.items()
            },
            'statistics': self.get_statistics()
        }


class IdeaExtractor:
    """Main orchestrator for idea extraction"""
    
    def __init__(self):
        self.code_analyzer = CodeStructureAnalyzer()
        self.nl_extractor = NaturalLanguageIdeaExtractor()
        self.topic_modeler = TopicModeler()
        self.knowledge_base = IdeaKnowledgeBase()
    
    def extract_ideas_from_node(self, node: Node) -> List[AbstractIdea]:
        """Extract all ideas from a solution node"""
        ideas = []
        
        # Extract from code structure
        code_patterns = self.code_analyzer.extract_patterns(node.code)
        ideas.extend(self._patterns_to_ideas(code_patterns, node))
        
        # Extract from natural language plan
        nl_ideas = self.nl_extractor.extract_ideas_from_plan(node.plan, node.code)
        ideas.extend(self._nl_to_abstract_ideas(nl_ideas, node))
        
        return ideas
    
    def extract_ideas_from_journal(self, nodes: List[Node]) -> List[AbstractIdea]:
        """Extract ideas from multiple solution nodes"""
        all_ideas = []
        
        for node in nodes:
            ideas = self.extract_ideas_from_node(node)
            all_ideas.extend(ideas)
        
        # Discover topics across all solutions
        plans = [node.plan for node in nodes if node.plan]
        if len(plans) >= 5:
            topic_results = self.topic_modeler.fit_topics(plans)
            topic_ideas = self._topics_to_ideas(topic_results, nodes)
            all_ideas.extend(topic_ideas)
        
        # Merge similar ideas
        all_ideas = self._merge_similar_ideas(all_ideas)
        
        # Compute idea statistics
        all_ideas = self._compute_idea_statistics(all_ideas, nodes)
        
        # Store in knowledge base
        for idea in all_ideas:
            self.knowledge_base.add_idea(idea)
        
        # Create relationships between ideas
        self._create_idea_relationships()
        
        return all_ideas
    
    def _patterns_to_ideas(self, patterns: Dict, node: Node) -> List[AbstractIdea]:
        """Convert code patterns to abstract ideas"""
        ideas = []
        
        # Ensemble patterns
        if patterns.get('ensemble', {}).get('is_ensemble'):
            ensemble_types = patterns['ensemble']['ensemble_types']
            ideas.append(AbstractIdea(
                id=f"idea_ensemble_{node.id}",
                concept=f"ensemble using {', '.join(ensemble_types)}",
                description=f"Combine multiple models using {', '.join(ensemble_types)} to improve predictions",
                category="approach",
                keywords=['ensemble'] + ensemble_types,
                confidence=0.9,
                supporting_nodes=[node.id],
                code_snippets=[node.code[:200]]
            ))
        
        # Feature engineering patterns
        if patterns.get('feature_engineering', {}).get('has_feature_engineering'):
            techniques = patterns['feature_engineering']['techniques']
            ideas.append(AbstractIdea(
                id=f"idea_features_{node.id}",
                concept=f"feature engineering with {', '.join(techniques)}",
                description=f"Create new features using {', '.join(techniques)}",
                category="technique",
                keywords=['feature_engineering'] + techniques,
                confidence=0.8,
                supporting_nodes=[node.id]
            ))
        
        # Validation patterns
        if patterns.get('validation', {}).get('has_cross_validation'):
            strategies = patterns['validation']['validation_strategies']
            ideas.append(AbstractIdea(
                id=f"idea_validation_{node.id}",
                concept=f"validation with {', '.join(strategies)}",
                description=f"Use {', '.join(strategies)} for robust model evaluation",
                category="technique",
                keywords=['validation', 'cross_validation'] + strategies,
                confidence=0.85,
                supporting_nodes=[node.id]
            ))
        
        return ideas
    
    def _nl_to_abstract_ideas(self, nl_ideas: List[Dict], node: Node) -> List[AbstractIdea]:
        """Convert NL-extracted ideas to AbstractIdea objects"""
        abstract_ideas = []
        
        for i, nl_idea in enumerate(nl_ideas):
            abstract_ideas.append(AbstractIdea(
                id=f"idea_nl_{node.id}_{i}",
                concept=nl_idea['concept'],
                description=nl_idea.get('source', ''),
                category=nl_idea['category'],
                keywords=nl_idea['concept'].split(),
                confidence=nl_idea['confidence'],
                supporting_nodes=[node.id],
                plan_excerpts=[nl_idea.get('source', '')]
            ))
        
        return abstract_ideas
    
    def _topics_to_ideas(self, topic_results: Dict, nodes: List[Node]) -> List[AbstractIdea]:
        """Convert discovered topics to ideas"""
        ideas = []
        
        for topic in topic_results.get('lda_topics', []):
            ideas.append(AbstractIdea(
                id=f"topic_{topic['id']}",
                concept=topic['label'],
                description=f"Recurring theme: {', '.join(topic['keywords'][:5])}",
                category="pattern",
                keywords=topic['keywords'][:10],
                confidence=0.7,
                supporting_nodes=[node.id for node in nodes]
            ))
        
        return ideas
    
    def _merge_similar_ideas(self, ideas: List[AbstractIdea]) -> List[AbstractIdea]:
        """Merge similar ideas based on concept similarity"""
        if not ideas or self.nl_extractor.sentence_model is None:
            return ideas
        
        # Compute embeddings
        concepts = [idea.concept for idea in ideas]
        embeddings = self.nl_extractor.sentence_model.encode(concepts)
        
        # Cluster similar ideas using DBSCAN
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        # Merge ideas in same cluster
        merged = {}
        for idea, label in zip(ideas, labels):
            if label == -1:  # Noise point
                merged[idea.id] = idea
            else:
                cluster_key = f"cluster_{label}"
                if cluster_key not in merged:
                    merged[cluster_key] = idea
                else:
                    # Merge with existing idea in cluster
                    existing = merged[cluster_key]
                    existing.supporting_nodes.extend(idea.supporting_nodes)
                    existing.code_snippets.extend(idea.code_snippets)
                    existing.plan_excerpts.extend(idea.plan_excerpts)
                    existing.keywords = list(set(existing.keywords + idea.keywords))
                    existing.confidence = max(existing.confidence, idea.confidence)
                    existing.frequency += 1
        
        return list(merged.values())
    
    def _compute_idea_statistics(self, ideas: List[AbstractIdea], nodes: List[Node]) -> List[AbstractIdea]:
        """Compute success rate and score improvement for ideas"""
        node_map = {node.id: node for node in nodes}
        
        for idea in ideas:
            successful = 0
            total = len(idea.supporting_nodes)
            score_improvements = []
            
            for node_id in idea.supporting_nodes:
                if node_id in node_map:
                    node = node_map[node_id]
                    if not node.is_buggy:
                        successful += 1
                    
                    if hasattr(node, 'metric') and node.metric:
                        score_improvements.append(node.metric.value)
            
            idea.success_rate = successful / total if total > 0 else 0
            idea.avg_score_improvement = np.mean(score_improvements) if score_improvements else 0
            idea.frequency = total
        
        return ideas
    
    def _create_idea_relationships(self):
        """Create relationships between ideas based on co-occurrence"""
        idea_ids = list(self.knowledge_base.ideas.keys())
        
        for i, id1 in enumerate(idea_ids):
            idea1 = self.knowledge_base.ideas[id1]
            
            for id2 in idea_ids[i+1:]:
                idea2 = self.knowledge_base.ideas[id2]
                
                # Check if ideas co-occur in same solutions
                common_nodes = set(idea1.supporting_nodes) & set(idea2.supporting_nodes)
                
                if len(common_nodes) >= 2:
                    self.knowledge_base.link_ideas(id1, id2, "co-occurs")
    
    def get_top_ideas(self, category: Optional[str] = None, top_k: int = 10) -> List[AbstractIdea]:
        """Get top ideas by success rate and frequency"""
        if category:
            ideas = self.knowledge_base.search_by_category(category)
        else:
            ideas = list(self.knowledge_base.ideas.values())
        
        # Sort by composite score
        ideas.sort(
            key=lambda x: (x.success_rate * 0.5 + x.frequency / 10 * 0.3 + x.confidence * 0.2),
            reverse=True
        )
        
        return ideas[:top_k]
    
    def generate_insight_report(self) -> str:
        """Generate a human-readable report of extracted insights"""
        stats = self.knowledge_base.get_statistics()
        
        report = ["# Extracted Ideas and Insights\n"]
        report.append(f"Total ideas: {stats['total_ideas']}\n")
        report.append(f"Categories: {stats['categories']}\n")
        report.append(f"Average success rate: {stats['avg_success_rate']:.2%}\n\n")
        
        report.append("## Top Ideas by Category\n")
        for category in stats['categories']:
            report.append(f"\n### {category.title()}\n")
            top_ideas = self.get_top_ideas(category=category, top_k=5)
            
            for i, idea in enumerate(top_ideas, 1):
                report.append(f"{i}. **{idea.concept}**\n")
                report.append(f"   - Success rate: {idea.success_rate:.2%}\n")
                report.append(f"   - Used in {idea.frequency} solutions\n")
                report.append(f"   - Avg improvement: {idea.avg_score_improvement:.3f}\n")
        
        return ''.join(report)
