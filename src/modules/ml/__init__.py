"""
Data Copilot Lab - Machine Learning Module
Comprehensive ML tools including feature engineering, model training, AutoML, and explainability
"""

from src.modules.ml.feature_engineering import (
    FeatureEngineer,
    SelectionMethod,
    ExtractionMethod,
)
from src.modules.ml.classification import (
    ClassificationTrainer,
    ClassifierType,
)
from src.modules.ml.regression import (
    RegressionTrainer,
    RegressorType,
)
from src.modules.ml.clustering import (
    ClusteringTrainer,
    ClusteringMethod,
)
from src.modules.ml.automl import (
    AutoML,
    TaskType,
)
from src.modules.ml.explainability import (
    ModelExplainer,
)
from src.modules.ml.model_registry import (
    ModelRegistry,
)

__all__ = [
    # Feature Engineering
    "FeatureEngineer",
    "SelectionMethod",
    "ExtractionMethod",
    # Classification
    "ClassificationTrainer",
    "ClassifierType",
    # Regression
    "RegressionTrainer",
    "RegressorType",
    # Clustering
    "ClusteringTrainer",
    "ClusteringMethod",
    # AutoML
    "AutoML",
    "TaskType",
    # Explainability
    "ModelExplainer",
    # Model Registry
    "ModelRegistry",
]
