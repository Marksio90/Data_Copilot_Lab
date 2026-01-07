"""
Data Copilot Lab - Smart Suggestions
Intelligent suggestions for data science workflows
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.modules.ai_assistant.llm_integration import LLMIntegration, LLMModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SuggestionCategory(str, Enum):
    """Suggestion categories"""
    DATA_QUALITY = "data_quality"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_SELECTION = "model_selection"
    HYPERPARAMETER = "hyperparameter"
    EVALUATION = "evaluation"
    OPTIMIZATION = "optimization"
    NEXT_STEPS = "next_steps"


class SmartSuggestions:
    """
    Intelligent suggestions for data science workflows

    Provides context-aware suggestions for:
    - Data quality improvements
    - Feature engineering ideas
    - Model selection
    - Hyperparameter tuning
    - Next analysis steps
    - Code improvements
    """

    def __init__(
        self,
        llm_model: LLMModel = LLMModel.GPT35_TURBO,
        api_key: Optional[str] = None,
        use_llm: bool = True
    ):
        """
        Initialize Smart Suggestions

        Args:
            llm_model: LLM model to use
            api_key: API key
            use_llm: Use LLM for enhanced suggestions
        """
        self.logger = logger
        self.use_llm = use_llm

        if use_llm:
            self.llm = LLMIntegration(model=llm_model, api_key=api_key, temperature=0.6)

    def suggest_data_quality_improvements(
        self,
        data: pd.DataFrame,
        analysis_results: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest data quality improvements

        Args:
            data: DataFrame
            analysis_results: Results from data quality analysis

        Returns:
            List of suggestions
        """
        self.logger.info("Generating data quality suggestions")

        suggestions = []

        # Rule-based suggestions
        missing_pct = (data.isnull().sum().sum() / data.size) * 100
        if missing_pct > 5:
            suggestions.append({
                "category": "data_quality",
                "priority": "high",
                "issue": f"High missing data ({missing_pct:.1f}%)",
                "suggestion": "Consider imputation strategies: median for numeric, mode for categorical, or advanced methods like KNN imputation",
                "action": "apply_missing_data_handler"
            })

        duplicate_pct = (data.duplicated().sum() / len(data)) * 100
        if duplicate_pct > 1:
            suggestions.append({
                "category": "data_quality",
                "priority": "medium",
                "issue": f"Duplicates detected ({duplicate_pct:.1f}%)",
                "suggestion": "Review and remove duplicate rows to avoid data leakage",
                "action": "remove_duplicates"
            })

        # Check for constant columns
        for col in data.select_dtypes(include=[np.number]).columns:
            if data[col].nunique() == 1:
                suggestions.append({
                    "category": "data_quality",
                    "priority": "medium",
                    "issue": f"Column '{col}' has constant value",
                    "suggestion": f"Remove constant column '{col}' as it provides no information",
                    "action": f"drop_column:{col}"
                })

        # LLM-enhanced suggestions
        if self.use_llm and analysis_results:
            llm_suggestions = self._get_llm_suggestions(
                "data quality",
                {"data_shape": data.shape, "analysis": analysis_results}
            )
            suggestions.extend(llm_suggestions)

        return suggestions

    def suggest_feature_engineering(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        task_type: str = "classification"
    ) -> List[Dict[str, Any]]:
        """
        Suggest feature engineering techniques

        Args:
            data: DataFrame
            target: Target variable
            task_type: 'classification' or 'regression'

        Returns:
            List of feature engineering suggestions
        """
        self.logger.info("Generating feature engineering suggestions")

        suggestions = []

        # Datetime features
        datetime_cols = data.select_dtypes(include=['datetime']).columns
        if len(datetime_cols) > 0:
            suggestions.append({
                "category": "feature_engineering",
                "priority": "high",
                "technique": "datetime_features",
                "suggestion": f"Extract features from datetime columns: {', '.join(datetime_cols)}",
                "example": "year, month, day, dayofweek, is_weekend, hour",
                "action": "extract_date_features"
            })

        # Polynomial features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if target and target in numeric_cols:
            numeric_cols = numeric_cols.drop(target)

        if len(numeric_cols) > 1 and len(numeric_cols) <= 5:
            suggestions.append({
                "category": "feature_engineering",
                "priority": "medium",
                "technique": "polynomial_features",
                "suggestion": f"Create polynomial and interaction features from {len(numeric_cols)} numeric columns",
                "example": "x1*x2, x1^2, etc.",
                "action": "create_polynomial_features"
            })

        # Binning for continuous features
        for col in numeric_cols[:3]:  # Check first 3 numeric columns
            if data[col].nunique() > 50:
                suggestions.append({
                    "category": "feature_engineering",
                    "priority": "low",
                    "technique": "binning",
                    "suggestion": f"Consider binning continuous column '{col}' for non-linear relationships",
                    "example": "Create 5-10 bins using quantiles",
                    "action": f"bin_feature:{col}"
                })

        # Scaling
        if len(numeric_cols) > 0:
            ranges = data[numeric_cols].max() - data[numeric_cols].min()
            if ranges.max() / ranges.min() > 100:
                suggestions.append({
                    "category": "feature_engineering",
                    "priority": "high",
                    "technique": "scaling",
                    "suggestion": "Features have very different scales. Apply standardization or normalization",
                    "example": "StandardScaler or MinMaxScaler",
                    "action": "scale_features"
                })

        # Encoding categorical
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                n_unique = data[col].nunique()
                if n_unique <= 10:
                    suggestions.append({
                        "category": "feature_engineering",
                        "priority": "high",
                        "technique": "encoding",
                        "suggestion": f"Encode categorical column '{col}' ({n_unique} categories)",
                        "example": "OneHotEncoding for low cardinality",
                        "action": f"encode_categorical:{col}"
                    })
                elif n_unique > 50:
                    suggestions.append({
                        "category": "feature_engineering",
                        "priority": "medium",
                        "technique": "encoding",
                        "suggestion": f"High cardinality column '{col}' ({n_unique} categories). Consider target encoding or frequency encoding",
                        "example": "TargetEncoder or FrequencyEncoder",
                        "action": f"encode_high_cardinality:{col}"
                    })

        return suggestions

    def suggest_models(
        self,
        data_info: Dict[str, Any],
        task_type: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest appropriate models

        Args:
            data_info: Information about the data
            task_type: 'classification', 'regression', or 'clustering'
            constraints: Constraints (training_time, interpretability, etc.)

        Returns:
            List of model suggestions
        """
        self.logger.info(f"Suggesting models for {task_type}")

        suggestions = []
        n_samples = data_info.get('n_samples', 0)
        n_features = data_info.get('n_features', 0)

        if task_type == "classification":
            models = self._suggest_classification_models(n_samples, n_features, constraints)
        elif task_type == "regression":
            models = self._suggest_regression_models(n_samples, n_features, constraints)
        elif task_type == "clustering":
            models = self._suggest_clustering_models(n_samples, n_features, constraints)
        else:
            models = []

        for model in models:
            suggestions.append({
                "category": "model_selection",
                "model": model["name"],
                "priority": model["priority"],
                "rationale": model["rationale"],
                "pros": model["pros"],
                "cons": model["cons"],
                "action": f"train_model:{model['name']}"
            })

        return suggestions

    def suggest_hyperparameters(
        self,
        model_type: str,
        data_size: int,
        optimization_goal: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Suggest hyperparameter ranges for tuning

        Args:
            model_type: Type of model
            data_size: Number of samples
            optimization_goal: What to optimize for

        Returns:
            Hyperparameter suggestions
        """
        self.logger.info(f"Suggesting hyperparameters for {model_type}")

        # Default parameter grids
        param_suggestions = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "rationale": "Good starting points for random forest"
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.3],
                "max_depth": [3, 5, 7],
                "rationale": "Balanced exploration of XGBoost parameter space"
            },
            "logistic_regression": {
                "C": [0.001, 0.01, 0.1, 1, 10],
                "penalty": ['l1', 'l2'],
                "rationale": "Regularization strength and type"
            }
        }

        # Adjust based on data size
        if data_size > 100000:  # Large dataset
            if model_type == "random_forest":
                param_suggestions["random_forest"]["n_estimators"] = [100, 200]
                param_suggestions["random_forest"]["max_depth"] = [10, 20]

        return param_suggestions.get(model_type, {
            "message": f"No default suggestions for {model_type}",
            "recommendation": "Use RandomizedSearchCV with broad ranges"
        })

    def suggest_next_steps(
        self,
        workflow_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Suggest next steps in analysis workflow

        Args:
            workflow_state: Current state of analysis

        Returns:
            List of next step suggestions
        """
        self.logger.info("Suggesting next steps")

        suggestions = []
        completed_steps = workflow_state.get('completed_steps', [])

        # Define workflow order
        typical_workflow = [
            ("data_loading", "Load and inspect data"),
            ("data_cleaning", "Clean data (missing values, outliers, duplicates)"),
            ("eda", "Exploratory Data Analysis"),
            ("feature_engineering", "Feature engineering and selection"),
            ("model_training", "Train baseline models"),
            ("model_evaluation", "Evaluate and compare models"),
            ("hyperparameter_tuning", "Tune hyperparameters of best model"),
            ("final_evaluation", "Final evaluation on test set"),
            ("model_deployment", "Save model for deployment")
        ]

        # Find next uncompleted steps
        for step, description in typical_workflow:
            if step not in completed_steps:
                suggestions.append({
                    "category": "next_steps",
                    "step": step,
                    "description": description,
                    "priority": "high" if len(suggestions) == 0 else "medium"
                })

                if len(suggestions) >= 3:  # Return top 3 suggestions
                    break

        # LLM-enhanced suggestions
        if self.use_llm:
            context = f"Completed steps: {', '.join(completed_steps)}"
            llm_suggestions = self._get_llm_suggestions("next_steps", {"workflow": context})
            suggestions.extend(llm_suggestions[:2])  # Add top 2 LLM suggestions

        return suggestions

    def _suggest_classification_models(
        self,
        n_samples: int,
        n_features: int,
        constraints: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest classification models"""
        models = []

        # Logistic Regression
        models.append({
            "name": "logistic_regression",
            "priority": "high",
            "rationale": "Fast, interpretable baseline",
            "pros": ["Fast training", "Interpretable", "Works well with linear relationships"],
            "cons": ["Assumes linearity", "May underfit complex patterns"]
        })

        # Random Forest
        models.append({
            "name": "random_forest",
            "priority": "high",
            "rationale": "Robust, handles non-linearity, provides feature importance",
            "pros": ["Handles non-linear relationships", "Feature importance", "Robust to outliers"],
            "cons": ["Can be slow on large datasets", "Less interpretable"]
        })

        # XGBoost (if enough data)
        if n_samples > 100:
            models.append({
                "name": "xgboost",
                "priority": "high",
                "rationale": "State-of-the-art performance, handles missing values",
                "pros": ["Excellent performance", "Built-in regularization", "Handles missing data"],
                "cons": ["Requires tuning", "Can overfit", "Less interpretable"]
            })

        # SVM (if small-medium dataset)
        if n_samples < 10000:
            models.append({
                "name": "svm",
                "priority": "medium",
                "rationale": "Effective in high-dimensional spaces",
                "pros": ["Effective for high-dimensional data", "Memory efficient"],
                "cons": ["Slow for large datasets", "Requires feature scaling"]
            })

        return models

    def _suggest_regression_models(
        self,
        n_samples: int,
        n_features: int,
        constraints: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest regression models"""
        models = []

        models.append({
            "name": "linear_regression",
            "priority": "high",
            "rationale": "Simple baseline, highly interpretable",
            "pros": ["Fast", "Interpretable", "Works well for linear relationships"],
            "cons": ["Assumes linearity", "Sensitive to outliers"]
        })

        models.append({
            "name": "ridge",
            "priority": "high",
            "rationale": "Regularized linear model, handles multicollinearity",
            "pros": ["Handles multicollinearity", "Prevents overfitting"],
            "cons": ["Still assumes linearity"]
        })

        models.append({
            "name": "random_forest",
            "priority": "high",
            "rationale": "Handles non-linearity, robust",
            "pros": ["Non-linear", "Feature importance", "Robust"],
            "cons": ["Can be slow", "Less interpretable"]
        })

        if n_samples > 100:
            models.append({
                "name": "xgboost",
                "priority": "high",
                "rationale": "Top performance for structured data",
                "pros": ["Excellent performance", "Regularization", "Fast"],
                "cons": ["Requires tuning", "Less interpretable"]
            })

        return models

    def _suggest_clustering_models(
        self,
        n_samples: int,
        n_features: int,
        constraints: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest clustering models"""
        models = []

        models.append({
            "name": "kmeans",
            "priority": "high",
            "rationale": "Fast, simple, works well for spherical clusters",
            "pros": ["Fast", "Simple", "Scales well"],
            "cons": ["Assumes spherical clusters", "Need to specify k"]
        })

        models.append({
            "name": "dbscan",
            "priority": "medium",
            "rationale": "Finds arbitrary-shaped clusters, handles noise",
            "pros": ["Finds arbitrary shapes", "Identifies noise", "No need to specify k"],
            "cons": ["Sensitive to parameters", "Struggles with varying densities"]
        })

        models.append({
            "name": "hierarchical",
            "priority": "medium",
            "rationale": "Creates hierarchy, good for visualization",
            "pros": ["Hierarchical structure", "Good visualization", "No need to specify k upfront"],
            "cons": ["Slow for large datasets", "Sensitive to outliers"]
        })

        return models

    def _get_llm_suggestions(
        self,
        suggestion_type: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get LLM-enhanced suggestions"""
        if not self.use_llm:
            return []

        system_message = f"""You are a data science expert. Provide {suggestion_type} suggestions.
Return a JSON array of suggestions with fields: category, priority, suggestion, rationale."""

        prompt = f"Context:\n{context}\n\nProvide {suggestion_type} suggestions:"

        try:
            response = self.llm.generate_text(prompt, system_message=system_message)
            # Try to parse JSON
            import json
            suggestions = json.loads(response)
            if isinstance(suggestions, dict) and 'suggestions' in suggestions:
                suggestions = suggestions['suggestions']
            return suggestions if isinstance(suggestions, list) else []
        except Exception as e:
            self.logger.debug(f"LLM suggestion parsing failed: {e}")
            return []
