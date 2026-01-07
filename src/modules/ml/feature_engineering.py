"""
Data Copilot Lab - Feature Engineering
Advanced feature engineering tools for ML pipelines
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
    f_classif,
    f_regression,
    chi2,
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from src.core.exceptions import InvalidParameterError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SelectionMethod(str, Enum):
    """Feature selection methods"""
    VARIANCE = "variance"
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    F_TEST = "f_test"
    CHI2 = "chi2"
    RFE = "rfe"
    L1_REGULARIZATION = "l1"


class ExtractionMethod(str, Enum):
    """Dimensionality reduction methods"""
    PCA = "pca"
    LDA = "lda"
    TSNE = "tsne"
    SVD = "svd"


class FeatureEngineer:
    """
    Comprehensive feature engineering toolkit

    Provides feature selection, extraction, transformation,
    and domain-specific feature generation
    """

    def __init__(self):
        self.logger = logger
        self._transformers = {}
        self._feature_info = {}

    # =====================================================================
    # FEATURE SELECTION
    # =====================================================================

    def select_by_variance(
        self,
        data: pd.DataFrame,
        threshold: float = 0.01,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Remove features with low variance

        Args:
            data: DataFrame
            threshold: Variance threshold (features below this are removed)
            columns: Specific columns to consider (None = all numeric)

        Returns:
            DataFrame with low-variance features removed
        """
        self.logger.info(f"Selecting features by variance (threshold={threshold})")

        result = data.copy()

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        selector = VarianceThreshold(threshold=threshold)
        selected_data = selector.fit_transform(data[columns])

        # Get selected feature names
        selected_mask = selector.get_support()
        selected_cols = [col for col, selected in zip(columns, selected_mask) if selected]
        removed_cols = [col for col, selected in zip(columns, selected_mask) if not selected]

        # Create result DataFrame
        result = data.drop(columns=removed_cols)

        self._feature_info['variance_selection'] = {
            'method': 'variance_threshold',
            'threshold': threshold,
            'original_features': len(columns),
            'selected_features': len(selected_cols),
            'removed_features': removed_cols
        }

        self.logger.info(f"Removed {len(removed_cols)} low-variance features: {removed_cols}")

        return result

    def select_by_correlation(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        threshold: float = 0.1,
        method: str = 'pearson'
    ) -> List[str]:
        """
        Select features by correlation with target

        Args:
            data: DataFrame
            target: Target column name
            threshold: Minimum absolute correlation
            method: Correlation method

        Returns:
            List of selected feature names
        """
        if target is None or target not in data.columns:
            raise InvalidParameterError("Target column required for correlation selection")

        self.logger.info(f"Selecting features by correlation with {target}")

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)

        correlations = {}
        for col in numeric_cols:
            corr = data[col].corr(data[target], method=method)
            if abs(corr) >= threshold:
                correlations[col] = abs(corr)

        selected = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, _ in selected]

        self._feature_info['correlation_selection'] = {
            'method': 'correlation',
            'target': target,
            'threshold': threshold,
            'selected_features': selected_features,
            'correlations': dict(selected)
        }

        self.logger.info(f"Selected {len(selected_features)} features by correlation")

        return selected_features

    def select_by_mutual_info(
        self,
        data: pd.DataFrame,
        target: str,
        k: int = 10,
        task: str = 'classification'
    ) -> List[str]:
        """
        Select top k features by mutual information

        Args:
            data: DataFrame
            target: Target column name
            k: Number of features to select
            task: 'classification' or 'regression'

        Returns:
            List of selected feature names
        """
        if target not in data.columns:
            raise InvalidParameterError(f"Target '{target}' not found")

        self.logger.info(f"Selecting top {k} features by mutual information")

        X = data.drop(columns=[target])
        y = data[target]

        # Get numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]

        if task == 'classification':
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression

        selector = SelectKBest(score_func=mi_func, k=min(k, len(numeric_cols)))
        selector.fit(X, y)

        # Get selected features
        selected_mask = selector.get_support()
        selected_features = [col for col, selected in zip(numeric_cols, selected_mask) if selected]

        # Get scores
        scores = dict(zip(numeric_cols, selector.scores_))

        self._feature_info['mutual_info_selection'] = {
            'method': 'mutual_information',
            'task': task,
            'k': k,
            'selected_features': selected_features,
            'scores': scores
        }

        self.logger.info(f"Selected features: {selected_features}")

        return selected_features

    def select_by_rfe(
        self,
        data: pd.DataFrame,
        target: str,
        estimator: Any,
        n_features: int = 10
    ) -> List[str]:
        """
        Recursive Feature Elimination

        Args:
            data: DataFrame
            target: Target column name
            estimator: Sklearn estimator with fit method
            n_features: Number of features to select

        Returns:
            List of selected feature names
        """
        if target not in data.columns:
            raise InvalidParameterError(f"Target '{target}' not found")

        self.logger.info(f"Performing RFE to select {n_features} features")

        X = data.drop(columns=[target])
        y = data[target]

        # Get numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]

        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        rfe.fit(X, y)

        # Get selected features
        selected_mask = rfe.get_support()
        selected_features = [col for col, selected in zip(numeric_cols, selected_mask) if selected]

        # Get rankings
        rankings = dict(zip(numeric_cols, rfe.ranking_))

        self._feature_info['rfe_selection'] = {
            'method': 'rfe',
            'n_features': n_features,
            'selected_features': selected_features,
            'rankings': rankings
        }

        self.logger.info(f"RFE selected features: {selected_features}")

        return selected_features

    # =====================================================================
    # FEATURE EXTRACTION / DIMENSIONALITY REDUCTION
    # =====================================================================

    def extract_pca(
        self,
        data: pd.DataFrame,
        n_components: Union[int, float] = 0.95,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply PCA for dimensionality reduction

        Args:
            data: DataFrame
            n_components: Number of components or variance to retain (0-1)
            columns: Columns to apply PCA on (None = all numeric)

        Returns:
            DataFrame with PCA components
        """
        self.logger.info(f"Applying PCA (n_components={n_components})")

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        X = data[columns].fillna(0)  # PCA requires no NaN

        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X)

        # Create component names
        component_names = [f'PC{i+1}' for i in range(components.shape[1])]

        # Create result DataFrame
        result = pd.DataFrame(
            components,
            columns=component_names,
            index=data.index
        )

        # Add non-PCA columns back
        other_cols = [col for col in data.columns if col not in columns]
        for col in other_cols:
            result[col] = data[col]

        self._transformers['pca'] = pca
        self._feature_info['pca'] = {
            'method': 'pca',
            'n_components': components.shape[1],
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist()
        }

        self.logger.info(f"PCA created {components.shape[1]} components "
                        f"(variance retained: {np.sum(pca.explained_variance_ratio_):.2%})")

        return result

    def extract_lda(
        self,
        data: pd.DataFrame,
        target: str,
        n_components: Optional[int] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply Linear Discriminant Analysis

        Args:
            data: DataFrame
            target: Target column name
            n_components: Number of components (None = min(n_classes-1, n_features))
            columns: Columns to apply LDA on (None = all numeric)

        Returns:
            DataFrame with LDA components
        """
        if target not in data.columns:
            raise InvalidParameterError(f"Target '{target}' not found")

        self.logger.info(f"Applying LDA")

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if target in columns:
                columns.remove(target)

        X = data[columns].fillna(0)
        y = data[target]

        lda = LDA(n_components=n_components)
        components = lda.fit_transform(X, y)

        # Create component names
        component_names = [f'LD{i+1}' for i in range(components.shape[1])]

        # Create result DataFrame
        result = pd.DataFrame(
            components,
            columns=component_names,
            index=data.index
        )

        # Add non-LDA columns back
        other_cols = [col for col in data.columns if col not in columns]
        for col in other_cols:
            result[col] = data[col]

        self._transformers['lda'] = lda
        self._feature_info['lda'] = {
            'method': 'lda',
            'n_components': components.shape[1],
            'explained_variance_ratio': lda.explained_variance_ratio_.tolist()
        }

        self.logger.info(f"LDA created {components.shape[1]} components")

        return result

    def extract_tsne(
        self,
        data: pd.DataFrame,
        n_components: int = 2,
        columns: Optional[List[str]] = None,
        perplexity: float = 30.0,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Apply t-SNE for dimensionality reduction (primarily for visualization)

        Args:
            data: DataFrame
            n_components: Number of dimensions (typically 2 or 3)
            columns: Columns to apply t-SNE on (None = all numeric)
            perplexity: t-SNE perplexity parameter
            random_state: Random seed

        Returns:
            DataFrame with t-SNE components
        """
        self.logger.info(f"Applying t-SNE (n_components={n_components})")

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        X = data[columns].fillna(0)

        # t-SNE can be slow, so limit to reasonable sample size
        if len(X) > 10000:
            self.logger.warning(f"Large dataset ({len(X)} rows). Consider sampling for t-SNE.")

        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        components = tsne.fit_transform(X)

        # Create component names
        component_names = [f'TSNE{i+1}' for i in range(n_components)]

        # Create result DataFrame
        result = pd.DataFrame(
            components,
            columns=component_names,
            index=data.index
        )

        # Add non-t-SNE columns back
        other_cols = [col for col in data.columns if col not in columns]
        for col in other_cols:
            result[col] = data[col]

        self._feature_info['tsne'] = {
            'method': 'tsne',
            'n_components': n_components,
            'perplexity': perplexity
        }

        self.logger.info(f"t-SNE created {n_components} components")

        return result

    # =====================================================================
    # FEATURE TRANSFORMATION
    # =====================================================================

    def create_polynomial_features(
        self,
        data: pd.DataFrame,
        degree: int = 2,
        columns: Optional[List[str]] = None,
        include_bias: bool = False,
        interaction_only: bool = False
    ) -> pd.DataFrame:
        """
        Generate polynomial and interaction features

        Args:
            data: DataFrame
            degree: Polynomial degree
            columns: Columns to create polynomials from (None = all numeric)
            include_bias: Include bias column (all 1s)
            interaction_only: Only interaction features (no powers)

        Returns:
            DataFrame with polynomial features
        """
        self.logger.info(f"Creating polynomial features (degree={degree})")

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        X = data[columns]

        poly = PolynomialFeatures(
            degree=degree,
            include_bias=include_bias,
            interaction_only=interaction_only
        )
        poly_features = poly.fit_transform(X)

        # Get feature names
        feature_names = poly.get_feature_names_out(columns)

        # Create result DataFrame
        poly_df = pd.DataFrame(
            poly_features,
            columns=feature_names,
            index=data.index
        )

        # Add non-polynomial columns back
        other_cols = [col for col in data.columns if col not in columns]
        for col in other_cols:
            poly_df[col] = data[col]

        self._transformers['polynomial'] = poly
        self._feature_info['polynomial'] = {
            'method': 'polynomial_features',
            'degree': degree,
            'original_features': len(columns),
            'new_features': len(feature_names),
            'interaction_only': interaction_only
        }

        self.logger.info(f"Created {len(feature_names)} polynomial features")

        return poly_df

    def create_binned_features(
        self,
        data: pd.DataFrame,
        columns: Union[str, List[str]],
        n_bins: int = 5,
        strategy: str = 'quantile',
        encode: str = 'ordinal'
    ) -> pd.DataFrame:
        """
        Bin continuous features into discrete bins

        Args:
            data: DataFrame
            columns: Column(s) to bin
            n_bins: Number of bins
            strategy: 'uniform', 'quantile', or 'kmeans'
            encode: 'ordinal', 'onehot', or 'onehot-dense'

        Returns:
            DataFrame with binned features
        """
        if isinstance(columns, str):
            columns = [columns]

        self.logger.info(f"Binning features: {columns}")

        result = data.copy()

        for col in columns:
            if col not in data.columns:
                raise InvalidParameterError(f"Column '{col}' not found")

            discretizer = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
            binned = discretizer.fit_transform(data[[col]])

            if encode == 'ordinal':
                result[f'{col}_binned'] = binned
            else:
                # One-hot encoded
                bin_names = [f'{col}_bin_{i}' for i in range(n_bins)]
                binned_df = pd.DataFrame(binned, columns=bin_names, index=data.index)
                result = pd.concat([result, binned_df], axis=1)

            self._transformers[f'bins_{col}'] = discretizer

        self._feature_info['binning'] = {
            'method': 'binning',
            'columns': columns,
            'n_bins': n_bins,
            'strategy': strategy,
            'encode': encode
        }

        self.logger.info(f"Created binned features for {len(columns)} columns")

        return result

    # =====================================================================
    # TEXT FEATURES
    # =====================================================================

    def extract_text_features(
        self,
        data: pd.DataFrame,
        text_column: str,
        method: str = 'tfidf',
        max_features: int = 100,
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract features from text column

        Args:
            data: DataFrame
            text_column: Column containing text
            method: 'tfidf' or 'count'
            max_features: Maximum number of features to extract
            **kwargs: Additional parameters for vectorizer

        Returns:
            DataFrame with text features
        """
        if text_column not in data.columns:
            raise InvalidParameterError(f"Column '{text_column}' not found")

        self.logger.info(f"Extracting text features from '{text_column}' using {method}")

        # Fill NaN with empty string
        texts = data[text_column].fillna('').astype(str)

        if method == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=max_features, **kwargs)
        elif method == 'count':
            vectorizer = CountVectorizer(max_features=max_features, **kwargs)
        else:
            raise InvalidParameterError(f"Unknown method: {method}")

        features = vectorizer.fit_transform(texts)

        # Create feature names
        feature_names = [f'{text_column}_{name}' for name in vectorizer.get_feature_names_out()]

        # Create DataFrame
        text_df = pd.DataFrame(
            features.toarray(),
            columns=feature_names,
            index=data.index
        )

        # Combine with original data
        result = pd.concat([data, text_df], axis=1)

        self._transformers[f'text_{text_column}'] = vectorizer
        self._feature_info['text_features'] = {
            'method': method,
            'column': text_column,
            'max_features': max_features,
            'actual_features': len(feature_names)
        }

        self.logger.info(f"Extracted {len(feature_names)} text features")

        return result

    # =====================================================================
    # DATE FEATURES
    # =====================================================================

    def extract_date_features(
        self,
        data: pd.DataFrame,
        date_column: str,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract features from datetime column

        Args:
            data: DataFrame
            date_column: Column containing dates
            features: List of features to extract (None = all)
                     Options: 'year', 'month', 'day', 'dayofweek', 'quarter',
                             'dayofyear', 'weekofyear', 'is_weekend', 'hour', 'minute'

        Returns:
            DataFrame with date features
        """
        if date_column not in data.columns:
            raise InvalidParameterError(f"Column '{date_column}' not found")

        self.logger.info(f"Extracting date features from '{date_column}'")

        result = data.copy()

        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(result[date_column]):
            result[date_column] = pd.to_datetime(result[date_column], errors='coerce')

        dt = result[date_column]

        # All available features
        all_features = {
            'year': lambda d: d.dt.year,
            'month': lambda d: d.dt.month,
            'day': lambda d: d.dt.day,
            'dayofweek': lambda d: d.dt.dayofweek,
            'quarter': lambda d: d.dt.quarter,
            'dayofyear': lambda d: d.dt.dayofyear,
            'weekofyear': lambda d: d.dt.isocalendar().week,
            'is_weekend': lambda d: d.dt.dayofweek.isin([5, 6]).astype(int),
            'hour': lambda d: d.dt.hour if hasattr(d.dt, 'hour') else None,
            'minute': lambda d: d.dt.minute if hasattr(d.dt, 'minute') else None,
        }

        # Select features to extract
        if features is None:
            features = list(all_features.keys())

        extracted_features = []
        for feat in features:
            if feat in all_features:
                try:
                    result[f'{date_column}_{feat}'] = all_features[feat](dt)
                    extracted_features.append(feat)
                except Exception as e:
                    self.logger.warning(f"Could not extract '{feat}': {e}")

        self._feature_info['date_features'] = {
            'method': 'date_extraction',
            'column': date_column,
            'extracted_features': extracted_features
        }

        self.logger.info(f"Extracted {len(extracted_features)} date features")

        return result

    # =====================================================================
    # UTILITIES
    # =====================================================================

    def get_feature_importance(
        self,
        data: pd.DataFrame,
        target: str,
        method: str = 'mutual_info',
        task: str = 'classification'
    ) -> Dict[str, float]:
        """
        Calculate feature importance scores

        Args:
            data: DataFrame
            target: Target column
            method: 'mutual_info', 'f_test', or 'chi2'
            task: 'classification' or 'regression'

        Returns:
            Dictionary of {feature: importance_score}
        """
        if target not in data.columns:
            raise InvalidParameterError(f"Target '{target}' not found")

        X = data.drop(columns=[target])
        y = data[target]

        # Get numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]

        if method == 'mutual_info':
            if task == 'classification':
                scores = mutual_info_classif(X, y)
            else:
                scores = mutual_info_regression(X, y)
        elif method == 'f_test':
            if task == 'classification':
                scores, _ = f_classif(X, y)
            else:
                scores, _ = f_regression(X, y)
        elif method == 'chi2':
            scores, _ = chi2(X, y)
        else:
            raise InvalidParameterError(f"Unknown method: {method}")

        importance = dict(zip(numeric_cols, scores))

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return importance

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about applied transformations"""
        return self._feature_info

    def get_transformers(self) -> Dict[str, Any]:
        """Get fitted transformers for later use"""
        return self._transformers
