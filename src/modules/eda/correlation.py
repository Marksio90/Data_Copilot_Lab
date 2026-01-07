"""
Data Copilot Lab - Correlation Analysis
Analyze correlations between variables
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.core.exceptions import InvalidParameterError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CorrelationAnalyzer:
    """
    Analyze correlations between variables

    Supports multiple correlation methods and provides
    detailed analysis of relationships
    """

    def __init__(self):
        self.logger = logger

    def correlation_matrix(
        self,
        data: pd.DataFrame,
        method: str = 'pearson',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix

        Args:
            data: DataFrame
            method: 'pearson', 'spearman', or 'kendall'
            columns: Specific columns (None = all numeric)

        Returns:
            Correlation matrix
        """
        if columns:
            data = data[columns]
        else:
            data = data.select_dtypes(include=[np.number])

        if data.empty:
            raise InvalidParameterError("No numeric columns found")

        return data.corr(method=method)

    def correlation_with_target(
        self,
        data: pd.DataFrame,
        target: str,
        method: str = 'pearson',
        threshold: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate correlation of all features with target variable

        Args:
            data: DataFrame
            target: Target column name
            method: Correlation method
            threshold: Minimum absolute correlation to include

        Returns:
            Dictionary of {feature: correlation}
        """
        if target not in data.columns:
            raise InvalidParameterError(f"Target column '{target}' not found")

        numeric_data = data.select_dtypes(include=[np.number])

        if target not in numeric_data.columns:
            raise InvalidParameterError(f"Target '{target}' must be numeric")

        correlations = {}
        for col in numeric_data.columns:
            if col != target:
                corr = numeric_data[col].corr(numeric_data[target], method=method)
                if abs(corr) >= threshold:
                    correlations[col] = float(corr)

        # Sort by absolute value
        return dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))

    def find_highly_correlated(
        self,
        data: pd.DataFrame,
        threshold: float = 0.8,
        method: str = 'pearson'
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs of highly correlated features

        Args:
            data: DataFrame
            threshold: Minimum correlation threshold
            method: Correlation method

        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        corr_matrix = self.correlation_matrix(data, method)

        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        float(corr_value)
                    ))

        # Sort by absolute correlation
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return high_corr_pairs

    def correlation_significance(
        self,
        data: pd.DataFrame,
        col1: str,
        col2: str,
        method: str = 'pearson'
    ) -> Dict[str, Any]:
        """
        Test statistical significance of correlation

        Args:
            data: DataFrame
            col1: First column
            col2: Second column
            method: Correlation method

        Returns:
            Correlation coefficient and p-value
        """
        if col1 not in data.columns or col2 not in data.columns:
            raise InvalidParameterError("Columns not found")

        series1 = data[col1].dropna()
        series2 = data[col2].dropna()

        # Find common indices
        common_idx = series1.index.intersection(series2.index)
        series1 = series1.loc[common_idx]
        series2 = series2.loc[common_idx]

        if method == 'pearson':
            corr, pval = stats.pearsonr(series1, series2)
        elif method == 'spearman':
            corr, pval = stats.spearmanr(series1, series2)
        elif method == 'kendall':
            corr, pval = stats.kendalltau(series1, series2)
        else:
            raise InvalidParameterError(f"Unknown method: {method}")

        return {
            "correlation": float(corr),
            "p_value": float(pval),
            "significant": bool(pval < 0.05),
            "method": method,
            "n_samples": len(series1)
        }

    def partial_correlation(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        control: List[str]
    ) -> float:
        """
        Calculate partial correlation between x and y, controlling for other variables

        Args:
            data: DataFrame
            x: First variable
            y: Second variable
            control: List of control variables

        Returns:
            Partial correlation coefficient
        """
        # This is a simplified implementation
        # For production, consider using pingouin library

        cols = [x, y] + control
        data_subset = data[cols].dropna()

        # Calculate correlation matrix
        corr_matrix = data_subset.corr().values

        # Partial correlation formula using matrix inversion
        inv_corr = np.linalg.inv(corr_matrix)

        # Partial correlation between x (index 0) and y (index 1)
        partial_corr = -inv_corr[0, 1] / np.sqrt(inv_corr[0, 0] * inv_corr[1, 1])

        return float(partial_corr)
