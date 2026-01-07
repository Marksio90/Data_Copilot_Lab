"""
Data Copilot Lab - Automated EDA
Automated exploratory data analysis with comprehensive reports
"""

from typing import Any, Dict, Optional

import pandas as pd

from src.modules.eda.correlation import CorrelationAnalyzer
from src.modules.eda.statistics import StatisticalAnalyzer
from src.modules.eda.visualization import VisualizationEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AutoEDA:
    """
    Automated Exploratory Data Analysis

    Combines statistical analysis, visualization, and correlation
    analysis into comprehensive automated reports
    """

    def __init__(self):
        self.logger = logger
        self.stats_analyzer = StatisticalAnalyzer()
        self.viz_engine = VisualizationEngine()
        self.corr_analyzer = CorrelationAnalyzer()

    def analyze(self, data: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive automated EDA

        Args:
            data: DataFrame to analyze
            target: Optional target variable for supervised learning

        Returns:
            Comprehensive analysis report
        """
        self.logger.info("Starting automated EDA...")

        report = {
            "dataset_overview": self._dataset_overview(data),
            "descriptive_statistics": self.stats_analyzer.describe(data),
            "data_quality": self._data_quality_assessment(data),
            "correlations": self._correlation_analysis(data),
            "distribution_analysis": self._distribution_analysis(data),
            "insights": self._generate_insights(data, target),
            "recommendations": self._generate_recommendations(data)
        }

        if target and target in data.columns:
            report["target_analysis"] = self._target_analysis(data, target)

        self.logger.info("Automated EDA complete")

        return report

    def _dataset_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Basic dataset overview"""
        return {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.astype(str).to_dict(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024 * 1024)
        }

    def _data_quality_assessment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality"""
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()

        return {
            "completeness": {
                "total_cells": total_cells,
                "missing_cells": int(missing_cells),
                "missing_percentage": (missing_cells / total_cells * 100) if total_cells > 0 else 0,
                "complete_rows": int((~data.isnull().any(axis=1)).sum()),
                "complete_percentage": ((~data.isnull().any(axis=1)).sum() / len(data) * 100) if len(data) > 0 else 0
            },
            "duplicates": {
                "duplicate_rows": int(data.duplicated().sum()),
                "duplicate_percentage": (data.duplicated().sum() / len(data) * 100) if len(data) > 0 else 0
            },
            "columns_with_missing": data.columns[data.isnull().any()].tolist()
        }

    def _correlation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations"""
        numeric_data = data.select_dtypes(include=['number'])

        if len(numeric_data.columns) < 2:
            return {"message": "Not enough numeric columns for correlation analysis"}

        corr_matrix = self.corr_analyzer.correlation_matrix(data)
        high_corr = self.corr_analyzer.find_highly_correlated(data, threshold=0.7)

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "highly_correlated_pairs": [
                {"feature1": f1, "feature2": f2, "correlation": corr}
                for f1, f2, corr in high_corr
            ]
        }

    def _distribution_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric columns"""
        numeric_cols = data.select_dtypes(include=['number']).columns

        distributions = {}
        for col in numeric_cols[:10]:  # Limit to first 10
            try:
                dist_info = self.stats_analyzer.distribution_analysis(data, col)
                distributions[col] = {
                    "is_normal": dist_info["distribution_tests"]["shapiro_wilk"]["is_normal"],
                    "skewness": dist_info["parameters"]["skewness"],
                    "kurtosis": dist_info["parameters"]["kurtosis"],
                    "suggested_distribution": dist_info["distribution_type"]["suggested"]
                }
            except Exception as e:
                self.logger.warning(f"Could not analyze distribution for {col}: {e}")

        return distributions

    def _target_analysis(self, data: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Analyze target variable and its relationships"""
        if not pd.api.types.is_numeric_dtype(data[target]):
            # Categorical target
            return {
                "type": "categorical",
                "unique_values": int(data[target].nunique()),
                "value_counts": data[target].value_counts().to_dict(),
                "class_balance": data[target].value_counts(normalize=True).to_dict()
            }
        else:
            # Numeric target
            correlations = self.corr_analyzer.correlation_with_target(data, target)

            return {
                "type": "numeric",
                "statistics": data[target].describe().to_dict(),
                "top_correlations": dict(list(correlations.items())[:10])
            }

    def _generate_insights(self, data: pd.DataFrame, target: Optional[str] = None) -> List[str]:
        """Generate automated insights"""
        insights = []

        # Data size insights
        if len(data) < 100:
            insights.append("⚠️ Small dataset (< 100 rows). Results may not be statistically significant.")
        elif len(data) > 100000:
            insights.append("📊 Large dataset (> 100k rows). Consider sampling for faster exploration.")

        # Missing data insights
        missing_pct = (data.isnull().sum().sum() / data.size * 100)
        if missing_pct > 10:
            insights.append(f"⚠️ High missing data ({missing_pct:.1f}%). Data imputation recommended.")
        elif missing_pct == 0:
            insights.append("✅ No missing values detected.")

        # Duplicate insights
        dup_pct = (data.duplicated().sum() / len(data) * 100)
        if dup_pct > 5:
            insights.append(f"🔄 Duplicates detected ({dup_pct:.1f}%). Consider deduplication.")

        # Column type insights
        n_numeric = len(data.select_dtypes(include=['number']).columns)
        n_categorical = len(data.select_dtypes(include=['object', 'category']).columns)

        if n_numeric > n_categorical * 2:
            insights.append("📈 Numeric-heavy dataset. Regression or clustering may be suitable.")
        elif n_categorical > n_numeric * 2:
            insights.append("📝 Categorical-heavy dataset. Consider encoding methods.")

        return insights

    def _generate_recommendations(self, data: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Missing data recommendations
        cols_with_missing = data.columns[data.isnull().any()]
        if len(cols_with_missing) > 0:
            recommendations.append(
                f"🔧 Handle missing values in {len(cols_with_missing)} columns. "
                "Consider imputation or removal."
            )

        # High cardinality recommendations
        for col in data.select_dtypes(include=['object', 'category']).columns:
            if data[col].nunique() > 50:
                recommendations.append(
                    f"📊 Column '{col}' has high cardinality ({data[col].nunique()} unique values). "
                    "Consider grouping or target encoding."
                )

        # Correlation recommendations
        try:
            high_corr = self.corr_analyzer.find_highly_correlated(data, threshold=0.9)
            if high_corr:
                recommendations.append(
                    f"🔗 Found {len(high_corr)} highly correlated feature pairs (>0.9). "
                    "Consider removing redundant features."
                )
        except Exception:
            pass

        # Scaling recommendations
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            ranges = data[numeric_cols].max() - data[numeric_cols].min()
            if ranges.max() / ranges.min() > 100:
                recommendations.append(
                    "📏 Features have very different scales. Consider standardization."
                )

        return recommendations

    def quick_summary(self, data: pd.DataFrame) -> str:
        """
        Generate quick text summary

        Returns:
            Formatted text summary
        """
        n_rows, n_cols = data.shape
        missing_pct = (data.isnull().sum().sum() / data.size * 100)
        n_numeric = len(data.select_dtypes(include=['number']).columns)
        n_categorical = len(data.select_dtypes(include=['object', 'category']).columns)

        summary = f"""
Dataset Summary:
- Shape: {n_rows} rows × {n_cols} columns
- Numeric columns: {n_numeric}
- Categorical columns: {n_categorical}
- Missing data: {missing_pct:.2f}%
- Duplicates: {data.duplicated().sum()} rows ({data.duplicated().sum()/len(data)*100:.2f}%)
- Memory usage: {data.memory_usage(deep=True).sum() / (1024*1024):.2f} MB
        """

        return summary.strip()
