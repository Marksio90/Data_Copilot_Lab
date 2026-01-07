"""
Data Copilot Lab - Insight Generator
Automatically generate business insights from data analysis
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.modules.ai_assistant.llm_integration import LLMIntegration, LLMModel
from src.core.exceptions import InvalidParameterError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InsightType(str, Enum):
    """Types of insights"""
    TREND = "trend"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"
    PATTERN = "pattern"
    COMPARISON = "comparison"
    DISTRIBUTION = "distribution"
    FORECAST = "forecast"
    RECOMMENDATION = "recommendation"


class InsightPriority(str, Enum):
    """Insight priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InsightGenerator:
    """
    Automatically generate business insights from data

    Analyzes data and generates actionable insights:
    - Trends and patterns
    - Anomalies and outliers
    - Correlations and relationships
    - Comparisons and benchmarks
    - Forecasts and predictions
    - Recommendations
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_model: LLMModel = LLMModel.GPT35_TURBO,
        api_key: Optional[str] = None
    ):
        """
        Initialize Insight Generator

        Args:
            use_llm: Use LLM for enhanced insights
            llm_model: LLM model
            api_key: API key
        """
        self.logger = logger
        self.use_llm = use_llm

        if use_llm:
            self.llm = LLMIntegration(model=llm_model, api_key=api_key, temperature=0.7)

    def generate_insights(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        max_insights: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate comprehensive insights from data

        Args:
            data: DataFrame to analyze
            target: Target variable (optional)
            max_insights: Maximum number of insights

        Returns:
            List of insights
        """
        self.logger.info(f"Generating insights for dataset ({data.shape})")

        insights = []

        # Data quality insights
        insights.extend(self._generate_data_quality_insights(data))

        # Distribution insights
        insights.extend(self._generate_distribution_insights(data))

        # Correlation insights
        if target:
            insights.extend(self._generate_correlation_insights(data, target))

        # Trend insights (if datetime column exists)
        datetime_cols = data.select_dtypes(include=['datetime']).columns
        if len(datetime_cols) > 0:
            insights.extend(self._generate_trend_insights(data, datetime_cols[0]))

        # Anomaly insights
        insights.extend(self._generate_anomaly_insights(data))

        # Comparison insights
        insights.extend(self._generate_comparison_insights(data))

        # Sort by priority
        priority_order = {
            InsightPriority.CRITICAL: 0,
            InsightPriority.HIGH: 1,
            InsightPriority.MEDIUM: 2,
            InsightPriority.LOW: 3
        }
        insights.sort(key=lambda x: priority_order.get(x['priority'], 3))

        # Limit to max_insights
        insights = insights[:max_insights]

        # Enhance with LLM if available
        if self.use_llm:
            insights = self._enhance_insights_with_llm(insights, data)

        self.logger.info(f"Generated {len(insights)} insights")

        return insights

    def _generate_data_quality_insights(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate insights about data quality"""
        insights = []

        # Missing data insight
        missing_pct = (data.isnull().sum().sum() / data.size) * 100
        if missing_pct > 10:
            insights.append({
                "type": InsightType.ANOMALY,
                "priority": InsightPriority.HIGH,
                "title": "High Missing Data Detected",
                "description": f"{missing_pct:.1f}% of data is missing. This may impact analysis reliability.",
                "metric": missing_pct,
                "recommendation": "Consider imputation strategies or investigate data collection process."
            })

        # Duplicate insight
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / len(data)) * 100
            if duplicate_pct > 5:
                insights.append({
                    "type": InsightType.ANOMALY,
                    "priority": InsightPriority.MEDIUM,
                    "title": "Duplicate Records Found",
                    "description": f"{duplicate_count} duplicate records ({duplicate_pct:.1f}%) detected.",
                    "metric": duplicate_count,
                    "recommendation": "Review and remove duplicates to ensure data integrity."
                })

        return insights

    def _generate_distribution_insights(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate insights about data distributions"""
        insights = []

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols[:5]:  # Check first 5 numeric columns
            # Skewness
            skewness = data[col].skew()
            if abs(skewness) > 1:
                direction = "right" if skewness > 0 else "left"
                insights.append({
                    "type": InsightType.DISTRIBUTION,
                    "priority": InsightPriority.LOW,
                    "title": f"{col}: Skewed Distribution",
                    "description": f"'{col}' is highly skewed {direction} (skewness: {skewness:.2f}).",
                    "metric": skewness,
                    "recommendation": f"Consider log transformation or other normalization for '{col}'."
                })

            # Outliers using IQR
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()

            if outliers > 0:
                outlier_pct = (outliers / len(data)) * 100
                if outlier_pct > 5:
                    insights.append({
                        "type": InsightType.ANOMALY,
                        "priority": InsightPriority.MEDIUM,
                        "title": f"{col}: Outliers Detected",
                        "description": f"{outliers} outliers ({outlier_pct:.1f}%) found in '{col}'.",
                        "metric": outliers,
                        "recommendation": f"Investigate outliers in '{col}' - may indicate data issues or important events."
                    })

        return insights

    def _generate_correlation_insights(
        self,
        data: pd.DataFrame,
        target: str
    ) -> List[Dict[str, Any]]:
        """Generate insights about correlations with target"""
        insights = []

        if target not in data.columns:
            return insights

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if target not in numeric_cols:
            return insights

        # Calculate correlations
        correlations = {}
        for col in numeric_cols:
            if col != target:
                corr = data[col].corr(data[target])
                if not np.isnan(corr):
                    correlations[col] = corr

        # Sort by absolute correlation
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        # Top positive correlations
        if sorted_corrs and sorted_corrs[0][1] > 0.5:
            col, corr = sorted_corrs[0]
            insights.append({
                "type": InsightType.CORRELATION,
                "priority": InsightPriority.HIGH,
                "title": f"Strong Positive Correlation: {col}",
                "description": f"'{col}' has strong positive correlation ({corr:.3f}) with '{target}'.",
                "metric": corr,
                "recommendation": f"'{col}' is a strong predictor for '{target}'. Consider it as key feature."
            })

        # Strong negative correlations
        negative_corrs = [c for c in sorted_corrs if c[1] < -0.5]
        if negative_corrs:
            col, corr = negative_corrs[0]
            insights.append({
                "type": InsightType.CORRELATION,
                "priority": InsightPriority.MEDIUM,
                "title": f"Strong Negative Correlation: {col}",
                "description": f"'{col}' has strong negative correlation ({corr:.3f}) with '{target}'.",
                "metric": corr,
                "recommendation": f"Inverse relationship between '{col}' and '{target}' - explore further."
            })

        return insights

    def _generate_trend_insights(
        self,
        data: pd.DataFrame,
        datetime_col: str
    ) -> List[Dict[str, Any]]:
        """Generate insights about trends over time"""
        insights = []

        # Sort by datetime
        data_sorted = data.sort_values(datetime_col)

        numeric_cols = data_sorted.select_dtypes(include=[np.number]).columns

        for col in numeric_cols[:3]:  # Check first 3 numeric columns
            # Simple linear trend
            x = np.arange(len(data_sorted))
            y = data_sorted[col].values

            # Remove NaN
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                continue

            x_clean = x[mask]
            y_clean = y[mask]

            # Linear regression
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_clean, y_clean)

            if abs(r_value) > 0.5 and p_value < 0.05:
                direction = "increasing" if slope > 0 else "decreasing"
                change_pct = (slope * len(x_clean) / y_clean.mean()) * 100

                insights.append({
                    "type": InsightType.TREND,
                    "priority": InsightPriority.HIGH,
                    "title": f"{col}: {direction.capitalize()} Trend",
                    "description": f"'{col}' shows {direction} trend over time (R²={r_value**2:.3f}, change: {change_pct:+.1f}%).",
                    "metric": change_pct,
                    "recommendation": f"Monitor '{col}' trend - {direction} by {abs(change_pct):.1f}% over period."
                })

        return insights

    def _generate_anomaly_insights(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate insights about anomalies"""
        insights = []

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols[:3]:
            # Z-score method
            z_scores = np.abs(scipy_stats.zscore(data[col].dropna()))
            anomalies = (z_scores > 3).sum()

            if anomalies > 0:
                anomaly_pct = (anomalies / len(data)) * 100
                if anomaly_pct > 1:
                    insights.append({
                        "type": InsightType.ANOMALY,
                        "priority": InsightPriority.MEDIUM,
                        "title": f"{col}: Statistical Anomalies",
                        "description": f"{anomalies} values ({anomaly_pct:.1f}%) are statistical anomalies (>3 std dev) in '{col}'.",
                        "metric": anomalies,
                        "recommendation": "Investigate anomalous values - may represent important events or errors."
                    })

        return insights

    def _generate_comparison_insights(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate comparison insights"""
        insights = []

        # Compare categorical groups
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for cat_col in categorical_cols[:2]:
            unique_vals = data[cat_col].nunique()
            if 2 <= unique_vals <= 10:  # Reasonable number of groups
                for num_col in numeric_cols[:2]:
                    # Compare means across groups
                    groups = data.groupby(cat_col)[num_col].mean()
                    max_group = groups.idxmax()
                    min_group = groups.idxmin()
                    diff_pct = ((groups[max_group] - groups[min_group]) / groups[min_group]) * 100

                    if abs(diff_pct) > 20:
                        insights.append({
                            "type": InsightType.COMPARISON,
                            "priority": InsightPriority.MEDIUM,
                            "title": f"Group Difference: {num_col} by {cat_col}",
                            "description": f"'{max_group}' has {diff_pct:+.1f}% {'higher' if diff_pct > 0 else 'lower'} '{num_col}' than '{min_group}'.",
                            "metric": diff_pct,
                            "recommendation": f"Investigate why '{max_group}' differs significantly in '{num_col}'."
                        })

        return insights

    def _enhance_insights_with_llm(
        self,
        insights: List[Dict[str, Any]],
        data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Enhance insights with LLM-generated explanations"""
        if not self.use_llm or not insights:
            return insights

        try:
            # Prepare context
            context = {
                "data_shape": data.shape,
                "columns": list(data.columns),
                "insights_count": len(insights)
            }

            # Generate enhanced explanations for top insights
            for insight in insights[:3]:  # Enhance top 3
                system_message = """You are a business intelligence expert.
Provide a clear, actionable business explanation for this data insight."""

                prompt = f"""Insight: {insight['title']}
Description: {insight['description']}
Data context: {context}

Provide a brief (2-3 sentences) business-focused explanation and actionable recommendation."""

                enhanced = self.llm.generate_text(prompt, system_message=system_message, max_tokens=150)

                insight['enhanced_explanation'] = enhanced

        except Exception as e:
            self.logger.warning(f"LLM enhancement failed: {e}")

        return insights

    def generate_summary(self, insights: List[Dict[str, Any]]) -> str:
        """
        Generate summary of insights

        Args:
            insights: List of insights

        Returns:
            Summary text
        """
        if not insights:
            return "No significant insights found."

        summary_parts = [f"Analysis Summary ({len(insights)} insights):\n"]

        # Count by type
        by_type = {}
        for insight in insights:
            itype = insight['type']
            by_type[itype] = by_type.get(itype, 0) + 1

        summary_parts.append("\nInsight Types:")
        for itype, count in by_type.items():
            summary_parts.append(f"- {itype}: {count}")

        # Top priorities
        critical = [i for i in insights if i['priority'] == InsightPriority.CRITICAL]
        high = [i for i in insights if i['priority'] == InsightPriority.HIGH]

        summary_parts.append(f"\nPriorities:")
        summary_parts.append(f"- Critical: {len(critical)}")
        summary_parts.append(f"- High: {len(high)}")

        # Top 3 insights
        summary_parts.append(f"\nTop Insights:")
        for i, insight in enumerate(insights[:3], 1):
            summary_parts.append(f"{i}. {insight['title']}")

        return "\n".join(summary_parts)

    def export_insights(
        self,
        insights: List[Dict[str, Any]],
        format: str = "dict"
    ) -> Any:
        """
        Export insights in various formats

        Args:
            insights: List of insights
            format: 'dict', 'dataframe', or 'markdown'

        Returns:
            Exported insights
        """
        if format == "dict":
            return insights

        elif format == "dataframe":
            return pd.DataFrame(insights)

        elif format == "markdown":
            md_parts = ["# Insights\n"]
            for i, insight in enumerate(insights, 1):
                md_parts.append(f"## {i}. {insight['title']}")
                md_parts.append(f"**Type:** {insight['type']}")
                md_parts.append(f"**Priority:** {insight['priority']}")
                md_parts.append(f"\n{insight['description']}\n")
                if insight.get('recommendation'):
                    md_parts.append(f"**Recommendation:** {insight['recommendation']}\n")
                md_parts.append("---\n")

            return "\n".join(md_parts)

        else:
            raise InvalidParameterError(f"Unknown format: {format}")
