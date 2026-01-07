"""
Data Copilot Lab - Duplicate Remover
Detect and remove duplicate rows with fuzzy matching support
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from difflib import SequenceMatcher

from src.core.exceptions import DataCleaningError, InvalidParameterError
from src.modules.data_cleaning.base import DataCleaner
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DuplicateStrategy(str, Enum):
    """Strategies for handling duplicates"""
    KEEP_FIRST = "keep_first"  # Keep first occurrence
    KEEP_LAST = "keep_last"  # Keep last occurrence
    KEEP_NONE = "keep_none"  # Remove all duplicates
    MARK = "mark"  # Just mark duplicates without removing


class DuplicateRemover(DataCleaner):
    """
    Detect and remove duplicate rows

    Supports:
    - Exact duplicates
    - Duplicates based on subset of columns
    - Fuzzy matching for text similarity
    - Multiple keeping strategies
    """

    def __init__(self):
        super().__init__()

    def analyze(
        self,
        data: pd.DataFrame,
        subset: Optional[List[str]] = None,
        fuzzy: bool = False,
        threshold: float = 0.9
    ) -> Dict[str, Any]:
        """
        Analyze duplicate patterns in data

        Args:
            data: DataFrame to analyze
            subset: Columns to consider for duplicates (None = all columns)
            fuzzy: Whether to use fuzzy matching
            threshold: Similarity threshold for fuzzy matching (0-1)

        Returns:
            Dictionary with duplicate analysis
        """
        self.logger.info("Analyzing duplicates...")

        if subset:
            # Validate columns exist
            invalid_cols = set(subset) - set(data.columns)
            if invalid_cols:
                raise InvalidParameterError(f"Columns not found: {invalid_cols}")

        # Exact duplicates
        if not fuzzy:
            duplicates_mask = data.duplicated(subset=subset, keep=False)
        else:
            duplicates_mask = self._find_fuzzy_duplicates(data, subset, threshold)

        n_duplicates = duplicates_mask.sum()
        duplicate_groups = self._get_duplicate_groups(data, duplicates_mask, subset)

        return {
            "total_rows": len(data),
            "duplicate_rows": int(n_duplicates),
            "duplicate_percentage": (n_duplicates / len(data) * 100) if len(data) > 0 else 0,
            "unique_rows": len(data) - n_duplicates,
            "duplicate_groups": len(duplicate_groups),
            "subset_columns": subset if subset else "all columns",
            "fuzzy_matching": fuzzy,
            "sample_duplicates": duplicate_groups[:5] if duplicate_groups else [],
            "recommendations": self._generate_recommendations(n_duplicates, len(data))
        }

    def clean(
        self,
        data: pd.DataFrame,
        subset: Optional[List[str]] = None,
        strategy: Union[str, DuplicateStrategy] = DuplicateStrategy.KEEP_FIRST,
        fuzzy: bool = False,
        threshold: float = 0.9,
        **kwargs
    ) -> pd.DataFrame:
        """
        Remove or mark duplicate rows

        Args:
            data: DataFrame to clean
            subset: Columns to consider (None = all columns)
            strategy: Strategy for keeping/removing duplicates
            fuzzy: Whether to use fuzzy matching
            threshold: Similarity threshold for fuzzy (0-1)

        Returns:
            Cleaned DataFrame
        """
        self._store_original(data)
        result = data.copy()

        # Convert string to enum
        if isinstance(strategy, str):
            try:
                strategy = DuplicateStrategy(strategy)
            except ValueError:
                raise InvalidParameterError(
                    f"Invalid strategy: {strategy}. "
                    f"Valid strategies: {[s.value for s in DuplicateStrategy]}"
                )

        self.logger.info(f"Removing duplicates with strategy: {strategy.value}")

        if subset:
            # Validate columns
            invalid_cols = set(subset) - set(result.columns)
            if invalid_cols:
                raise InvalidParameterError(f"Columns not found: {invalid_cols}")

        # Find duplicates
        if not fuzzy:
            if strategy == DuplicateStrategy.MARK:
                result['is_duplicate'] = result.duplicated(subset=subset, keep=False)
            elif strategy == DuplicateStrategy.KEEP_FIRST:
                result = result.drop_duplicates(subset=subset, keep='first')
            elif strategy == DuplicateStrategy.KEEP_LAST:
                result = result.drop_duplicates(subset=subset, keep='last')
            elif strategy == DuplicateStrategy.KEEP_NONE:
                duplicates_mask = result.duplicated(subset=subset, keep=False)
                result = result[~duplicates_mask]
        else:
            result = self._handle_fuzzy_duplicates(result, subset, threshold, strategy)

        self._store_cleaned(result)

        # Generate report
        self._cleaning_report = {
            "strategy": strategy.value,
            "subset_columns": subset if subset else "all columns",
            "fuzzy_matching": fuzzy,
            "rows_before": len(data),
            "rows_after": len(result),
            "duplicates_removed": len(data) - len(result)
        }

        self.logger.info(f"Duplicate removal complete. Removed {self._cleaning_report['duplicates_removed']} rows")

        return result

    def find_duplicates(
        self,
        data: pd.DataFrame,
        subset: Optional[List[str]] = None,
        fuzzy: bool = False,
        threshold: float = 0.9
    ) -> pd.DataFrame:
        """
        Find and return only duplicate rows

        Args:
            data: DataFrame to search
            subset: Columns to consider
            fuzzy: Whether to use fuzzy matching
            threshold: Similarity threshold

        Returns:
            DataFrame containing only duplicate rows
        """
        if not fuzzy:
            duplicates_mask = data.duplicated(subset=subset, keep=False)
        else:
            duplicates_mask = self._find_fuzzy_duplicates(data, subset, threshold)

        return data[duplicates_mask].copy()

    def _find_fuzzy_duplicates(
        self,
        data: pd.DataFrame,
        subset: Optional[List[str]],
        threshold: float
    ) -> pd.Series:
        """
        Find duplicates using fuzzy string matching

        Uses SequenceMatcher for string similarity
        """
        self.logger.info(f"Finding fuzzy duplicates (threshold={threshold})")

        # Determine columns to compare
        compare_cols = subset if subset else data.select_dtypes(include=['object']).columns.tolist()

        if not compare_cols:
            # No string columns, fall back to exact matching
            return data.duplicated(subset=subset, keep=False)

        # Create a mask for duplicates
        duplicates_mask = pd.Series([False] * len(data), index=data.index)

        # For each row, find similar rows
        for i in range(len(data)):
            if duplicates_mask.iloc[i]:
                continue  # Already marked as duplicate

            # Compare with subsequent rows
            for j in range(i + 1, len(data)):
                if duplicates_mask.iloc[j]:
                    continue

                # Calculate similarity
                similarity = self._calculate_similarity(
                    data.iloc[i],
                    data.iloc[j],
                    compare_cols
                )

                if similarity >= threshold:
                    duplicates_mask.iloc[i] = True
                    duplicates_mask.iloc[j] = True

        return duplicates_mask

    def _calculate_similarity(
        self,
        row1: pd.Series,
        row2: pd.Series,
        columns: List[str]
    ) -> float:
        """
        Calculate average similarity between two rows

        Uses SequenceMatcher for string comparison
        """
        similarities = []

        for col in columns:
            val1 = str(row1[col]) if pd.notna(row1[col]) else ""
            val2 = str(row2[col]) if pd.notna(row2[col]) else ""

            # Use SequenceMatcher for string similarity
            similarity = SequenceMatcher(None, val1, val2).ratio()
            similarities.append(similarity)

        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _handle_fuzzy_duplicates(
        self,
        data: pd.DataFrame,
        subset: Optional[List[str]],
        threshold: float,
        strategy: DuplicateStrategy
    ) -> pd.DataFrame:
        """Handle fuzzy duplicates based on strategy"""
        duplicates_mask = self._find_fuzzy_duplicates(data, subset, threshold)

        if strategy == DuplicateStrategy.MARK:
            data['is_duplicate'] = duplicates_mask
            return data
        elif strategy == DuplicateStrategy.KEEP_NONE:
            return data[~duplicates_mask]
        else:
            # For KEEP_FIRST/KEEP_LAST, we need to group duplicates
            # This is complex for fuzzy matching, so we'll keep first by default
            result = []
            seen = set()

            for idx, row in data.iterrows():
                if duplicates_mask.loc[idx]:
                    # This is a duplicate, check if we've seen similar row
                    row_tuple = tuple(row[subset] if subset else row)
                    if row_tuple not in seen:
                        result.append(row)
                        seen.add(row_tuple)
                else:
                    result.append(row)

            return pd.DataFrame(result)

    def _get_duplicate_groups(
        self,
        data: pd.DataFrame,
        duplicates_mask: pd.Series,
        subset: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Get groups of duplicate rows

        Returns list of duplicate groups with their indices
        """
        if not duplicates_mask.any():
            return []

        duplicate_data = data[duplicates_mask]

        if subset is None:
            subset = data.columns.tolist()

        # Group by values in subset columns
        groups = duplicate_data.groupby(list(subset)).apply(
            lambda x: {
                "count": len(x),
                "indices": x.index.tolist(),
                "values": x.iloc[0][subset].to_dict() if subset else {}
            }
        ).tolist()

        return groups

    def merge_duplicates(
        self,
        data: pd.DataFrame,
        subset: List[str],
        merge_strategy: str = "first",
        aggregations: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Merge duplicate rows using aggregation

        Instead of dropping duplicates, aggregate their values

        Args:
            data: DataFrame to process
            subset: Columns that define duplicates
            merge_strategy: 'first', 'last', or 'aggregate'
            aggregations: Dict mapping column names to aggregation functions
                         (e.g., {'age': 'mean', 'score': 'sum'})

        Returns:
            DataFrame with merged duplicates
        """
        self.logger.info(f"Merging duplicates with strategy: {merge_strategy}")

        if merge_strategy in ['first', 'last']:
            return data.drop_duplicates(subset=subset, keep=merge_strategy)

        elif merge_strategy == 'aggregate':
            if aggregations is None:
                raise InvalidParameterError("aggregations required for 'aggregate' strategy")

            # Group by subset columns and aggregate others
            result = data.groupby(subset, as_index=False).agg(aggregations)

            return result

        else:
            raise InvalidParameterError(f"Unknown merge strategy: {merge_strategy}")

    def _generate_recommendations(self, n_duplicates: int, total_rows: int) -> List[str]:
        """Generate recommendations based on duplicate analysis"""
        recommendations = []

        if n_duplicates == 0:
            recommendations.append("✅ No duplicates found. Data is clean.")
            return recommendations

        duplicate_pct = (n_duplicates / total_rows * 100) if total_rows > 0 else 0

        if duplicate_pct > 20:
            recommendations.append(
                f"⚠️ High duplicate rate ({duplicate_pct:.1f}%). "
                "Investigate data collection process."
            )
        elif duplicate_pct > 5:
            recommendations.append(
                f"📊 Moderate duplicates ({duplicate_pct:.1f}%). "
                "Review and remove as needed."
            )
        else:
            recommendations.append(
                f"✓ Low duplicate rate ({duplicate_pct:.1f}%). "
                "Safe to remove."
            )

        # Suggest strategy
        if duplicate_pct < 1:
            recommendations.append("💡 Suggestion: Use KEEP_FIRST strategy to preserve order")
        elif duplicate_pct > 10:
            recommendations.append("💡 Suggestion: Review duplicates manually before removal")
        else:
            recommendations.append("💡 Suggestion: Safe to use KEEP_FIRST or KEEP_LAST")

        return recommendations

    def compare_rows(
        self,
        data: pd.DataFrame,
        index1: int,
        index2: int,
        subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare two specific rows in detail

        Args:
            data: DataFrame
            index1: First row index
            index2: Second row index
            subset: Columns to compare (None = all)

        Returns:
            Comparison result
        """
        if index1 not in data.index or index2 not in data.index:
            raise InvalidParameterError("Invalid row indices")

        row1 = data.loc[index1]
        row2 = data.loc[index2]

        compare_cols = subset if subset else data.columns.tolist()

        comparison = {
            "index1": index1,
            "index2": index2,
            "identical": row1.equals(row2),
            "differences": {},
            "similarities": {}
        }

        for col in compare_cols:
            val1 = row1[col]
            val2 = row2[col]

            if val1 == val2 or (pd.isna(val1) and pd.isna(val2)):
                comparison["similarities"][col] = str(val1)
            else:
                comparison["differences"][col] = {
                    "row1": str(val1),
                    "row2": str(val2)
                }

                # Calculate similarity for strings
                if isinstance(val1, str) and isinstance(val2, str):
                    similarity = SequenceMatcher(None, val1, val2).ratio()
                    comparison["differences"][col]["similarity"] = similarity

        return comparison

    def get_duplicate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed statistics about duplicates

        Returns:
            Dictionary with statistics
        """
        # Exact duplicates
        exact_duplicates = data.duplicated().sum()

        # Duplicates by column
        duplicate_by_column = {}
        for col in data.columns:
            col_duplicates = data[col].duplicated().sum()
            if col_duplicates > 0:
                duplicate_by_column[col] = int(col_duplicates)

        return {
            "total_rows": len(data),
            "exact_duplicates": int(exact_duplicates),
            "exact_duplicate_percentage": (exact_duplicates / len(data) * 100) if len(data) > 0 else 0,
            "unique_rows": len(data) - exact_duplicates,
            "duplicate_by_column": duplicate_by_column,
            "columns_with_duplicates": len(duplicate_by_column)
        }
