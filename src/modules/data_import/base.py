"""
Data Copilot Lab - Base Classes for Data Import
Abstract base classes defining the interface for all data importers
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from src.core.exceptions import DataImportError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataImporter(ABC):
    """
    Abstract base class for all data importers

    All concrete importers must implement the import_data method
    """

    def __init__(self):
        self.logger = logger
        self._data: Optional[pd.DataFrame] = None
        self._metadata: Dict[str, Any] = {}

    @abstractmethod
    def import_data(
        self,
        source: Union[str, Path, Any],
        **kwargs
    ) -> pd.DataFrame:
        """
        Import data from a source

        Args:
            source: Data source (file path, connection string, etc.)
            **kwargs: Additional importer-specific parameters

        Returns:
            DataFrame containing the imported data

        Raises:
            DataImportError: If import fails
        """
        pass

    @abstractmethod
    def validate_source(self, source: Union[str, Path, Any]) -> bool:
        """
        Validate that the source is accessible and readable

        Args:
            source: Data source to validate

        Returns:
            True if source is valid, False otherwise
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the imported data

        Returns:
            Dictionary containing metadata
        """
        return self._metadata

    def get_preview(self, n_rows: int = 10) -> pd.DataFrame:
        """
        Get a preview of the imported data

        Args:
            n_rows: Number of rows to preview

        Returns:
            DataFrame with first n_rows
        """
        if self._data is None:
            raise DataImportError("No data has been imported yet")
        return self._data.head(n_rows)

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the imported data

        Returns:
            Dictionary with data information (shape, dtypes, etc.)
        """
        if self._data is None:
            raise DataImportError("No data has been imported yet")

        return {
            "n_rows": len(self._data),
            "n_columns": len(self._data.columns),
            "columns": list(self._data.columns),
            "dtypes": {col: str(dtype) for col, dtype in self._data.dtypes.items()},
            "memory_usage": self._data.memory_usage(deep=True).sum(),
            "null_counts": self._data.isnull().sum().to_dict()
        }


class FileImporter(DataImporter):
    """
    Base class for file-based importers

    Provides common functionality for importing from files
    """

    def __init__(self):
        super().__init__()
        self._file_path: Optional[Path] = None

    def validate_source(self, source: Union[str, Path]) -> bool:
        """
        Validate that the file exists and is readable

        Args:
            source: File path

        Returns:
            True if file is valid, False otherwise
        """
        try:
            file_path = Path(source)

            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                return False

            if not file_path.is_file():
                self.logger.error(f"Path is not a file: {file_path}")
                return False

            if not file_path.stat().st_size > 0:
                self.logger.error(f"File is empty: {file_path}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating file: {e}")
            return False

    def get_file_info(self) -> Dict[str, Any]:
        """
        Get information about the source file

        Returns:
            Dictionary with file information
        """
        if self._file_path is None:
            raise DataImportError("No file has been imported yet")

        stat = self._file_path.stat()

        return {
            "file_name": self._file_path.name,
            "file_path": str(self._file_path.absolute()),
            "file_size": stat.st_size,
            "file_size_mb": stat.st_size / (1024 * 1024),
            "modified_time": stat.st_mtime,
            "extension": self._file_path.suffix
        }


class DatabaseImporter(DataImporter):
    """
    Base class for database importers

    Provides common functionality for importing from databases
    """

    def __init__(self):
        super().__init__()
        self._connection_string: Optional[str] = None
        self._query: Optional[str] = None

    @abstractmethod
    def connect(self, connection_string: str, **kwargs) -> bool:
        """
        Establish connection to the database

        Args:
            connection_string: Database connection string
            **kwargs: Additional connection parameters

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self):
        """
        Close database connection
        """
        pass

    @abstractmethod
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results

        Args:
            query: SQL query to execute

        Returns:
            DataFrame with query results
        """
        pass

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get information about the database connection

        Returns:
            Dictionary with connection information
        """
        return {
            "connection_string": self._connection_string,
            "query": self._query
        }
