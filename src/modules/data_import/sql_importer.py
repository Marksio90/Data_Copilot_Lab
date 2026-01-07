"""
Data Copilot Lab - SQL Database Importer
Import data from SQL databases (PostgreSQL, MySQL, SQLite, etc.)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.core.config import settings
from src.core.exceptions import (
    DatabaseConnectionError,
    DataImportError,
    DataParsingError
)
from src.modules.data_import.base import DatabaseImporter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SQLImporter(DatabaseImporter):
    """
    SQL database importer supporting:
    - PostgreSQL
    - MySQL/MariaDB
    - SQLite
    - MS SQL Server
    - Oracle (with appropriate drivers)

    Features:
    - Query execution
    - Table inspection
    - Schema browsing
    - Connection pooling
    """

    SUPPORTED_DIALECTS = [
        'postgresql',
        'mysql',
        'sqlite',
        'mssql',
        'oracle'
    ]

    def __init__(self):
        super().__init__()
        self._engine: Optional[Engine] = None
        self._dialect: Optional[str] = None
        self._is_connected: bool = False

    def validate_source(self, source: Union[str, Path]) -> bool:
        """
        Validate database connection string

        Args:
            source: Database connection string

        Returns:
            True if valid, False otherwise
        """
        try:
            parsed = urlparse(str(source))
            dialect = parsed.scheme.split('+')[0] if '+' in parsed.scheme else parsed.scheme

            if dialect not in self.SUPPORTED_DIALECTS:
                logger.error(f"Unsupported database dialect: {dialect}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating connection string: {e}")
            return False

    def connect(
        self,
        connection_string: str,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        **kwargs
    ) -> bool:
        """
        Establish connection to the database

        Args:
            connection_string: SQLAlchemy connection string
                Examples:
                - sqlite:///path/to/database.db
                - postgresql://user:password@localhost:5432/dbname
                - mysql://user:password@localhost:3306/dbname
            echo: Whether to log all SQL statements
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            **kwargs: Additional engine parameters

        Returns:
            True if connection successful

        Raises:
            DatabaseConnectionError: If connection fails
        """
        try:
            if not self.validate_source(connection_string):
                raise DatabaseConnectionError("Invalid connection string")

            logger.info("Connecting to database...")

            # Parse connection string to get dialect
            parsed = urlparse(connection_string)
            self._dialect = parsed.scheme.split('+')[0]

            # Create engine
            engine_params = {
                'echo': echo,
            }

            # Add pool settings for non-SQLite databases
            if self._dialect != 'sqlite':
                engine_params.update({
                    'pool_size': pool_size,
                    'max_overflow': max_overflow,
                    'pool_pre_ping': True,  # Verify connections before using
                })

            engine_params.update(kwargs)

            self._engine = create_engine(connection_string, **engine_params)

            # Test connection
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            self._connection_string = connection_string
            self._is_connected = True

            logger.info(f"Successfully connected to {self._dialect} database")
            return True

        except SQLAlchemyError as e:
            logger.error(f"Database connection error: {e}", exc_info=True)
            raise DatabaseConnectionError(f"Failed to connect to database: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error during connection: {e}", exc_info=True)
            raise DatabaseConnectionError(f"Connection failed: {str(e)}")

    def disconnect(self):
        """Close database connection"""
        if self._engine:
            self._engine.dispose()
            self._is_connected = False
            logger.info("Database connection closed")

    def import_data(
        self,
        source: Union[str, Dict],
        query: Optional[str] = None,
        table_name: Optional[str] = None,
        schema: Optional[str] = None,
        chunksize: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Import data from SQL database

        Args:
            source: Connection string or dict with connection params
            query: SQL query to execute (alternative to table_name)
            table_name: Table name to import (alternative to query)
            schema: Database schema name
            chunksize: Number of rows per chunk (None = load all at once)
            **kwargs: Additional pandas.read_sql parameters

        Returns:
            DataFrame with imported data

        Raises:
            DataImportError: If import fails
        """
        try:
            # Connect if not already connected
            if not self._is_connected:
                if isinstance(source, dict):
                    connection_string = self._build_connection_string(source)
                else:
                    connection_string = str(source)

                self.connect(connection_string)

            # Determine what to execute
            if query is None and table_name is None:
                raise DataImportError(
                    "Either 'query' or 'table_name' must be provided"
                )

            if query:
                logger.info(f"Executing query: {query[:100]}...")
                self._query = query
                sql = query
            else:
                logger.info(f"Reading table: {table_name}")
                self._query = f"SELECT * FROM {table_name}"
                if schema:
                    sql = f"SELECT * FROM {schema}.{table_name}"
                else:
                    sql = f"SELECT * FROM {table_name}"

            # Execute query
            if chunksize:
                # Read in chunks
                logger.info(f"Reading data in chunks of {chunksize} rows")
                chunks = pd.read_sql(
                    sql,
                    self._engine,
                    chunksize=chunksize,
                    **kwargs
                )
                self._data = pd.concat(chunks, ignore_index=True)
            else:
                # Read all at once
                self._data = pd.read_sql(sql, self._engine, **kwargs)

            # Store metadata
            self._metadata = {
                'source_type': 'sql',
                'dialect': self._dialect,
                'query': self._query,
                'table_name': table_name,
                'schema': schema,
                'rows_imported': len(self._data),
                'columns_imported': len(self._data.columns),
            }

            logger.info(
                f"Successfully imported {len(self._data)} rows "
                f"and {len(self._data.columns)} columns"
            )

            return self._data

        except SQLAlchemyError as e:
            logger.error(f"SQL execution error: {e}", exc_info=True)
            raise DataParsingError(f"Failed to execute SQL: {str(e)}")

        except Exception as e:
            logger.error(f"Error importing from database: {e}", exc_info=True)
            raise DataImportError(f"Failed to import data: {str(e)}")

    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute SQL query and return results

        Args:
            query: SQL query to execute
            params: Query parameters (for parameterized queries)

        Returns:
            DataFrame with query results
        """
        if not self._is_connected:
            raise DatabaseConnectionError("Not connected to database")

        try:
            logger.info(f"Executing query: {query[:100]}...")

            if params:
                result = pd.read_sql(text(query), self._engine, params=params)
            else:
                result = pd.read_sql(query, self._engine)

            logger.info(f"Query returned {len(result)} rows")
            return result

        except SQLAlchemyError as e:
            logger.error(f"Query execution error: {e}", exc_info=True)
            raise DataParsingError(f"Query failed: {str(e)}")

    def get_tables(self, schema: Optional[str] = None) -> List[str]:
        """
        Get list of tables in the database

        Args:
            schema: Schema name (None = default schema)

        Returns:
            List of table names
        """
        if not self._is_connected:
            raise DatabaseConnectionError("Not connected to database")

        try:
            inspector = inspect(self._engine)
            tables = inspector.get_table_names(schema=schema)
            logger.info(f"Found {len(tables)} tables")
            return tables

        except SQLAlchemyError as e:
            logger.error(f"Error getting tables: {e}")
            return []

    def get_schemas(self) -> List[str]:
        """
        Get list of schemas in the database

        Returns:
            List of schema names
        """
        if not self._is_connected:
            raise DatabaseConnectionError("Not connected to database")

        try:
            inspector = inspect(self._engine)
            schemas = inspector.get_schema_names()
            logger.info(f"Found {len(schemas)} schemas")
            return schemas

        except SQLAlchemyError as e:
            logger.error(f"Error getting schemas: {e}")
            return []

    def get_table_info(
        self,
        table_name: str,
        schema: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get information about a table

        Args:
            table_name: Table name
            schema: Schema name

        Returns:
            Dictionary with table information
        """
        if not self._is_connected:
            raise DatabaseConnectionError("Not connected to database")

        try:
            inspector = inspect(self._engine)

            # Get columns
            columns = inspector.get_columns(table_name, schema=schema)

            # Get primary keys
            pk = inspector.get_pk_constraint(table_name, schema=schema)

            # Get foreign keys
            fk = inspector.get_foreign_keys(table_name, schema=schema)

            # Get indexes
            indexes = inspector.get_indexes(table_name, schema=schema)

            # Get row count (with a simple query)
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            if schema:
                count_query = f"SELECT COUNT(*) as count FROM {schema}.{table_name}"

            row_count = pd.read_sql(count_query, self._engine).iloc[0]['count']

            return {
                'table_name': table_name,
                'schema': schema,
                'row_count': row_count,
                'column_count': len(columns),
                'columns': [
                    {
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col['nullable'],
                        'default': col.get('default'),
                    }
                    for col in columns
                ],
                'primary_keys': pk.get('constrained_columns', []),
                'foreign_keys': [
                    {
                        'columns': fk_item['constrained_columns'],
                        'referred_table': fk_item['referred_table'],
                        'referred_columns': fk_item['referred_columns'],
                    }
                    for fk_item in fk
                ],
                'indexes': [
                    {
                        'name': idx['name'],
                        'columns': idx['column_names'],
                        'unique': idx['unique'],
                    }
                    for idx in indexes
                ]
            }

        except SQLAlchemyError as e:
            logger.error(f"Error getting table info: {e}")
            raise DataImportError(f"Failed to get table info: {str(e)}")

    def test_connection(self) -> bool:
        """
        Test if database connection is alive

        Returns:
            True if connection is active, False otherwise
        """
        if not self._is_connected or not self._engine:
            return False

        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def _build_connection_string(self, params: Dict[str, str]) -> str:
        """
        Build connection string from parameters

        Args:
            params: Dictionary with connection parameters
                Required keys: dialect, host, database, user, password
                Optional keys: port, driver

        Returns:
            SQLAlchemy connection string
        """
        dialect = params.get('dialect')
        driver = params.get('driver')
        user = params.get('user')
        password = params.get('password')
        host = params.get('host')
        port = params.get('port')
        database = params.get('database')

        if not all([dialect, database]):
            raise DataImportError(
                "Missing required connection parameters: dialect, database"
            )

        # Handle SQLite (file-based)
        if dialect == 'sqlite':
            return f"sqlite:///{database}"

        # Handle other databases
        if not all([user, password, host]):
            raise DataImportError(
                "Missing required connection parameters: user, password, host"
            )

        # Build connection string
        if driver:
            conn_str = f"{dialect}+{driver}://"
        else:
            conn_str = f"{dialect}://"

        conn_str += f"{user}:{password}@{host}"

        if port:
            conn_str += f":{port}"

        conn_str += f"/{database}"

        return conn_str

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    def __del__(self):
        """Destructor - ensure connection is closed"""
        if hasattr(self, '_engine') and self._engine:
            self.disconnect()
