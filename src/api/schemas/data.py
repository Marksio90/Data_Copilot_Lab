"""
Data Copilot Lab - Pydantic Schemas for Data Import API
Request and response models for data import endpoints
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class DataSourceType(str, Enum):
    """Supported data source types"""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    XML = "xml"
    SQL = "sql"


class CSVImportRequest(BaseModel):
    """Request model for CSV import"""
    file_path: str = Field(..., description="Path to CSV file")
    delimiter: Optional[str] = Field(None, description="Column delimiter (auto-detected if None)")
    encoding: Optional[str] = Field(None, description="File encoding (auto-detected if None)")
    has_header: bool = Field(True, description="Whether first row is header")
    skip_rows: Optional[int] = Field(None, description="Number of rows to skip")
    nrows: Optional[int] = Field(None, description="Number of rows to read")


class ExcelImportRequest(BaseModel):
    """Request model for Excel import"""
    file_path: str = Field(..., description="Path to Excel file")
    sheet_name: Union[str, int, None] = Field(0, description="Sheet name or index")
    has_header: bool = Field(True, description="Whether first row is header")
    skip_rows: Optional[int] = Field(None, description="Number of rows to skip")
    nrows: Optional[int] = Field(None, description="Number of rows to read")
    usecols: Optional[Union[str, List[int]]] = Field(None, description="Columns to read")


class JSONImportRequest(BaseModel):
    """Request model for JSON import"""
    file_path: str = Field(..., description="Path to JSON file")
    orient: str = Field("records", description="JSON orientation")
    normalize: bool = Field(True, description="Normalize nested structures")
    max_level: Optional[int] = Field(None, description="Maximum normalization level")
    record_path: Optional[Union[str, List[str]]] = Field(None, description="Path to records")

    @validator('orient')
    def validate_orient(cls, v):
        valid_orients = ['records', 'index', 'columns', 'values']
        if v not in valid_orients:
            raise ValueError(f"orient must be one of {valid_orients}")
        return v


class XMLImportRequest(BaseModel):
    """Request model for XML import"""
    file_path: str = Field(..., description="Path to XML file")
    record_tag: Optional[str] = Field(None, description="Tag name for records")
    include_attributes: bool = Field(True, description="Include XML attributes")
    parser: str = Field("etree", description="Parser to use (etree or beautifulsoup)")

    @validator('parser')
    def validate_parser(cls, v):
        valid_parsers = ['etree', 'beautifulsoup']
        if v not in valid_parsers:
            raise ValueError(f"parser must be one of {valid_parsers}")
        return v


class SQLConnectionParams(BaseModel):
    """SQL connection parameters"""
    dialect: str = Field(..., description="Database dialect (postgresql, mysql, sqlite, etc.)")
    host: Optional[str] = Field(None, description="Database host")
    port: Optional[int] = Field(None, description="Database port")
    database: str = Field(..., description="Database name or file path (for SQLite)")
    user: Optional[str] = Field(None, description="Database user")
    password: Optional[str] = Field(None, description="Database password")
    driver: Optional[str] = Field(None, description="Database driver")


class SQLImportRequest(BaseModel):
    """Request model for SQL import"""
    connection: Union[str, SQLConnectionParams] = Field(
        ...,
        description="Connection string or connection parameters"
    )
    query: Optional[str] = Field(None, description="SQL query to execute")
    table_name: Optional[str] = Field(None, description="Table name to import")
    schema: Optional[str] = Field(None, description="Database schema")
    chunksize: Optional[int] = Field(None, description="Rows per chunk")

    @validator('query', 'table_name')
    def validate_query_or_table(cls, v, values):
        if not v and not values.get('table_name') and not values.get('query'):
            raise ValueError("Either query or table_name must be provided")
        return v


class DataImportResponse(BaseModel):
    """Response model for data import"""
    success: bool = Field(..., description="Whether import was successful")
    dataset_id: str = Field(..., description="Unique identifier for the imported dataset")
    source_type: DataSourceType = Field(..., description="Type of data source")
    metadata: Dict[str, Any] = Field(..., description="Import metadata")
    preview: Optional[List[Dict[str, Any]]] = Field(None, description="Data preview")
    info: Dict[str, Any] = Field(..., description="Dataset information")
    message: Optional[str] = Field(None, description="Additional message")
    created_at: datetime = Field(default_factory=datetime.now, description="Import timestamp")


class DataPreviewRequest(BaseModel):
    """Request model for data preview"""
    dataset_id: str = Field(..., description="Dataset identifier")
    n_rows: int = Field(10, ge=1, le=1000, description="Number of rows to preview")
    offset: int = Field(0, ge=0, description="Row offset")


class DataPreviewResponse(BaseModel):
    """Response model for data preview"""
    dataset_id: str = Field(..., description="Dataset identifier")
    data: List[Dict[str, Any]] = Field(..., description="Preview data")
    total_rows: int = Field(..., description="Total number of rows")
    columns: List[str] = Field(..., description="Column names")


class DataInfoRequest(BaseModel):
    """Request model for dataset information"""
    dataset_id: str = Field(..., description="Dataset identifier")


class DataInfoResponse(BaseModel):
    """Response model for dataset information"""
    dataset_id: str = Field(..., description="Dataset identifier")
    n_rows: int = Field(..., description="Number of rows")
    n_columns: int = Field(..., description="Number of columns")
    columns: List[str] = Field(..., description="Column names")
    dtypes: Dict[str, str] = Field(..., description="Column data types")
    memory_usage: int = Field(..., description="Memory usage in bytes")
    null_counts: Dict[str, int] = Field(..., description="Null value counts per column")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")


class DatasetListResponse(BaseModel):
    """Response model for list of datasets"""
    datasets: List[Dict[str, Any]] = Field(..., description="List of datasets")
    total: int = Field(..., description="Total number of datasets")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Any] = Field(None, description="Additional error details")


class ColumnTypeSuggestion(BaseModel):
    """Column type conversion suggestion"""
    column: str = Field(..., description="Column name")
    current_type: str = Field(..., description="Current data type")
    suggested_type: str = Field(..., description="Suggested data type")
    reason: Optional[str] = Field(None, description="Reason for suggestion")


class TypeConversionRequest(BaseModel):
    """Request model for type conversions"""
    dataset_id: str = Field(..., description="Dataset identifier")
    conversions: Dict[str, str] = Field(
        ...,
        description="Column type conversions (column_name: target_type)"
    )


class TypeConversionResponse(BaseModel):
    """Response model for type conversions"""
    success: bool = Field(..., description="Whether conversion was successful")
    dataset_id: str = Field(..., description="Dataset identifier")
    conversions_applied: Dict[str, str] = Field(..., description="Applied conversions")
    errors: Optional[Dict[str, str]] = Field(None, description="Conversion errors")


# SQL-specific schemas

class TableListRequest(BaseModel):
    """Request for listing database tables"""
    connection: Union[str, SQLConnectionParams] = Field(..., description="Database connection")
    schema: Optional[str] = Field(None, description="Schema name")


class TableListResponse(BaseModel):
    """Response with list of tables"""
    tables: List[str] = Field(..., description="List of table names")
    schema: Optional[str] = Field(None, description="Schema name")
    total: int = Field(..., description="Total number of tables")


class TableInfoRequest(BaseModel):
    """Request for table information"""
    connection: Union[str, SQLConnectionParams] = Field(..., description="Database connection")
    table_name: str = Field(..., description="Table name")
    schema: Optional[str] = Field(None, description="Schema name")


class TableInfoResponse(BaseModel):
    """Response with table information"""
    table_name: str = Field(..., description="Table name")
    schema: Optional[str] = Field(None, description="Schema name")
    row_count: int = Field(..., description="Number of rows")
    column_count: int = Field(..., description="Number of columns")
    columns: List[Dict[str, Any]] = Field(..., description="Column information")
    primary_keys: List[str] = Field(..., description="Primary key columns")
    foreign_keys: List[Dict[str, Any]] = Field(..., description="Foreign key constraints")
    indexes: List[Dict[str, Any]] = Field(..., description="Table indexes")


# Excel-specific schemas

class SheetListRequest(BaseModel):
    """Request for listing Excel sheets"""
    file_path: str = Field(..., description="Path to Excel file")


class SheetListResponse(BaseModel):
    """Response with list of sheets"""
    file_path: str = Field(..., description="Path to Excel file")
    sheets: List[str] = Field(..., description="List of sheet names")
    total: int = Field(..., description="Total number of sheets")
