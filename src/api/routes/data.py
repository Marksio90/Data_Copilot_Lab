"""
Data Copilot Lab - Data Import API Routes
FastAPI routes for data import functionality
"""

import uuid
from pathlib import Path
from typing import Dict

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse

from src.api.schemas.data import (
    CSVImportRequest,
    DataImportResponse,
    DataInfoRequest,
    DataInfoResponse,
    DataPreviewRequest,
    DataPreviewResponse,
    DatasetListResponse,
    ExcelImportRequest,
    JSONImportRequest,
    SQLImportRequest,
    SheetListRequest,
    SheetListResponse,
    TableInfoRequest,
    TableInfoResponse,
    TableListRequest,
    TableListResponse,
    TypeConversionRequest,
    TypeConversionResponse,
    XMLImportRequest,
)
from src.core.config import settings
from src.core.exceptions import DataImportError, DatabaseConnectionError
from src.modules.data_import.csv_importer import CSVImporter
from src.modules.data_import.excel_importer import ExcelImporter
from src.modules.data_import.json_importer import JSONImporter, XMLImporter
from src.modules.data_import.sql_importer import SQLImporter
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter()

# In-memory storage for datasets (in production, use database or cache)
_datasets: Dict[str, pd.DataFrame] = {}
_metadata: Dict[str, dict] = {}


def _store_dataset(df: pd.DataFrame, metadata: dict) -> str:
    """Store dataset and return unique ID"""
    dataset_id = str(uuid.uuid4())
    _datasets[dataset_id] = df
    _metadata[dataset_id] = metadata
    logger.info(f"Stored dataset {dataset_id}")
    return dataset_id


def _get_dataset(dataset_id: str) -> pd.DataFrame:
    """Retrieve dataset by ID"""
    if dataset_id not in _datasets:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )
    return _datasets[dataset_id]


# CSV Import Endpoints

@router.post("/import/csv", response_model=DataImportResponse, status_code=status.HTTP_201_CREATED)
async def import_csv(request: CSVImportRequest):
    """
    Import data from CSV file

    - **file_path**: Path to CSV file
    - **delimiter**: Column delimiter (auto-detected if None)
    - **encoding**: File encoding (auto-detected if None)
    - **has_header**: Whether first row is header
    """
    try:
        logger.info(f"Importing CSV file: {request.file_path}")

        importer = CSVImporter()
        df = importer.import_data(
            source=request.file_path,
            delimiter=request.delimiter,
            encoding=request.encoding,
            has_header=request.has_header,
            skip_rows=request.skip_rows,
            nrows=request.nrows
        )

        metadata = importer.get_metadata()
        dataset_id = _store_dataset(df, metadata)

        return DataImportResponse(
            success=True,
            dataset_id=dataset_id,
            source_type="csv",
            metadata=metadata,
            preview=df.head(5).to_dict('records'),
            info=importer.get_info(),
            message=f"Successfully imported {len(df)} rows"
        )

    except DataImportError as e:
        logger.error(f"CSV import error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during CSV import"
        )


@router.post("/import/upload/csv", response_model=DataImportResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload and import CSV file

    Handles file upload and imports data in one operation
    """
    try:
        # Save uploaded file
        file_path = settings.upload_dir / file.filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        logger.info(f"File uploaded: {file_path}")

        # Import using CSV importer
        importer = CSVImporter()
        df = importer.import_data(source=file_path)

        metadata = importer.get_metadata()
        dataset_id = _store_dataset(df, metadata)

        return DataImportResponse(
            success=True,
            dataset_id=dataset_id,
            source_type="csv",
            metadata=metadata,
            preview=df.head(5).to_dict('records'),
            info=importer.get_info(),
            message=f"Successfully uploaded and imported {len(df)} rows"
        )

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )


# Excel Import Endpoints

@router.post("/import/excel", response_model=DataImportResponse)
async def import_excel(request: ExcelImportRequest):
    """
    Import data from Excel file

    - **file_path**: Path to Excel file
    - **sheet_name**: Sheet name or index (0-based)
    - **has_header**: Whether first row is header
    """
    try:
        logger.info(f"Importing Excel file: {request.file_path}")

        importer = ExcelImporter()
        df = importer.import_data(
            source=request.file_path,
            sheet_name=request.sheet_name,
            has_header=request.has_header,
            skip_rows=request.skip_rows,
            nrows=request.nrows,
            usecols=request.usecols
        )

        metadata = importer.get_metadata()
        dataset_id = _store_dataset(df, metadata)

        return DataImportResponse(
            success=True,
            dataset_id=dataset_id,
            source_type="excel",
            metadata=metadata,
            preview=df.head(5).to_dict('records'),
            info=importer.get_info(),
            message=f"Successfully imported {len(df)} rows"
        )

    except DataImportError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during Excel import"
        )


@router.post("/excel/sheets", response_model=SheetListResponse)
async def get_excel_sheets(request: SheetListRequest):
    """Get list of sheets in Excel file"""
    try:
        importer = ExcelImporter()
        # Import to get sheet names
        importer.import_data(source=request.file_path, sheet_name=None)
        sheets = importer.get_sheet_names()

        return SheetListResponse(
            file_path=request.file_path,
            sheets=sheets,
            total=len(sheets)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read sheets: {str(e)}"
        )


# JSON Import Endpoints

@router.post("/import/json", response_model=DataImportResponse)
async def import_json(request: JSONImportRequest):
    """
    Import data from JSON file

    - **file_path**: Path to JSON file
    - **orient**: JSON orientation
    - **normalize**: Normalize nested structures
    """
    try:
        logger.info(f"Importing JSON file: {request.file_path}")

        importer = JSONImporter()
        df = importer.import_data(
            source=request.file_path,
            orient=request.orient,
            normalize=request.normalize,
            max_level=request.max_level,
            record_path=request.record_path
        )

        metadata = importer.get_metadata()
        dataset_id = _store_dataset(df, metadata)

        return DataImportResponse(
            success=True,
            dataset_id=dataset_id,
            source_type="json",
            metadata=metadata,
            preview=df.head(5).to_dict('records'),
            info=importer.get_info(),
            message=f"Successfully imported {len(df)} rows"
        )

    except DataImportError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during JSON import"
        )


# XML Import Endpoints

@router.post("/import/xml", response_model=DataImportResponse)
async def import_xml(request: XMLImportRequest):
    """
    Import data from XML file

    - **file_path**: Path to XML file
    - **record_tag**: Tag name for records
    - **include_attributes**: Include XML attributes
    """
    try:
        logger.info(f"Importing XML file: {request.file_path}")

        importer = XMLImporter()
        df = importer.import_data(
            source=request.file_path,
            record_tag=request.record_tag,
            include_attributes=request.include_attributes,
            parser=request.parser
        )

        metadata = importer.get_metadata()
        dataset_id = _store_dataset(df, metadata)

        return DataImportResponse(
            success=True,
            dataset_id=dataset_id,
            source_type="xml",
            metadata=metadata,
            preview=df.head(5).to_dict('records'),
            info=importer.get_info(),
            message=f"Successfully imported {len(df)} rows"
        )

    except DataImportError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during XML import"
        )


# SQL Import Endpoints

@router.post("/import/sql", response_model=DataImportResponse)
async def import_sql(request: SQLImportRequest):
    """
    Import data from SQL database

    - **connection**: Connection string or parameters
    - **query**: SQL query to execute
    - **table_name**: Table name to import
    """
    try:
        logger.info("Importing from SQL database")

        importer = SQLImporter()
        df = importer.import_data(
            source=request.connection,
            query=request.query,
            table_name=request.table_name,
            schema=request.schema,
            chunksize=request.chunksize
        )

        metadata = importer.get_metadata()
        dataset_id = _store_dataset(df, metadata)

        # Disconnect after import
        importer.disconnect()

        return DataImportResponse(
            success=True,
            dataset_id=dataset_id,
            source_type="sql",
            metadata=metadata,
            preview=df.head(5).to_dict('records'),
            info=importer.get_info(),
            message=f"Successfully imported {len(df)} rows"
        )

    except DatabaseConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection failed: {str(e)}"
        )
    except DataImportError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during SQL import"
        )


@router.post("/sql/tables", response_model=TableListResponse)
async def get_sql_tables(request: TableListRequest):
    """Get list of tables in database"""
    try:
        importer = SQLImporter()
        importer.connect(
            connection_string=request.connection
            if isinstance(request.connection, str)
            else importer._build_connection_string(request.connection.dict())
        )

        tables = importer.get_tables(schema=request.schema)
        importer.disconnect()

        return TableListResponse(
            tables=tables,
            schema=request.schema,
            total=len(tables)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get tables: {str(e)}"
        )


@router.post("/sql/table/info", response_model=TableInfoResponse)
async def get_sql_table_info(request: TableInfoRequest):
    """Get information about a database table"""
    try:
        importer = SQLImporter()
        importer.connect(
            connection_string=request.connection
            if isinstance(request.connection, str)
            else importer._build_connection_string(request.connection.dict())
        )

        table_info = importer.get_table_info(
            table_name=request.table_name,
            schema=request.schema
        )
        importer.disconnect()

        return TableInfoResponse(**table_info)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get table info: {str(e)}"
        )


# Dataset Management Endpoints

@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets():
    """Get list of all imported datasets"""
    datasets = []
    for dataset_id, metadata in _metadata.items():
        df = _datasets[dataset_id]
        datasets.append({
            "dataset_id": dataset_id,
            "source_type": metadata.get("source_type"),
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "metadata": metadata
        })

    return DatasetListResponse(
        datasets=datasets,
        total=len(datasets)
    )


@router.post("/dataset/preview", response_model=DataPreviewResponse)
async def preview_dataset(request: DataPreviewRequest):
    """Get preview of dataset"""
    try:
        df = _get_dataset(request.dataset_id)

        # Get subset of data
        start_idx = request.offset
        end_idx = start_idx + request.n_rows
        preview_df = df.iloc[start_idx:end_idx]

        return DataPreviewResponse(
            dataset_id=request.dataset_id,
            data=preview_df.to_dict('records'),
            total_rows=len(df),
            columns=list(df.columns)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/dataset/info", response_model=DataInfoResponse)
async def get_dataset_info(request: DataInfoRequest):
    """Get detailed information about dataset"""
    try:
        df = _get_dataset(request.dataset_id)
        metadata = _metadata.get(request.dataset_id, {})

        return DataInfoResponse(
            dataset_id=request.dataset_id,
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            memory_usage=df.memory_usage(deep=True).sum(),
            null_counts=df.isnull().sum().to_dict(),
            metadata=metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    if dataset_id not in _datasets:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_id} not found"
        )

    del _datasets[dataset_id]
    if dataset_id in _metadata:
        del _metadata[dataset_id]

    logger.info(f"Deleted dataset {dataset_id}")

    return {"success": True, "message": f"Dataset {dataset_id} deleted"}


# Utility Endpoints

@router.get("/formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "formats": [
            {
                "type": "csv",
                "extensions": CSVImporter.SUPPORTED_EXTENSIONS,
                "description": "Comma-separated values files"
            },
            {
                "type": "excel",
                "extensions": ExcelImporter.SUPPORTED_EXTENSIONS,
                "description": "Microsoft Excel files"
            },
            {
                "type": "json",
                "extensions": JSONImporter.SUPPORTED_EXTENSIONS,
                "description": "JSON files"
            },
            {
                "type": "xml",
                "extensions": XMLImporter.SUPPORTED_EXTENSIONS,
                "description": "XML files"
            },
            {
                "type": "sql",
                "dialects": SQLImporter.SUPPORTED_DIALECTS,
                "description": "SQL databases"
            }
        ]
    }
