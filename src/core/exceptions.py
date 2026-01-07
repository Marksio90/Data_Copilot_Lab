"""
Data Copilot Lab - Custom Exceptions
Centralized exception handling
"""

from typing import Any, Optional


class DataCopilotException(Exception):
    """Base exception for Data Copilot Lab"""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Any] = None
    ):
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details
        super().__init__(self.message)


# Data Import Exceptions
class DataImportError(DataCopilotException):
    """Base exception for data import errors"""
    pass


class FileNotFoundError(DataImportError):
    """File not found error"""
    pass


class UnsupportedFileFormatError(DataImportError):
    """Unsupported file format error"""
    pass


class DataParsingError(DataImportError):
    """Error parsing data"""
    pass


class DatabaseConnectionError(DataImportError):
    """Database connection error"""
    pass


# Data Cleaning Exceptions
class DataCleaningError(DataCopilotException):
    """Base exception for data cleaning errors"""
    pass


class InvalidDataError(DataCleaningError):
    """Invalid data error"""
    pass


class MissingDataError(DataCleaningError):
    """Missing data error"""
    pass


# ML Exceptions
class MLError(DataCopilotException):
    """Base exception for ML errors"""
    pass


class ModelTrainingError(MLError):
    """Error during model training"""
    pass


class ModelPredictionError(MLError):
    """Error during model prediction"""
    pass


class InvalidModelError(MLError):
    """Invalid model error"""
    pass


class ModelNotFoundError(MLError):
    """Model not found error"""
    pass


# AI Assistant Exceptions
class AIAssistantError(DataCopilotException):
    """Base exception for AI assistant errors"""
    pass


class LLMAPIError(AIAssistantError):
    """LLM API error"""
    pass


class InvalidPromptError(AIAssistantError):
    """Invalid prompt error"""
    pass


# Storage Exceptions
class StorageError(DataCopilotException):
    """Base exception for storage errors"""
    pass


class FileUploadError(StorageError):
    """File upload error"""
    pass


class FileDownloadError(StorageError):
    """File download error"""
    pass


class StorageQuotaExceededError(StorageError):
    """Storage quota exceeded"""
    pass


# Validation Exceptions
class ValidationError(DataCopilotException):
    """Base exception for validation errors"""
    pass


class InvalidParameterError(ValidationError):
    """Invalid parameter error"""
    pass


class MissingParameterError(ValidationError):
    """Missing required parameter"""
    pass


# Authentication Exceptions
class AuthenticationError(DataCopilotException):
    """Base exception for authentication errors"""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Invalid credentials"""
    pass


class UnauthorizedError(AuthenticationError):
    """Unauthorized access"""
    pass


class TokenExpiredError(AuthenticationError):
    """Token expired"""
    pass


# General Exceptions
class ConfigurationError(DataCopilotException):
    """Configuration error"""
    pass


class ResourceNotFoundError(DataCopilotException):
    """Resource not found"""
    pass


class RateLimitExceededError(DataCopilotException):
    """Rate limit exceeded"""
    pass


class TimeoutError(DataCopilotException):
    """Operation timeout"""
    pass
