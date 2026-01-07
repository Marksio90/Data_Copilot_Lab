"""
Data Copilot Lab - AI Assistant Module
Conversational AI interface with code generation and smart suggestions
"""

from src.modules.ai_assistant.llm_integration import (
    LLMIntegration,
    LLMProvider,
    LLMModel,
)
from src.modules.ai_assistant.conversational_interface import (
    ConversationalInterface,
    IntentType,
)
from src.modules.ai_assistant.chat_history import (
    ChatHistory,
)
from src.modules.ai_assistant.code_generator import (
    CodeGenerator,
    PipelineType,
    CodeTemplate,
)
from src.modules.ai_assistant.smart_suggestions import (
    SmartSuggestions,
    SuggestionCategory,
)

__all__ = [
    # LLM Integration
    "LLMIntegration",
    "LLMProvider",
    "LLMModel",
    # Conversational Interface
    "ConversationalInterface",
    "IntentType",
    # Chat History
    "ChatHistory",
    # Code Generator
    "CodeGenerator",
    "PipelineType",
    "CodeTemplate",
    # Smart Suggestions
    "SmartSuggestions",
    "SuggestionCategory",
]
