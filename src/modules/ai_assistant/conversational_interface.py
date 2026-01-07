"""
Data Copilot Lab - Conversational Interface
Natural language interface for data science tasks
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.modules.ai_assistant.llm_integration import LLMIntegration, LLMModel
from src.modules.ai_assistant.chat_history import ChatHistory
from src.core.exceptions import InvalidParameterError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IntentType(str, Enum):
    """User intent types"""
    DATA_QUERY = "data_query"  # Query/filter data
    DATA_ANALYSIS = "data_analysis"  # Analyze data
    VISUALIZATION = "visualization"  # Create visualizations
    MODEL_TRAINING = "model_training"  # Train ML models
    CODE_GENERATION = "code_generation"  # Generate code
    EXPLANATION = "explanation"  # Explain concepts/results
    SUGGESTION = "suggestion"  # Get suggestions
    GENERAL_QUESTION = "general_question"  # General questions
    UNKNOWN = "unknown"


class ConversationalInterface:
    """
    Natural Language Interface for Data Science

    Allows users to interact with data and ML models using
    natural language. Understands context and maintains
    conversation history.
    """

    def __init__(
        self,
        llm_model: Union[str, LLMModel] = LLMModel.GPT35_TURBO,
        api_key: Optional[str] = None,
        enable_history: bool = True,
        max_history_length: int = 20
    ):
        """
        Initialize Conversational Interface

        Args:
            llm_model: LLM model to use
            api_key: OpenAI API key
            enable_history: Enable conversation history
            max_history_length: Maximum messages in history
        """
        self.logger = logger
        self.llm = LLMIntegration(model=llm_model, api_key=api_key)

        # Conversation history
        self.enable_history = enable_history
        if enable_history:
            self.history = ChatHistory(max_length=max_history_length)
        else:
            self.history = None

        # Context
        self.context = {
            "current_dataset": None,
            "dataset_info": None,
            "recent_operations": [],
            "trained_models": [],
            "available_variables": {}
        }

        self.logger.info("Conversational Interface initialized")

    def chat(
        self,
        message: str,
        include_context: bool = True,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send a message and get response

        Args:
            message: User message
            include_context: Include data context in prompt
            stream: Enable streaming

        Returns:
            Response dictionary with 'response', 'intent', 'actions'
        """
        self.logger.info(f"User message: {message[:100]}...")

        # Detect intent
        intent = self._detect_intent(message)
        self.logger.debug(f"Detected intent: {intent}")

        # Prepare messages for LLM
        messages = self._prepare_messages(message, include_context)

        # Get LLM response
        llm_response = self.llm.chat_completion(messages, stream=stream)

        response_text = llm_response['content']

        # Parse response for actions
        actions = self._extract_actions(response_text, intent)

        # Store in history
        if self.enable_history:
            self.history.add_message("user", message)
            self.history.add_message("assistant", response_text, metadata={
                "intent": intent.value,
                "actions": actions
            })

        result = {
            "response": response_text,
            "intent": intent.value,
            "actions": actions,
            "tokens_used": llm_response.get('usage', {}).get('total_tokens', 0)
        }

        return result

    def ask_about_data(
        self,
        question: str,
        data: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Ask question about data

        Args:
            question: Question about the data
            data: DataFrame to analyze (None = use current dataset)

        Returns:
            Answer
        """
        if data is not None:
            self.set_current_dataset(data)

        # Get data context
        data_context = self._get_data_context()

        context_str = f"""Available data information:
- Shape: {data_context.get('shape')}
- Columns: {data_context.get('columns')}
- Data types: {data_context.get('dtypes')}
- Sample values available"""

        answer = self.llm.answer_question(
            question,
            context=context_str,
            conversation_history=self._get_conversation_history()
        )

        # Store in history
        if self.enable_history:
            self.history.add_message("user", question, metadata={"type": "data_question"})
            self.history.add_message("assistant", answer)

        return answer

    def generate_analysis_plan(
        self,
        goal: str,
        data_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate step-by-step analysis plan

        Args:
            goal: Analysis goal
            data_info: Information about the data

        Returns:
            Analysis plan with steps
        """
        self.logger.info(f"Generating analysis plan for: {goal}")

        system_message = """You are a data science expert. Create a detailed, step-by-step analysis plan.
Return a JSON object with:
{
  "goal": "the analysis goal",
  "steps": [
    {"step": 1, "action": "description", "rationale": "why"},
    ...
  ],
  "expected_outcomes": ["list of expected outcomes"],
  "estimated_time": "time estimate"
}"""

        context = ""
        if data_info:
            context = f"\n\nData information:\n{json.dumps(data_info, indent=2)}"

        prompt = f"Create an analysis plan for: {goal}{context}"

        response = self.llm.generate_text(
            prompt,
            system_message=system_message,
            temperature=0.4
        )

        # Parse JSON
        try:
            plan = json.loads(response)
        except json.JSONDecodeError:
            # Fallback to plain text
            plan = {
                "goal": goal,
                "raw_plan": response
            }

        return plan

    def explain_result(
        self,
        result: Any,
        result_type: str = "general",
        context: Optional[str] = None
    ) -> str:
        """
        Explain analysis result in plain language

        Args:
            result: Result to explain (dict, number, DataFrame, etc.)
            result_type: Type of result ('model_metrics', 'statistical_test', 'correlation', etc.)
            context: Additional context

        Returns:
            Plain language explanation
        """
        self.logger.info(f"Explaining result of type: {result_type}")

        system_message = """You are a data science communicator. Explain technical results in plain,
understandable language. Be concise but thorough."""

        # Format result for LLM
        if isinstance(result, dict):
            result_str = json.dumps(result, indent=2)
        elif isinstance(result, pd.DataFrame):
            result_str = result.to_string()
        else:
            result_str = str(result)

        prompt = f"Result type: {result_type}\n\nResult:\n{result_str}"
        if context:
            prompt += f"\n\nContext: {context}"

        prompt += "\n\nExplain this result in plain language:"

        explanation = self.llm.generate_text(prompt, system_message=system_message)

        return explanation

    def suggest_next_steps(
        self,
        current_state: str,
        goal: Optional[str] = None
    ) -> List[str]:
        """
        Suggest next analysis steps

        Args:
            current_state: Description of current analysis state
            goal: Overall analysis goal

        Returns:
            List of suggested next steps
        """
        system_message = """You are a data science advisor. Suggest logical next steps
for data analysis. Return a JSON array of suggestions."""

        prompt = f"Current state: {current_state}"
        if goal:
            prompt += f"\nGoal: {goal}"

        prompt += "\n\nWhat should be done next?"

        response = self.llm.generate_text(
            prompt,
            system_message=system_message,
            temperature=0.5
        )

        # Parse JSON
        try:
            suggestions = json.loads(response)
            if isinstance(suggestions, dict) and 'suggestions' in suggestions:
                suggestions = suggestions['suggestions']
        except json.JSONDecodeError:
            # Split by newlines as fallback
            suggestions = [s.strip() for s in response.split('\n') if s.strip()]

        return suggestions

    def interpret_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Interpret complex query and extract intent

        Args:
            query: Natural language query

        Returns:
            Structured interpretation
        """
        data_context = self._get_data_context()

        interpretation = self.llm.analyze_data_query(query, data_context)

        return interpretation

    def set_current_dataset(self, data: pd.DataFrame, name: Optional[str] = None):
        """Set current dataset for context"""
        self.context['current_dataset'] = data
        self.context['dataset_info'] = {
            "name": name or "current_dataset",
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "sample": data.head(3).to_dict('records')
        }
        self.logger.info(f"Current dataset set: {data.shape}")

    def add_trained_model(self, model_id: str, model_info: Dict[str, Any]):
        """Add information about trained model"""
        self.context['trained_models'].append({
            "id": model_id,
            "info": model_info,
            "timestamp": datetime.now().isoformat()
        })

    def clear_context(self):
        """Clear conversation context"""
        self.context = {
            "current_dataset": None,
            "dataset_info": None,
            "recent_operations": [],
            "trained_models": [],
            "available_variables": {}
        }
        self.logger.info("Context cleared")

    def clear_history(self):
        """Clear conversation history"""
        if self.enable_history:
            self.history.clear()
            self.logger.info("History cleared")

    def get_conversation_summary(self) -> str:
        """Get summary of conversation"""
        if not self.enable_history:
            return "History not enabled"

        history = self.history.get_recent_messages(10)

        # Create summary prompt
        messages_text = "\n".join([
            f"{msg['role']}: {msg['content'][:200]}"
            for msg in history
        ])

        prompt = f"Summarize this conversation:\n\n{messages_text}"

        summary = self.llm.generate_text(
            prompt,
            system_message="You are a helpful assistant. Provide concise summaries.",
            temperature=0.3
        )

        return summary

    def _detect_intent(self, message: str) -> IntentType:
        """Detect user intent from message"""
        message_lower = message.lower()

        # Simple keyword-based intent detection
        if any(word in message_lower for word in ['train', 'model', 'predict', 'classify', 'regression']):
            return IntentType.MODEL_TRAINING

        elif any(word in message_lower for word in ['plot', 'visualize', 'chart', 'graph', 'show']):
            return IntentType.VISUALIZATION

        elif any(word in message_lower for word in ['filter', 'select', 'where', 'query', 'find']):
            return IntentType.DATA_QUERY

        elif any(word in message_lower for word in ['analyze', 'statistics', 'correlation', 'distribution']):
            return IntentType.DATA_ANALYSIS

        elif any(word in message_lower for word in ['explain', 'what is', 'how does', 'why']):
            return IntentType.EXPLANATION

        elif any(word in message_lower for word in ['suggest', 'recommend', 'what should']):
            return IntentType.SUGGESTION

        elif any(word in message_lower for word in ['code', 'generate', 'write', 'create function']):
            return IntentType.CODE_GENERATION

        elif '?' in message:
            return IntentType.GENERAL_QUESTION

        return IntentType.UNKNOWN

    def _prepare_messages(self, user_message: str, include_context: bool) -> List[Dict[str, str]]:
        """Prepare messages for LLM"""
        messages = []

        # System message
        system_prompt = """You are a helpful data science assistant. You help users with:
- Data analysis and exploration
- Creating visualizations
- Training machine learning models
- Writing Python code for data science
- Explaining results and concepts

Be concise, practical, and provide code examples when relevant."""

        if include_context and self.context.get('dataset_info'):
            system_prompt += f"\n\nCurrent dataset information:\n{json.dumps(self.context['dataset_info'], indent=2)}"

        messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        if self.enable_history:
            history_messages = self._get_conversation_history()
            messages.extend(history_messages)

        # Add current message
        messages.append({"role": "user", "content": user_message})

        return messages

    def _get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history for LLM"""
        if not self.enable_history:
            return []

        history = self.history.get_recent_messages(10)

        return [
            {"role": msg['role'], "content": msg['content']}
            for msg in history
        ]

    def _get_data_context(self) -> Dict[str, Any]:
        """Get current data context"""
        return self.context.get('dataset_info', {})

    def _extract_actions(self, response: str, intent: IntentType) -> List[Dict[str, Any]]:
        """Extract actionable items from response"""
        actions = []

        # Look for code blocks
        if "```" in response:
            code_blocks = []
            parts = response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Code blocks at odd indices
                    lines = part.strip().split('\n')
                    language = lines[0].strip() if lines[0].strip() in ['python', 'py', 'sql'] else 'python'
                    code = '\n'.join(lines[1:]) if lines[0].strip() in ['python', 'py', 'sql'] else part.strip()
                    code_blocks.append({"language": language, "code": code})

            if code_blocks:
                actions.append({
                    "type": "code_execution",
                    "code_blocks": code_blocks
                })

        # Look for specific action keywords
        if intent == IntentType.VISUALIZATION:
            actions.append({"type": "create_visualization"})

        elif intent == IntentType.MODEL_TRAINING:
            actions.append({"type": "train_model"})

        return actions

    def get_context(self) -> Dict[str, Any]:
        """Get current context"""
        return self.context

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = self.llm.get_usage_stats()

        if self.enable_history:
            stats['messages_in_history'] = self.history.count_messages()

        return stats
