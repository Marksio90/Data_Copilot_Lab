"""
Data Copilot Lab - Code Generator
Automatic generation of data science code and ML pipelines
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from src.modules.ai_assistant.llm_integration import LLMIntegration, LLMModel
from src.core.exceptions import InvalidParameterError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineType(str, Enum):
    """ML pipeline types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"


class CodeTemplate(str, Enum):
    """Code templates"""
    DATA_LOADING = "data_loading"
    DATA_CLEANING = "data_cleaning"
    EDA = "eda"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    PREDICTION = "prediction"
    VISUALIZATION = "visualization"


class CodeGenerator:
    """
    Generate data science code automatically

    Features:
    - Generate complete ML pipelines
    - Create data analysis scripts
    - Generate visualization code
    - Custom code generation from natural language
    - Template-based generation
    """

    def __init__(
        self,
        llm_model: LLMModel = LLMModel.GPT35_TURBO,
        api_key: Optional[str] = None
    ):
        """
        Initialize Code Generator

        Args:
            llm_model: LLM model to use
            api_key: API key
        """
        self.logger = logger
        self.llm = LLMIntegration(model=llm_model, api_key=api_key, temperature=0.2)

    def generate_pipeline(
        self,
        pipeline_type: Union[str, PipelineType],
        data_info: Dict[str, Any],
        target: Optional[str] = None,
        requirements: Optional[List[str]] = None
    ) -> str:
        """
        Generate complete ML pipeline

        Args:
            pipeline_type: Type of ML pipeline
            data_info: Information about the data
            target: Target variable name
            requirements: Additional requirements

        Returns:
            Generated pipeline code
        """
        if isinstance(pipeline_type, str):
            pipeline_type = PipelineType(pipeline_type)

        self.logger.info(f"Generating {pipeline_type.value} pipeline")

        # Build context
        context = self._build_pipeline_context(pipeline_type, data_info, target, requirements)

        system_message = f"""You are an expert data scientist and Python programmer.
Generate a complete, production-ready {pipeline_type.value} pipeline.

The code should:
- Follow best practices
- Be well-commented
- Include error handling
- Be modular and reusable
- Use scikit-learn, pandas, and numpy

Return ONLY the Python code without explanations."""

        prompt = f"""Generate a complete {pipeline_type.value} pipeline with the following specifications:

{context}

Generate the complete pipeline code:"""

        code = self.llm.generate_text(prompt, system_message=system_message)

        return code

    def generate_from_description(
        self,
        description: str,
        context: Optional[str] = None,
        style: str = "concise"
    ) -> str:
        """
        Generate code from natural language description

        Args:
            description: What the code should do
            context: Additional context (imports, data structure, etc.)
            style: 'concise', 'verbose', or 'production'

        Returns:
            Generated code
        """
        self.logger.info(f"Generating code from description: {description[:50]}...")

        system_message = f"""You are an expert Python programmer for data science.
Generate {style} code that follows best practices.
Return ONLY the code without explanations."""

        prompt_parts = [f"Task: {description}"]

        if context:
            prompt_parts.append(f"\nContext:\n{context}")

        if style == "production":
            prompt_parts.append("\nInclude type hints, docstrings, and error handling.")
        elif style == "concise":
            prompt_parts.append("\nKeep it concise and focused.")

        prompt = "\n".join(prompt_parts)

        code = self.llm.generate_code(description, context=context)

        return code

    def generate_from_template(
        self,
        template: Union[str, CodeTemplate],
        parameters: Dict[str, Any]
    ) -> str:
        """
        Generate code from template

        Args:
            template: Template type
            parameters: Template parameters

        Returns:
            Generated code
        """
        if isinstance(template, str):
            template = CodeTemplate(template)

        self.logger.info(f"Generating code from template: {template.value}")

        template_prompts = {
            CodeTemplate.DATA_LOADING: self._generate_data_loading,
            CodeTemplate.DATA_CLEANING: self._generate_data_cleaning,
            CodeTemplate.EDA: self._generate_eda,
            CodeTemplate.FEATURE_ENGINEERING: self._generate_feature_engineering,
            CodeTemplate.MODEL_TRAINING: self._generate_model_training,
            CodeTemplate.MODEL_EVALUATION: self._generate_model_evaluation,
            CodeTemplate.PREDICTION: self._generate_prediction,
            CodeTemplate.VISUALIZATION: self._generate_visualization
        }

        if template not in template_prompts:
            raise InvalidParameterError(f"Template {template} not implemented")

        return template_prompts[template](parameters)

    def generate_test_code(
        self,
        code: str,
        framework: str = "pytest"
    ) -> str:
        """
        Generate test code for given code

        Args:
            code: Code to test
            framework: Test framework ('pytest' or 'unittest')

        Returns:
            Test code
        """
        self.logger.info(f"Generating {framework} tests")

        system_message = f"""You are an expert in {framework}. Generate comprehensive test cases.
Include edge cases, happy path, and error conditions."""

        prompt = f"""Generate {framework} tests for this code:

```python
{code}
```

Generate comprehensive test code:"""

        test_code = self.llm.generate_text(prompt, system_message=system_message)

        return test_code

    def optimize_code(
        self,
        code: str,
        optimization_goals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate optimized version of code

        Args:
            code: Code to optimize
            optimization_goals: List of goals ('speed', 'memory', 'readability')

        Returns:
            Dict with optimized code and explanations
        """
        self.logger.info("Optimizing code")

        goals_str = ", ".join(optimization_goals) if optimization_goals else "performance"

        system_message = f"""You are an expert Python optimizer.
Optimize code for: {goals_str}
Provide both the optimized code and explanations of changes."""

        prompt = f"""Optimize this code for {goals_str}:

```python
{code}
```

Provide:
1. Optimized code
2. Explanation of optimizations
3. Expected performance improvement"""

        response = self.llm.generate_text(prompt, system_message=system_message)

        # Parse response (try to extract code and explanation)
        parts = response.split("```python")
        if len(parts) > 1:
            code_part = parts[1].split("```")[0].strip()
            explanation = response.replace(f"```python\n{code_part}\n```", "").strip()
        else:
            code_part = response
            explanation = "See inline comments"

        return {
            "optimized_code": code_part,
            "explanation": explanation,
            "original_code": code
        }

    def refactor_code(
        self,
        code: str,
        refactoring_type: str = "general"
    ) -> str:
        """
        Refactor code

        Args:
            code: Code to refactor
            refactoring_type: 'general', 'modularity', 'naming', 'structure'

        Returns:
            Refactored code
        """
        self.logger.info(f"Refactoring code ({refactoring_type})")

        system_message = f"""You are an expert code refactorer.
Focus on {refactoring_type}.
Maintain functionality while improving code quality."""

        prompt = f"""Refactor this code (focus: {refactoring_type}):

```python
{code}
```

Return the refactored code:"""

        refactored = self.llm.generate_text(prompt, system_message=system_message)

        return refactored

    # Template generation methods

    def _generate_data_loading(self, params: Dict[str, Any]) -> str:
        """Generate data loading code"""
        file_type = params.get('file_type', 'csv')
        file_path = params.get('file_path', 'data.csv')

        description = f"Load data from {file_type} file at {file_path}"
        context = "Use pandas. Include error handling and logging."

        return self.llm.generate_code(description, context=context)

    def _generate_data_cleaning(self, params: Dict[str, Any]) -> str:
        """Generate data cleaning code"""
        operations = params.get('operations', ['handle_missing', 'remove_duplicates'])

        description = f"Clean dataset by: {', '.join(operations)}"
        context = """Use pandas and Data Copilot Lab's cleaning modules.
Include: missing value handling, duplicate removal, outlier detection."""

        return self.llm.generate_code(description, context=context)

    def _generate_eda(self, params: Dict[str, Any]) -> str:
        """Generate EDA code"""
        columns = params.get('columns', [])
        target = params.get('target')

        description = "Perform exploratory data analysis"
        if columns:
            description += f" on columns: {', '.join(columns)}"
        if target:
            description += f" with target variable: {target}"

        context = """Use Data Copilot Lab's EDA modules.
Include: descriptive statistics, distributions, correlations, visualizations."""

        return self.llm.generate_code(description, context=context)

    def _generate_feature_engineering(self, params: Dict[str, Any]) -> str:
        """Generate feature engineering code"""
        techniques = params.get('techniques', ['polynomial', 'interactions', 'scaling'])

        description = f"Apply feature engineering techniques: {', '.join(techniques)}"
        context = """Use Data Copilot Lab's FeatureEngineer.
Include feature selection, transformation, and creation."""

        return self.llm.generate_code(description, context=context)

    def _generate_model_training(self, params: Dict[str, Any]) -> str:
        """Generate model training code"""
        model_type = params.get('model_type', 'random_forest')
        task = params.get('task', 'classification')

        description = f"Train {model_type} model for {task}"
        context = f"""Use Data Copilot Lab's {task.capitalize()}Trainer.
Include train/test split, cross-validation, and hyperparameter tuning."""

        return self.llm.generate_code(description, context=context)

    def _generate_model_evaluation(self, params: Dict[str, Any]) -> str:
        """Generate model evaluation code"""
        task = params.get('task', 'classification')
        metrics = params.get('metrics', ['accuracy', 'precision', 'recall'])

        description = f"Evaluate {task} model using metrics: {', '.join(metrics)}"
        context = """Use Data Copilot Lab's ModelExplainer.
Include metrics calculation, confusion matrix, and model explanation."""

        return self.llm.generate_code(description, context=context)

    def _generate_prediction(self, params: Dict[str, Any]) -> str:
        """Generate prediction code"""
        model_path = params.get('model_path', 'model.pkl')

        description = f"Load model from {model_path} and make predictions"
        context = """Use Data Copilot Lab's ModelRegistry.
Include model loading, preprocessing, and prediction."""

        return self.llm.generate_code(description, context=context)

    def _generate_visualization(self, params: Dict[str, Any]) -> str:
        """Generate visualization code"""
        chart_type = params.get('chart_type', 'scatter')
        x = params.get('x')
        y = params.get('y')

        description = f"Create {chart_type} plot"
        if x and y:
            description += f" of {y} vs {x}"

        context = """Use Data Copilot Lab's VisualizationEngine.
Create interactive Plotly charts."""

        return self.llm.generate_code(description, context=context)

    def _build_pipeline_context(
        self,
        pipeline_type: PipelineType,
        data_info: Dict[str, Any],
        target: Optional[str],
        requirements: Optional[List[str]]
    ) -> str:
        """Build context for pipeline generation"""
        context_parts = []

        # Data info
        context_parts.append("Data Information:")
        if 'shape' in data_info:
            context_parts.append(f"- Shape: {data_info['shape']}")
        if 'columns' in data_info:
            context_parts.append(f"- Columns: {', '.join(data_info['columns'][:10])}")
        if 'dtypes' in data_info:
            context_parts.append(f"- Data types: {len(data_info['dtypes'])} features")

        # Target
        if target:
            context_parts.append(f"\nTarget variable: {target}")

        # Pipeline specifics
        context_parts.append(f"\nPipeline type: {pipeline_type.value}")

        if pipeline_type == PipelineType.CLASSIFICATION:
            context_parts.append("Include: data preprocessing, feature engineering, model training (try multiple algorithms), evaluation, and explainability.")
        elif pipeline_type == PipelineType.REGRESSION:
            context_parts.append("Include: data preprocessing, feature engineering, model training (linear and non-linear models), evaluation with residual analysis.")
        elif pipeline_type == PipelineType.CLUSTERING:
            context_parts.append("Include: data preprocessing, dimensionality reduction, clustering (try multiple algorithms), evaluation, and visualization.")

        # Requirements
        if requirements:
            context_parts.append(f"\nAdditional requirements:")
            for req in requirements:
                context_parts.append(f"- {req}")

        # Tools
        context_parts.append("\nUse Data Copilot Lab modules:")
        context_parts.append("- from src.modules.data_cleaning import MissingDataHandler, DataStandardizer")
        context_parts.append("- from src.modules.ml import FeatureEngineer, ClassificationTrainer, RegressionTrainer")
        context_parts.append("- from src.modules.ml import ModelExplainer, ModelRegistry")

        return "\n".join(context_parts)
