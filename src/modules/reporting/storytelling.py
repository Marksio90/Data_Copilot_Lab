"""
Data Copilot Lab - Storytelling Tools
Transform data analysis into compelling narratives
"""

from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

from src.modules.ai_assistant.llm_integration import LLMIntegration, LLMModel
from src.core.exceptions import InvalidParameterError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StoryStructure(str, Enum):
    """Story structure templates"""
    PROBLEM_SOLUTION = "problem_solution"
    BEFORE_AFTER = "before_after"
    JOURNEY = "journey"
    COMPARISON = "comparison"
    PYRAMID = "pyramid"  # Most important first
    CHRONOLOGICAL = "chronological"


class NarrativeElement(str, Enum):
    """Elements of a data story"""
    CONTEXT = "context"
    CHALLENGE = "challenge"
    DATA = "data"
    INSIGHT = "insight"
    ACTION = "action"
    OUTCOME = "outcome"


class StorytellingEngine:
    """
    Transform data analysis into compelling narratives

    Creates data-driven stories with:
    - Clear narrative structure
    - Context and background
    - Key insights highlighted
    - Actionable recommendations
    - Visualizations that support the story
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_model: LLMModel = LLMModel.GPT35_TURBO,
        api_key: Optional[str] = None
    ):
        """
        Initialize Storytelling Engine

        Args:
            use_llm: Use LLM for narrative generation
            llm_model: LLM model
            api_key: API key
        """
        self.logger = logger
        self.use_llm = use_llm

        if use_llm:
            self.llm = LLMIntegration(model=llm_model, api_key=api_key, temperature=0.7)

        self.story_elements = []

    def create_story(
        self,
        title: str,
        structure: Union[str, StoryStructure] = StoryStructure.PROBLEM_SOLUTION,
        audience: str = "business_stakeholders"
    ) -> "DataStory":
        """
        Create a new data story

        Args:
            title: Story title
            structure: Narrative structure
            audience: Target audience

        Returns:
            DataStory object
        """
        return DataStory(title, structure, audience, llm=self.llm if self.use_llm else None)

    def generate_narrative(
        self,
        insights: List[Dict[str, Any]],
        data_context: Dict[str, Any],
        goal: str
    ) -> str:
        """
        Generate narrative from insights

        Args:
            insights: List of insights
            data_context: Context about the data
            goal: Analysis goal

        Returns:
            Generated narrative
        """
        if not self.use_llm:
            return self._generate_template_narrative(insights, data_context, goal)

        self.logger.info("Generating narrative from insights")

        system_message = """You are a data storytelling expert. Create compelling narratives from data insights.

Your narratives should:
- Start with context and the business question
- Present insights in a logical flow
- Use clear, jargon-free language
- Connect insights to business impact
- End with actionable recommendations

Write in a professional yet engaging style."""

        # Prepare insights summary
        insights_text = "\n".join([
            f"- {insight['title']}: {insight['description']}"
            for insight in insights[:5]
        ])

        prompt = f"""Goal: {goal}

Data Context:
- Shape: {data_context.get('shape')}
- Columns: {data_context.get('columns', [])}

Key Insights:
{insights_text}

Create a compelling narrative (3-4 paragraphs) that tells the story of this analysis."""

        narrative = self.llm.generate_text(prompt, system_message=system_message)

        return narrative

    def _generate_template_narrative(
        self,
        insights: List[Dict[str, Any]],
        data_context: Dict[str, Any],
        goal: str
    ) -> str:
        """Generate narrative using template (without LLM)"""
        parts = []

        parts.append(f"# Analysis: {goal}\n")
        parts.append(f"We analyzed a dataset with {data_context.get('shape', ['?', '?'])[0]} records ")
        parts.append(f"and {data_context.get('shape', ['?', '?'])[1]} variables.\n\n")

        parts.append("## Key Findings\n")
        for insight in insights[:5]:
            parts.append(f"- **{insight['title']}**: {insight['description']}\n")

        parts.append("\n## Recommendations\n")
        for insight in insights[:3]:
            if insight.get('recommendation'):
                parts.append(f"- {insight['recommendation']}\n")

        return "".join(parts)


class DataStory:
    """
    A complete data-driven story

    Combines narrative, data, visualizations, and insights
    into a coherent storyline
    """

    def __init__(
        self,
        title: str,
        structure: Union[str, StoryStructure],
        audience: str = "business_stakeholders",
        llm: Optional[LLMIntegration] = None
    ):
        """
        Initialize Data Story

        Args:
            title: Story title
            structure: Narrative structure
            audience: Target audience
            llm: LLM integration (optional)
        """
        self.logger = logger
        self.title = title
        self.structure = StoryStructure(structure) if isinstance(structure, str) else structure
        self.audience = audience
        self.llm = llm

        self.elements = []
        self.visualizations = []
        self.metadata = {
            "title": title,
            "structure": self.structure.value,
            "audience": audience
        }

    def add_context(
        self,
        context: str,
        background: Optional[str] = None
    ):
        """
        Add context/background

        Args:
            context: Main context
            background: Additional background
        """
        self.elements.append({
            "type": NarrativeElement.CONTEXT,
            "content": context,
            "background": background
        })

    def add_challenge(
        self,
        challenge: str,
        why_important: str
    ):
        """
        Add business challenge/problem

        Args:
            challenge: Description of challenge
            why_important: Why it matters
        """
        self.elements.append({
            "type": NarrativeElement.CHALLENGE,
            "challenge": challenge,
            "importance": why_important
        })

    def add_data_description(
        self,
        description: str,
        data_summary: Dict[str, Any]
    ):
        """
        Add data description

        Args:
            description: Description of data
            data_summary: Summary statistics
        """
        self.elements.append({
            "type": NarrativeElement.DATA,
            "description": description,
            "summary": data_summary
        })

    def add_insight(
        self,
        insight: str,
        evidence: Any,
        visualization: Optional[go.Figure] = None
    ):
        """
        Add key insight

        Args:
            insight: Insight description
            evidence: Supporting evidence (data, metrics)
            visualization: Supporting chart
        """
        element = {
            "type": NarrativeElement.INSIGHT,
            "insight": insight,
            "evidence": evidence
        }

        if visualization:
            viz_id = f"viz_{len(self.visualizations)}"
            self.visualizations.append({
                "id": viz_id,
                "figure": visualization
            })
            element["visualization_id"] = viz_id

        self.elements.append(element)

    def add_action(
        self,
        action: str,
        rationale: str,
        priority: str = "medium"
    ):
        """
        Add recommended action

        Args:
            action: Action to take
            rationale: Why this action
            priority: Action priority
        """
        self.elements.append({
            "type": NarrativeElement.ACTION,
            "action": action,
            "rationale": rationale,
            "priority": priority
        })

    def add_outcome(
        self,
        expected_outcome: str,
        success_metrics: Optional[List[str]] = None
    ):
        """
        Add expected outcome

        Args:
            expected_outcome: Description of expected outcome
            success_metrics: How to measure success
        """
        self.elements.append({
            "type": NarrativeElement.OUTCOME,
            "outcome": expected_outcome,
            "metrics": success_metrics or []
        })

    def generate_narrative_flow(self) -> str:
        """
        Generate narrative text from story elements

        Returns:
            Complete narrative
        """
        if self.llm:
            return self._generate_llm_narrative()
        else:
            return self._generate_template_narrative()

    def _generate_llm_narrative(self) -> str:
        """Generate narrative using LLM"""
        system_message = f"""You are a data storytelling expert writing for {self.audience}.

Create a compelling narrative following the {self.structure.value} structure.

Guidelines:
- Use clear, jargon-free language
- Connect data to business impact
- Build a logical flow
- End with actionable takeaways"""

        # Prepare elements summary
        elements_text = []
        for element in self.elements:
            etype = element['type']
            if etype == NarrativeElement.CONTEXT:
                elements_text.append(f"Context: {element['content']}")
            elif etype == NarrativeElement.CHALLENGE:
                elements_text.append(f"Challenge: {element['challenge']}")
            elif etype == NarrativeElement.INSIGHT:
                elements_text.append(f"Insight: {element['insight']}")
            elif etype == NarrativeElement.ACTION:
                elements_text.append(f"Action: {element['action']}")
            elif etype == NarrativeElement.OUTCOME:
                elements_text.append(f"Expected Outcome: {element['outcome']}")

        prompt = f"""Story Title: {self.title}

Story Elements:
{chr(10).join(elements_text)}

Create a compelling narrative (4-5 paragraphs) that weaves these elements into a coherent story."""

        narrative = self.llm.generate_text(prompt, system_message=system_message)

        return narrative

    def _generate_template_narrative(self) -> str:
        """Generate narrative using template"""
        parts = [f"# {self.title}\n\n"]

        for element in self.elements:
            etype = element['type']

            if etype == NarrativeElement.CONTEXT:
                parts.append(f"## Context\n{element['content']}\n\n")

            elif etype == NarrativeElement.CHALLENGE:
                parts.append(f"## The Challenge\n{element['challenge']}\n\n")
                parts.append(f"**Why it matters:** {element['importance']}\n\n")

            elif etype == NarrativeElement.DATA:
                parts.append(f"## The Data\n{element['description']}\n\n")

            elif etype == NarrativeElement.INSIGHT:
                parts.append(f"### Insight\n{element['insight']}\n\n")

            elif etype == NarrativeElement.ACTION:
                parts.append(f"### Recommended Action\n**{element['action']}**\n")
                parts.append(f"*Rationale:* {element['rationale']}\n\n")

            elif etype == NarrativeElement.OUTCOME:
                parts.append(f"## Expected Outcome\n{element['outcome']}\n\n")

        return "".join(parts)

    def get_story_structure(self) -> Dict[str, Any]:
        """Get story structure"""
        return {
            "title": self.title,
            "structure": self.structure.value,
            "audience": self.audience,
            "elements": self.elements,
            "visualizations_count": len(self.visualizations)
        }

    def export_to_dict(self) -> Dict[str, Any]:
        """Export story as dictionary"""
        return {
            "metadata": self.metadata,
            "elements": self.elements,
            "visualizations": [
                {"id": viz["id"], "figure": "plotly_figure"}
                for viz in self.visualizations
            ]
        }


class StoryTemplate:
    """Pre-built story templates for common scenarios"""

    @staticmethod
    def sales_performance_story(
        sales_data: Dict[str, Any],
        insights: List[Dict[str, Any]]
    ) -> DataStory:
        """Create sales performance story"""
        story = DataStory(
            "Sales Performance Analysis",
            StoryStructure.BEFORE_AFTER,
            audience="sales_leadership"
        )

        story.add_context(
            "Sales performance analysis covering the most recent period",
            background="Understanding sales trends is critical for strategic planning"
        )

        story.add_data_description(
            f"Analysis of {sales_data.get('total_transactions', 0)} transactions",
            data_summary=sales_data
        )

        for insight in insights[:3]:
            story.add_insight(
                insight['description'],
                evidence=insight.get('metric')
            )

        return story

    @staticmethod
    def customer_churn_story(
        churn_rate: float,
        risk_factors: List[str]
    ) -> DataStory:
        """Create customer churn story"""
        story = DataStory(
            "Customer Churn Analysis",
            StoryStructure.PROBLEM_SOLUTION,
            audience="customer_success"
        )

        story.add_challenge(
            f"Customer churn rate is {churn_rate:.1f}%",
            why_important="High churn impacts revenue and growth"
        )

        story.add_insight(
            f"Identified {len(risk_factors)} key churn risk factors",
            evidence={"risk_factors": risk_factors}
        )

        story.add_action(
            "Implement proactive retention program targeting high-risk customers",
            rationale="Early intervention can reduce churn by 30-40%",
            priority="high"
        )

        return story

    @staticmethod
    def marketing_campaign_story(
        campaign_data: Dict[str, Any]
    ) -> DataStory:
        """Create marketing campaign story"""
        story = DataStory(
            "Marketing Campaign Performance",
            StoryStructure.COMPARISON,
            audience="marketing_team"
        )

        story.add_context(
            "Evaluation of recent marketing campaigns across channels"
        )

        roi = campaign_data.get('roi', 0)
        story.add_insight(
            f"Campaign achieved {roi:.1f}% ROI",
            evidence=campaign_data
        )

        story.add_outcome(
            "Optimize budget allocation to highest-performing channels",
            success_metrics=["Increased ROI", "Lower CAC", "Higher conversion rate"]
        )

        return story
