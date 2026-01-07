"""
Data Copilot Lab - Reporting Module
Professional reports, insights, storytelling, and business dashboards
"""

from src.modules.reporting.report_generator import (
    ReportGenerator,
    ReportFormat,
    ReportSection,
)
from src.modules.reporting.insight_generator import (
    InsightGenerator,
    InsightType,
    InsightPriority,
)
from src.modules.reporting.storytelling import (
    StorytellingEngine,
    DataStory,
    StoryStructure,
    NarrativeElement,
    StoryTemplate,
)
from src.modules.reporting.dashboard import (
    DashboardBuilder,
    DashboardLayout,
    MetricType,
    DashboardTemplate,
)

__all__ = [
    # Report Generator
    "ReportGenerator",
    "ReportFormat",
    "ReportSection",
    # Insight Generator
    "InsightGenerator",
    "InsightType",
    "InsightPriority",
    # Storytelling
    "StorytellingEngine",
    "DataStory",
    "StoryStructure",
    "NarrativeElement",
    "StoryTemplate",
    # Dashboard
    "DashboardBuilder",
    "DashboardLayout",
    "MetricType",
    "DashboardTemplate",
]
