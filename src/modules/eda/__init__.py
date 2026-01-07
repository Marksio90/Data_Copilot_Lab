"""
Data Copilot Lab - Exploratory Data Analysis Module
Comprehensive EDA tools including statistics, visualization, and correlation analysis
"""

from src.modules.eda.statistics import StatisticalAnalyzer
from src.modules.eda.visualization import VisualizationEngine, ChartType
from src.modules.eda.correlation import CorrelationAnalyzer
from src.modules.eda.auto_eda import AutoEDA

__all__ = [
    "StatisticalAnalyzer",
    "VisualizationEngine",
    "ChartType",
    "CorrelationAnalyzer",
    "AutoEDA",
]
