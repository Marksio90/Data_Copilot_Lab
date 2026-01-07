"""
Data Copilot Lab - Visualization Engine
Create interactive visualizations using Plotly
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.exceptions import InvalidParameterError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChartType(str, Enum):
    """Supported chart types"""
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    HEATMAP = "heatmap"
    CORRELATION = "correlation"
    PAIRPLOT = "pairplot"
    DISTRIBUTION = "distribution"
    TIME_SERIES = "timeseries"
    CANDLESTICK = "candlestick"


class VisualizationEngine:
    """
    Create interactive visualizations using Plotly

    Supports 15+ chart types for EDA
    """

    def __init__(self, theme: str = "plotly"):
        self.logger = logger
        self.theme = theme

    def create_chart(
        self,
        data: pd.DataFrame,
        chart_type: Union[str, ChartType],
        x: Optional[str] = None,
        y: Optional[str] = None,
        color: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create a chart

        Args:
            data: DataFrame
            chart_type: Type of chart
            x: Column for x-axis
            y: Column for y-axis
            color: Column for color coding
            title: Chart title
            **kwargs: Additional chart-specific parameters

        Returns:
            Plotly Figure object
        """
        if isinstance(chart_type, str):
            chart_type = ChartType(chart_type)

        self.logger.info(f"Creating {chart_type.value} chart")

        if chart_type == ChartType.HISTOGRAM:
            return self.histogram(data, x, color, title, **kwargs)
        elif chart_type == ChartType.BOX:
            return self.box_plot(data, x, y, color, title, **kwargs)
        elif chart_type == ChartType.VIOLIN:
            return self.violin_plot(data, x, y, color, title, **kwargs)
        elif chart_type == ChartType.SCATTER:
            return self.scatter_plot(data, x, y, color, title, **kwargs)
        elif chart_type == ChartType.LINE:
            return self.line_plot(data, x, y, color, title, **kwargs)
        elif chart_type == ChartType.BAR:
            return self.bar_chart(data, x, y, color, title, **kwargs)
        elif chart_type == ChartType.PIE:
            return self.pie_chart(data, x, y, title, **kwargs)
        elif chart_type == ChartType.HEATMAP:
            return self.heatmap(data, title, **kwargs)
        elif chart_type == ChartType.CORRELATION:
            return self.correlation_heatmap(data, title, **kwargs)
        else:
            raise InvalidParameterError(f"Chart type {chart_type} not implemented in create_chart")

    def histogram(
        self,
        data: pd.DataFrame,
        column: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
        bins: int = 30,
        **kwargs
    ) -> go.Figure:
        """Create histogram"""
        if column not in data.columns:
            raise InvalidParameterError(f"Column '{column}' not found")

        title = title or f"Distribution of {column}"

        fig = px.histogram(
            data,
            x=column,
            color=color,
            nbins=bins,
            title=title,
            template=self.theme,
            **kwargs
        )

        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Count",
            showlegend=bool(color)
        )

        return fig

    def box_plot(
        self,
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: Optional[str] = None,
        color: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create box plot"""
        if y and y not in data.columns:
            raise InvalidParameterError(f"Column '{y}' not found")

        title = title or f"Box Plot{' of ' + y if y else ''}"

        fig = px.box(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            template=self.theme,
            **kwargs
        )

        return fig

    def violin_plot(
        self,
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: Optional[str] = None,
        color: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create violin plot"""
        if y and y not in data.columns:
            raise InvalidParameterError(f"Column '{y}' not found")

        title = title or f"Violin Plot{' of ' + y if y else ''}"

        fig = px.violin(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            template=self.theme,
            box=True,
            **kwargs
        )

        return fig

    def scatter_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
        size: Optional[str] = None,
        trendline: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create scatter plot"""
        if x not in data.columns or y not in data.columns:
            raise InvalidParameterError(f"Columns '{x}' or '{y}' not found")

        title = title or f"{y} vs {x}"

        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            size=size,
            title=title,
            template=self.theme,
            trendline=trendline,
            **kwargs
        )

        return fig

    def line_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        color: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create line plot"""
        if x not in data.columns:
            raise InvalidParameterError(f"Column '{x}' not found")

        if isinstance(y, str):
            y_cols = [y]
        else:
            y_cols = y

        for col in y_cols:
            if col not in data.columns:
                raise InvalidParameterError(f"Column '{col}' not found")

        title = title or f"Line Plot"

        fig = px.line(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            template=self.theme,
            **kwargs
        )

        return fig

    def bar_chart(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
        orientation: str = 'v',
        **kwargs
    ) -> go.Figure:
        """Create bar chart"""
        if x not in data.columns or y not in data.columns:
            raise InvalidParameterError(f"Columns '{x}' or '{y}' not found")

        title = title or f"{y} by {x}"

        fig = px.bar(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            template=self.theme,
            orientation=orientation,
            **kwargs
        )

        return fig

    def pie_chart(
        self,
        data: pd.DataFrame,
        names: str,
        values: str,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create pie chart"""
        if names not in data.columns or values not in data.columns:
            raise InvalidParameterError(f"Columns '{names}' or '{values}' not found")

        title = title or f"Distribution of {values}"

        fig = px.pie(
            data,
            names=names,
            values=values,
            title=title,
            template=self.theme,
            **kwargs
        )

        return fig

    def heatmap(
        self,
        data: pd.DataFrame,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create heatmap from DataFrame"""
        title = title or "Heatmap"

        # Use only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])

        fig = px.imshow(
            numeric_data,
            title=title,
            template=self.theme,
            aspect="auto",
            **kwargs
        )

        return fig

    def correlation_heatmap(
        self,
        data: pd.DataFrame,
        title: Optional[str] = None,
        method: str = 'pearson',
        **kwargs
    ) -> go.Figure:
        """Create correlation heatmap"""
        title = title or f"Correlation Heatmap ({method})"

        # Calculate correlation
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr(method=method)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate='%{text}',
            textfont={"size": 10},
            **kwargs
        ))

        fig.update_layout(
            title=title,
            template=self.theme,
            xaxis_title="",
            yaxis_title="",
            width=800,
            height=800
        )

        return fig

    def pair_plot(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        color: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """Create pair plot (scatter plot matrix)"""
        title = title or "Pair Plot"

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()[:5]  # Limit to 5

        fig = px.scatter_matrix(
            data,
            dimensions=columns,
            color=color,
            title=title,
            template=self.theme
        )

        fig.update_traces(diagonal_visible=False)

        return fig

    def distribution_plot(
        self,
        data: pd.DataFrame,
        column: str,
        title: Optional[str] = None,
        show_rug: bool = True,
        show_hist: bool = True,
        show_curve: bool = True
    ) -> go.Figure:
        """Create distribution plot with histogram and KDE"""
        if column not in data.columns:
            raise InvalidParameterError(f"Column '{column}' not found")

        title = title or f"Distribution of {column}"

        series = data[column].dropna()

        fig = go.Figure()

        # Histogram
        if show_hist:
            fig.add_trace(go.Histogram(
                x=series,
                name="Histogram",
                opacity=0.7,
                nbinsx=30
            ))

        # KDE curve (approximation using histogram with many bins)
        if show_curve:
            hist, bin_edges = np.histogram(series, bins=100, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            fig.add_trace(go.Scatter(
                x=bin_centers,
                y=hist,
                name="Density",
                mode='lines',
                line=dict(color='red', width=2)
            ))

        # Rug plot
        if show_rug:
            fig.add_trace(go.Scatter(
                x=series,
                y=[0] * len(series),
                mode='markers',
                name="Rug",
                marker=dict(symbol='line-ns-open', size=10, color='black'),
                showlegend=False
            ))

        fig.update_layout(
            title=title,
            xaxis_title=column,
            yaxis_title="Density / Count",
            template=self.theme,
            barmode='overlay'
        )

        return fig

    def time_series_plot(
        self,
        data: pd.DataFrame,
        date_column: str,
        value_columns: Union[str, List[str]],
        title: Optional[str] = None,
        show_trend: bool = False,
        **kwargs
    ) -> go.Figure:
        """Create time series plot"""
        if date_column not in data.columns:
            raise InvalidParameterError(f"Column '{date_column}' not found")

        if isinstance(value_columns, str):
            value_columns = [value_columns]

        for col in value_columns:
            if col not in data.columns:
                raise InvalidParameterError(f"Column '{col}' not found")

        title = title or "Time Series"

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data = data.copy()
            data[date_column] = pd.to_datetime(data[date_column])

        fig = go.Figure()

        for col in value_columns:
            fig.add_trace(go.Scatter(
                x=data[date_column],
                y=data[col],
                name=col,
                mode='lines+markers'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            template=self.theme,
            hovermode='x unified'
        )

        return fig

    def create_dashboard(
        self,
        data: pd.DataFrame,
        charts: List[Dict[str, Any]],
        rows: int = 2,
        cols: int = 2,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create multi-chart dashboard

        Args:
            data: DataFrame
            charts: List of chart specifications
                    Each dict should have: {'type': 'histogram', 'x': 'col1', ...}
            rows: Number of rows in subplot grid
            cols: Number of columns in subplot grid
            title: Dashboard title

        Returns:
            Figure with subplots
        """
        title = title or "Dashboard"

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[chart.get('title', f"Chart {i+1}") for i, chart in enumerate(charts[:rows*cols])]
        )

        for i, chart_spec in enumerate(charts[:rows*cols]):
            row = i // cols + 1
            col = i % cols + 1

            chart_type = chart_spec.get('type', 'histogram')
            x = chart_spec.get('x')
            y = chart_spec.get('y')

            # Create individual chart
            if chart_type == 'histogram' and x:
                trace = go.Histogram(x=data[x], name=x)
            elif chart_type == 'box' and y:
                trace = go.Box(y=data[y], name=y)
            elif chart_type == 'scatter' and x and y:
                trace = go.Scatter(x=data[x], y=data[y], mode='markers', name=f"{y} vs {x}")
            else:
                continue

            fig.add_trace(trace, row=row, col=col)

        fig.update_layout(
            title=title,
            template=self.theme,
            showlegend=False,
            height=300 * rows
        )

        return fig

    def to_json(self, fig: go.Figure) -> str:
        """Convert figure to JSON for API response"""
        return fig.to_json()

    def to_html(self, fig: go.Figure, include_plotlyjs: str = 'cdn') -> str:
        """Convert figure to HTML"""
        return fig.to_html(include_plotlyjs=include_plotlyjs)

    def save(self, fig: go.Figure, filename: str, format: str = 'html'):
        """Save figure to file"""
        if format == 'html':
            fig.write_html(filename)
        elif format == 'png':
            fig.write_image(filename)
        elif format == 'svg':
            fig.write_image(filename)
        elif format == 'pdf':
            fig.write_image(filename)
        elif format == 'json':
            fig.write_json(filename)
        else:
            raise InvalidParameterError(f"Unsupported format: {format}")

        self.logger.info(f"Saved figure to {filename}")
