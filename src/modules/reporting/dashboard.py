"""
Data Copilot Lab - Business Dashboard Builder
Create interactive business dashboards with key metrics and visualizations
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.exceptions import InvalidParameterError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DashboardLayout(str, Enum):
    """Dashboard layout templates"""
    EXECUTIVE = "executive"  # High-level KPIs
    OPERATIONAL = "operational"  # Detailed metrics
    ANALYTICAL = "analytical"  # Deep-dive analysis
    CUSTOM = "custom"


class MetricType(str, Enum):
    """Metric display types"""
    NUMBER = "number"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    CHANGE = "change"  # Shows increase/decrease
    GAUGE = "gauge"
    PROGRESS = "progress"


class DashboardBuilder:
    """
    Build interactive business dashboards

    Features:
    - KPI cards with metrics
    - Interactive charts
    - Filters and drill-downs
    - Responsive layouts
    - Export to HTML
    """

    def __init__(
        self,
        title: str = "Business Dashboard",
        layout: Union[str, DashboardLayout] = DashboardLayout.EXECUTIVE
    ):
        """
        Initialize Dashboard Builder

        Args:
            title: Dashboard title
            layout: Dashboard layout template
        """
        self.logger = logger
        self.title = title
        self.layout = DashboardLayout(layout) if isinstance(layout, str) else layout

        self.kpis = []
        self.charts = []
        self.filters = []
        self.metadata = {
            "title": title,
            "layout": self.layout.value
        }

    def add_kpi(
        self,
        name: str,
        value: Any,
        metric_type: Union[str, MetricType] = MetricType.NUMBER,
        change: Optional[float] = None,
        target: Optional[float] = None,
        format_string: Optional[str] = None
    ):
        """
        Add KPI metric

        Args:
            name: KPI name
            value: Current value
            metric_type: Type of metric
            change: Change from previous period (%)
            target: Target value
            format_string: Custom format string
        """
        if isinstance(metric_type, str):
            metric_type = MetricType(metric_type)

        kpi = {
            "name": name,
            "value": value,
            "type": metric_type.value,
            "change": change,
            "target": target,
            "format": format_string
        }

        self.kpis.append(kpi)
        self.logger.debug(f"Added KPI: {name} = {value}")

    def add_chart(
        self,
        title: str,
        figure: go.Figure,
        width: str = "full",
        height: int = 400
    ):
        """
        Add chart to dashboard

        Args:
            title: Chart title
            figure: Plotly figure
            width: Chart width ('full', 'half', 'third')
            height: Chart height in pixels
        """
        self.charts.append({
            "title": title,
            "figure": figure,
            "width": width,
            "height": height
        })
        self.logger.debug(f"Added chart: {title}")

    def add_metric_chart(
        self,
        title: str,
        data: pd.DataFrame,
        metric_col: str,
        time_col: Optional[str] = None,
        aggregation: str = "sum",
        chart_type: str = "line"
    ) -> go.Figure:
        """
        Create and add metric chart

        Args:
            title: Chart title
            data: DataFrame
            metric_col: Column with metric values
            time_col: Time column (None = use index)
            aggregation: Aggregation method
            chart_type: 'line', 'bar', or 'area'

        Returns:
            Created figure
        """
        if time_col:
            data_agg = data.groupby(time_col)[metric_col].agg(aggregation).reset_index()
            x = data_agg[time_col]
            y = data_agg[metric_col]
        else:
            x = data.index
            y = data[metric_col]

        if chart_type == "line":
            fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers', name=metric_col))
        elif chart_type == "bar":
            fig = go.Figure(data=go.Bar(x=x, y=y, name=metric_col))
        elif chart_type == "area":
            fig = go.Figure(data=go.Scatter(x=x, y=y, fill='tozeroy', name=metric_col))
        else:
            raise InvalidParameterError(f"Unknown chart type: {chart_type}")

        fig.update_layout(
            title=title,
            xaxis_title=time_col or "Index",
            yaxis_title=metric_col,
            hovermode='x unified'
        )

        self.add_chart(title, fig)

        return fig

    def add_comparison_chart(
        self,
        title: str,
        data: pd.DataFrame,
        category_col: str,
        value_col: str,
        top_n: int = 10
    ) -> go.Figure:
        """
        Create comparison bar chart

        Args:
            title: Chart title
            data: DataFrame
            category_col: Category column
            value_col: Value column
            top_n: Show top N categories

        Returns:
            Created figure
        """
        # Aggregate and get top N
        data_agg = data.groupby(category_col)[value_col].sum().sort_values(ascending=False).head(top_n)

        fig = go.Figure(data=go.Bar(
            x=data_agg.values,
            y=data_agg.index,
            orientation='h'
        ))

        fig.update_layout(
            title=title,
            xaxis_title=value_col,
            yaxis_title=category_col,
            height=max(300, top_n * 30)
        )

        self.add_chart(title, fig, width="half")

        return fig

    def add_distribution_chart(
        self,
        title: str,
        data: pd.DataFrame,
        column: str,
        chart_type: str = "histogram"
    ) -> go.Figure:
        """
        Create distribution chart

        Args:
            title: Chart title
            data: DataFrame
            column: Column to analyze
            chart_type: 'histogram' or 'box'

        Returns:
            Created figure
        """
        if chart_type == "histogram":
            fig = go.Figure(data=go.Histogram(x=data[column], name=column))
            fig.update_layout(
                title=title,
                xaxis_title=column,
                yaxis_title="Count"
            )
        elif chart_type == "box":
            fig = go.Figure(data=go.Box(y=data[column], name=column))
            fig.update_layout(
                title=title,
                yaxis_title=column
            )
        else:
            raise InvalidParameterError(f"Unknown chart type: {chart_type}")

        self.add_chart(title, fig, width="half")

        return fig

    def add_gauge(
        self,
        name: str,
        value: float,
        max_value: float,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Add gauge metric

        Args:
            name: Gauge name
            value: Current value
            max_value: Maximum value
            thresholds: Color thresholds {'red': 0.3, 'yellow': 0.7, 'green': 1.0}
        """
        if thresholds is None:
            thresholds = {'red': 0.3, 'yellow': 0.7, 'green': 1.0}

        # Determine color
        percentage = value / max_value
        if percentage < thresholds.get('red', 0.3):
            color = "red"
        elif percentage < thresholds.get('yellow', 0.7):
            color = "yellow"
        else:
            color = "green"

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={'text': name},
            gauge={
                'axis': {'range': [None, max_value]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, max_value * thresholds.get('red', 0.3)], 'color': "lightgray"},
                    {'range': [max_value * thresholds.get('red', 0.3), max_value * thresholds.get('yellow', 0.7)], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value
                }
            }
        ))

        fig.update_layout(height=250)

        self.add_chart(name, fig, width="third", height=250)

    def generate_html(self, output_path: Optional[str] = None) -> str:
        """
        Generate HTML dashboard

        Args:
            output_path: Output file path (None = return HTML string)

        Returns:
            HTML content
        """
        self.logger.info(f"Generating HTML dashboard with {len(self.kpis)} KPIs and {len(self.charts)} charts")

        html_parts = []

        # HTML header
        html_parts.append("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f7fa;
            padding: 20px;
        }}
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            color: #2c3e50;
            font-size: 2em;
            margin-bottom: 5px;
        }}
        .header .subtitle {{
            color: #7f8c8d;
        }}
        .kpi-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .kpi-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .kpi-label {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .kpi-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .kpi-change {{
            font-size: 0.9em;
            padding: 5px 10px;
            border-radius: 20px;
            display: inline-block;
        }}
        .kpi-change.positive {{
            background: #d4edda;
            color: #155724;
        }}
        .kpi-change.negative {{
            background: #f8d7da;
            color: #721c24;
        }}
        .charts-section {{
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 20px;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .chart-container.full {{
            grid-column: span 12;
        }}
        .chart-container.half {{
            grid-column: span 6;
        }}
        .chart-container.third {{
            grid-column: span 4;
        }}
        @media (max-width: 768px) {{
            .chart-container.half,
            .chart-container.third {{
                grid-column: span 12;
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>{title}</h1>
            <p class="subtitle">Real-time business intelligence dashboard</p>
        </div>
""".format(title=self.title))

        # KPI cards
        if self.kpis:
            html_parts.append('<div class="kpi-section">')

            for kpi in self.kpis:
                value_str = self._format_kpi_value(kpi['value'], kpi['type'], kpi.get('format'))

                change_html = ""
                if kpi.get('change') is not None:
                    change = kpi['change']
                    change_class = "positive" if change >= 0 else "negative"
                    change_html = f'<span class="kpi-change {change_class}">{"↑" if change >= 0 else "↓"} {abs(change):.1f}%</span>'

                html_parts.append(f"""
        <div class="kpi-card">
            <div class="kpi-label">{kpi['name']}</div>
            <div class="kpi-value">{value_str}</div>
            {change_html}
        </div>
""")

            html_parts.append('</div>')

        # Charts
        if self.charts:
            html_parts.append('<div class="charts-section">')

            for i, chart in enumerate(self.charts):
                chart_id = f"chart_{i}"
                width_class = chart['width']

                html_parts.append(f"""
        <div class="chart-container {width_class}">
            <div id="{chart_id}" style="height: {chart['height']}px;"></div>
        </div>
""")

            html_parts.append('</div>')

            # Chart scripts
            html_parts.append('<script>')
            for i, chart in enumerate(self.charts):
                chart_id = f"chart_{i}"
                chart_json = chart['figure'].to_json()
                html_parts.append(f"""
        var chartData_{i} = {chart_json};
        Plotly.newPlot('{chart_id}', chartData_{i}.data, chartData_{i}.layout, {{responsive: true}});
""")
            html_parts.append('</script>')

        # Footer
        html_parts.append("""
    </div>
</body>
</html>
""")

        html_content = "".join(html_parts)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            self.logger.info(f"Dashboard saved to: {output_path}")

        return html_content

    def _format_kpi_value(
        self,
        value: Any,
        metric_type: str,
        format_string: Optional[str]
    ) -> str:
        """Format KPI value based on type"""
        if format_string:
            return format_string.format(value)

        if metric_type == MetricType.NUMBER.value:
            if isinstance(value, float):
                return f"{value:,.2f}"
            return f"{value:,}"

        elif metric_type == MetricType.PERCENTAGE.value:
            return f"{value:.1f}%"

        elif metric_type == MetricType.CURRENCY.value:
            return f"${value:,.0f}"

        else:
            return str(value)

    def get_summary(self) -> Dict[str, Any]:
        """Get dashboard summary"""
        return {
            "title": self.title,
            "layout": self.layout.value,
            "kpis_count": len(self.kpis),
            "charts_count": len(self.charts),
            "kpis": [{"name": k["name"], "value": k["value"]} for k in self.kpis]
        }


class DashboardTemplate:
    """Pre-built dashboard templates"""

    @staticmethod
    def sales_dashboard(sales_data: pd.DataFrame) -> DashboardBuilder:
        """Create sales performance dashboard"""
        dashboard = DashboardBuilder("Sales Performance Dashboard", DashboardLayout.EXECUTIVE)

        # KPIs
        total_revenue = sales_data['revenue'].sum() if 'revenue' in sales_data.columns else 0
        total_orders = len(sales_data)
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

        dashboard.add_kpi("Total Revenue", total_revenue, MetricType.CURRENCY, change=12.5)
        dashboard.add_kpi("Total Orders", total_orders, MetricType.NUMBER, change=8.3)
        dashboard.add_kpi("Avg Order Value", avg_order_value, MetricType.CURRENCY, change=-2.1)

        return dashboard

    @staticmethod
    def marketing_dashboard(campaign_data: pd.DataFrame) -> DashboardBuilder:
        """Create marketing performance dashboard"""
        dashboard = DashboardBuilder("Marketing Performance Dashboard", DashboardLayout.OPERATIONAL)

        # KPIs
        total_impressions = campaign_data['impressions'].sum() if 'impressions' in campaign_data.columns else 0
        total_clicks = campaign_data['clicks'].sum() if 'clicks' in campaign_data.columns else 0
        ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0

        dashboard.add_kpi("Impressions", total_impressions, MetricType.NUMBER)
        dashboard.add_kpi("Clicks", total_clicks, MetricType.NUMBER)
        dashboard.add_kpi("CTR", ctr, MetricType.PERCENTAGE)

        return dashboard

    @staticmethod
    def customer_dashboard(customer_data: pd.DataFrame) -> DashboardBuilder:
        """Create customer analytics dashboard"""
        dashboard = DashboardBuilder("Customer Analytics Dashboard", DashboardLayout.ANALYTICAL)

        # KPIs
        total_customers = len(customer_data)
        active_customers = customer_data['active'].sum() if 'active' in customer_data.columns else 0
        churn_rate = ((total_customers - active_customers) / total_customers * 100) if total_customers > 0 else 0

        dashboard.add_kpi("Total Customers", total_customers, MetricType.NUMBER)
        dashboard.add_kpi("Active Customers", active_customers, MetricType.NUMBER)
        dashboard.add_kpi("Churn Rate", churn_rate, MetricType.PERCENTAGE)

        return dashboard
