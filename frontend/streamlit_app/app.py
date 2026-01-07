"""
Data Copilot Lab - Streamlit Frontend
Main entry point for the Streamlit web interface
"""

import streamlit as st
import requests
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Data Copilot Lab",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = "http://localhost:8000"


def check_api_health():
    """Check if backend API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def main():
    """Main application"""

    # Sidebar
    with st.sidebar:
        st.title("🚀 Data Copilot Lab")
        st.markdown("---")

        # API Status
        if check_api_health():
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.warning("Please start the backend API server")

        st.markdown("---")

        # Navigation
        st.subheader("Navigation")
        page = st.radio(
            "Go to",
            [
                "🏠 Home",
                "📊 Import Data",
                "🧹 Clean Data",
                "🔍 Explore Data (EDA)",
                "🤖 ML Modeling",
                "📈 Reports",
                "💬 AI Assistant"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Info
        st.caption(f"Version: 0.1.0-alpha")
        st.caption("Environment: Development")

    # Main content
    if "🏠 Home" in page:
        show_home_page()
    elif "📊 Import Data" in page:
        show_import_page()
    elif "🧹 Clean Data" in page:
        show_cleaning_page()
    elif "🔍 Explore Data" in page:
        show_eda_page()
    elif "🤖 ML Modeling" in page:
        show_ml_page()
    elif "📈 Reports" in page:
        show_reports_page()
    elif "💬 AI Assistant" in page:
        show_ai_assistant_page()


def show_home_page():
    """Home page"""
    st.title("🚀 Welcome to Data Copilot Lab")

    st.markdown("""
    ## Your AI-Powered Data Science Platform

    Data Copilot Lab is a comprehensive platform that supports the entire data science workflow:
    from data import and cleaning to model training and business reporting.

    ### 🎯 Key Features

    - **📊 Data Import** - Import from CSV, Excel, JSON, SQL databases
    - **🧹 Data Cleaning** - Handle missing values, outliers, duplicates
    - **🔍 EDA** - Interactive exploratory data analysis and visualization
    - **🤖 ML Modeling** - Train, evaluate, and deploy ML models with AutoML
    - **💬 AI Assistant** - Get help from AI copilot at every step
    - **📈 Reporting** - Generate automated reports and dashboards

    ### 🚀 Quick Start

    1. **Import your data** - Upload CSV, Excel, or connect to a database
    2. **Clean and prepare** - Use automated tools to clean your data
    3. **Explore** - Visualize patterns and relationships
    4. **Model** - Train ML models with just a few clicks
    5. **Report** - Generate insights and share with stakeholders

    ### 📚 Documentation

    - [Implementation Plan](https://github.com/YOUR_USERNAME/Data_Copilot_Lab/blob/main/IMPLEMENTATION_PLAN.md)
    - [API Documentation](http://localhost:8000/docs)
    - [User Guide](#)

    ---

    ### 🎬 Get Started

    Choose a section from the sidebar to begin your data science journey!
    """)

    # Quick stats (placeholder)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Datasets", "0", help="Number of imported datasets")

    with col2:
        st.metric("Models", "0", help="Number of trained models")

    with col3:
        st.metric("Reports", "0", help="Number of generated reports")

    with col4:
        st.metric("API Status", "✅", help="Backend API status")


def show_import_page():
    """Data import page (placeholder)"""
    st.title("📊 Import Data")
    st.info("🚧 This feature is under development")

    st.markdown("""
    ### Supported Formats

    - CSV/TSV files
    - Excel (.xls, .xlsx)
    - JSON files
    - SQL databases (PostgreSQL, MySQL, SQLite)
    - REST APIs (coming soon)
    """)

    # File uploader placeholder
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "json"],
        help="Upload your data file"
    )

    if uploaded_file:
        st.success(f"File uploaded: {uploaded_file.name}")
        st.info("Data preview will be available here")


def show_cleaning_page():
    """Data cleaning page (placeholder)"""
    st.title("🧹 Clean Data")
    st.info("🚧 This feature is under development")

    st.markdown("""
    ### Data Quality Tools

    - Missing value detection and handling
    - Outlier detection (IQR, Z-score, Isolation Forest)
    - Duplicate removal
    - Data type standardization
    - Custom transformation pipelines
    """)


def show_eda_page():
    """EDA page (placeholder)"""
    st.title("🔍 Exploratory Data Analysis")
    st.info("🚧 This feature is under development")

    st.markdown("""
    ### Analysis Tools

    - Descriptive statistics
    - Distribution analysis
    - Correlation analysis
    - Interactive visualizations
    - Automated insights
    """)


def show_ml_page():
    """ML modeling page (placeholder)"""
    st.title("🤖 Machine Learning")
    st.info("🚧 This feature is under development")

    st.markdown("""
    ### ML Capabilities

    - Classification models
    - Regression models
    - Clustering
    - AutoML (automatic model selection)
    - Hyperparameter tuning
    - Model evaluation and explainability
    """)


def show_reports_page():
    """Reports page (placeholder)"""
    st.title("📈 Reports & Dashboards")
    st.info("🚧 This feature is under development")

    st.markdown("""
    ### Reporting Features

    - Automated report generation (PDF, HTML)
    - Interactive dashboards
    - Data storytelling tools
    - Business insights
    - Export and sharing
    """)


def show_ai_assistant_page():
    """AI assistant page (placeholder)"""
    st.title("💬 AI Assistant")
    st.info("🚧 This feature is under development")

    st.markdown("""
    ### AI Copilot Features

    - Natural language queries
    - Code generation
    - Smart suggestions
    - Analysis interpretation
    - Best practices guidance
    """)

    # Chat interface placeholder
    st.text_area(
        "Ask me anything about your data...",
        placeholder="Example: What's the correlation between age and income?",
        height=100
    )

    if st.button("Send", type="primary"):
        st.info("AI Assistant will respond here")


if __name__ == "__main__":
    main()
