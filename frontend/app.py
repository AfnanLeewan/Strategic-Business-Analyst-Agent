"""
Streamlit Frontend for StratAI.
Interactive UI for document upload and strategic analysis.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="StratAI - Strategic Business Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .source-card {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None


def check_backend_health():
    """Check if backend is running."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def upload_file(file):
    """Upload file to backend."""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{BACKEND_URL}/upload", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "message": f"Error: {response.status_code}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


def analyze_query(query):
    """Send query to backend for analysis."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/analyze",
            json={"query": query}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "message": f"Error: {response.status_code}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


# Main App
st.markdown('<div class="main-header">📊 StratAI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Strategic Business Analyst Agent</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Backend status
    backend_status = check_backend_health()
    if backend_status:
        st.success("✅ Backend Connected")
    else:
        st.error("❌ Backend Disconnected")
        st.info(f"Trying to connect to: {BACKEND_URL}")
    
    st.markdown("---")
    
    # File Upload Section
    st.header("📁 Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Upload PDF or CSV",
        type=['pdf', 'csv'],
        help="Upload company documents (PDFs) or sales data (CSVs)"
    )
    
    if uploaded_file:
        if st.button("📤 Process File", type="primary"):
            with st.spinner("Processing file..."):
                result = upload_file(uploaded_file)
                
                if result.get("success"):
                    st.success(f"✅ {result['message']}")
                    st.info(f"📄 File: {result['file_name']}")
                    st.info(f"📊 Chunks Created: {result['chunks_created']}")
                    
                    # Store uploaded file info
                    st.session_state.uploaded_files.append({
                        "name": result['file_name'],
                        "type": result['file_type'],
                        "chunks": result['chunks_created'],
                        "timestamp": datetime.now()
                    })
                    
                    # Store forecast if available
                    if result.get("forecast") and result["forecast"].get("success"):
                        st.session_state.forecast_data = result["forecast"]
                        st.success("📈 Sales forecast generated!")
                else:
                    st.error(f"❌ {result.get('message', 'Upload failed')}")
    
    st.markdown("---")
    
    # Model Settings
    st.header("🎛️ Settings")
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls randomness in responses"
    )
    
    st.markdown("---")
    
    # Uploaded Files Summary
    if st.session_state.uploaded_files:
        st.header("📚 Indexed Documents")
        for idx, file_info in enumerate(st.session_state.uploaded_files, 1):
            st.text(f"{idx}. {file_info['name']} ({file_info['type']})")

# Main Content Area
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📈 Data Visualization", "📖 References"])

# Chat Tab
with tab1:
    st.header("Ask Strategic Questions")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**StratAI:** {message['content']}")
                if 'metadata' in message:
                    with st.expander("📊 Analysis Details"):
                        st.json(message['metadata'])
            st.markdown("---")
    
    # Query input
    query = st.text_input(
        "Your Question:",
        placeholder="e.g., What are our company's main strengths?",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 5])
    
    with col1:
        submit_button = st.button("🚀 Analyze", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Clear Chat")
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if submit_button and query:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })
        
        with st.spinner("🤔 Analyzing..."):
            result = analyze_query(query)
            
            if result.get("success"):
                # Add assistant message to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result['analysis'],
                    "metadata": {
                        "intent": result['intent'],
                        "internal_docs": result['metadata']['internal_docs_count'],
                        "external_results": result['metadata']['external_results_count']
                    },
                    "sources": result['sources']
                })
            else:
                st.error(f"❌ {result.get('message', 'Analysis failed')}")
        
        st.rerun()

# Data Visualization Tab
with tab2:
    st.header("Sales Data & Forecasting")
    
    if st.session_state.forecast_data:
        forecast = st.session_state.forecast_data
        
        # Display forecast summary
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📊 Forecast Summary")
        st.markdown(forecast['summary'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model performance
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Model R² Score",
                f"{forecast['model_score']:.4f}"
            )
        
        with col2:
            st.metric(
                "Historical Avg Sales",
                f"${forecast['historical_stats']['mean']:,.2f}"
            )
        
        with col3:
            predicted_mean = sum(p['predicted_sales'] for p in forecast['predictions']) / len(forecast['predictions'])
            st.metric(
                "Predicted Avg Sales",
                f"${predicted_mean:,.2f}"
            )
        
        # Forecast table
        st.markdown("### 📅 Detailed Forecast")
        
        forecast_df = pd.DataFrame(forecast['predictions'])
        forecast_df['predicted_sales'] = forecast_df['predicted_sales'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(forecast_df, use_container_width=True)
        
        # Visualization
        st.markdown("### 📈 Sales Trend Forecast")
        
        # Create plot data
        plot_data = pd.DataFrame(forecast['predictions'])
        plot_data['predicted_sales'] = plot_data['predicted_sales'].astype(float)
        
        fig = px.line(
            plot_data,
            x='date',
            y='predicted_sales',
            title='3-Month Sales Forecast',
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Predicted Sales ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("📊 Upload a CSV file with Date and Sales columns to see forecasts and visualizations.")
        
        # Example data format
        with st.expander("ℹ️ CSV Format Example"):
            example_df = pd.DataFrame({
                'Date': ['2024-01-01', '2024-02-01', '2024-03-01'],
                'Sales': [10000, 12000, 11500]
            })
            st.dataframe(example_df)

# References Tab
with tab3:
    st.header("Source Citations & References")
    
    # Get sources from latest chat message
    all_sources = []
    for message in st.session_state.chat_history:
        if message["role"] == "assistant" and "sources" in message:
            all_sources.extend(message["sources"])
    
    if all_sources:
        st.info(f"📚 Total sources referenced: {len(all_sources)}")
        
        # Group by type
        internal_sources = [s for s in all_sources if s.get("type") == "internal"]
        external_sources = [s for s in all_sources if s.get("type") == "external"]
        
        # Internal sources
        if internal_sources:
            st.markdown("### 📄 Internal Documents")
            for idx, source in enumerate(internal_sources, 1):
                st.markdown(f"""
                <div class="source-card">
                    <strong>{idx}. {source['name']}</strong> (Page {source['page']})<br>
                    <small>{source['preview']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # External sources
        if external_sources:
            st.markdown("### 🌐 External Sources")
            for idx, source in enumerate(external_sources, 1):
                st.markdown(f"""
                <div class="source-card">
                    <strong>{idx}. {source['name']}</strong><br>
                    <a href="{source['url']}" target="_blank">{source['url']}</a><br>
                    <small>{source['preview']}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("💬 Start asking questions to see source citations here.")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.9rem;">'
    'StratAI v1.0 | Powered by OpenAI GPT-4o & Tavily Search'
    '</div>',
    unsafe_allow_html=True
)
