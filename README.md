# StratAI - Strategic Business Analyst Agent

A RAG-based AI business intelligence tool that combines internal company knowledge with external market intelligence for strategic analysis.

## 🎯 Overview

StratAI is an AI-powered application that helps businesses analyze their operations by:
- **Internal Knowledge**: Processing company documents (PDFs, CSVs) using RAG
- **External Intelligence**: Fetching real-time market data and news via Tavily Search
- **Strategic Analysis**: Generating actionable insights with verifiable citations
- **Predictive Analytics**: Forecasting trends from historical data (optional)

## 🏗️ Architecture

- **Frontend**: Streamlit (Interactive UI)
- **Backend**: FastAPI (REST API)
- **Orchestration**: LangChain
- **Vector DB**: ChromaDB (Local persistence)
- **LLM**: OpenAI GPT-4o
- **Search**: Tavily API
- **Deployment**: Docker

## 📋 Prerequisites

- Python 3.10+
- OpenAI API Key
- Tavily API Key
- Docker & Docker Compose (for containerized deployment)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd /Users/afnan/Documents/ai-workshop/project
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Run Frontend (in a new terminal)

```bash
cd frontend
streamlit run app.py --server.port 8501
```

### 5. Access the Application

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

## 📁 Project Structure

```
project/
├── backend/
│   ├── core/
│   │   └── config.py          # Configuration management
│   ├── services/
│   │   ├── ingest.py          # Document processing & ETL
│   │   ├── rag_chain.py       # RAG engine & analysis
│   │   └── predictor.py       # Predictive analytics
│   └── main.py                # FastAPI app
├── frontend/
│   └── app.py                 # Streamlit UI
├── uploads/                   # Uploaded files
├── vectordb/                  # ChromaDB storage
├── requirements.txt
├── .env.example
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
└── README.md
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/upload` | POST | Upload & index documents |
| `/analyze` | POST | Analyze query with RAG |

## 📊 Features

### 1. Document Processing
- Upload PDFs (annual reports, strategy docs)
- Upload CSVs (sales data, financial sheets)
- Automatic text extraction and chunking
- Vector embedding generation

### 2. Hybrid RAG
- Intent classification (internal/external/hybrid)
- Internal document retrieval (ChromaDB)
- External market search (Tavily)
- Context merging and analysis

### 3. Strategic Analysis
- Professional business insights
- SWOT analysis
- Source citations
- Actionable recommendations

### 4. Predictive Analytics (Optional)
- Sales forecasting (3-month prediction)
- Linear regression on historical data
- Integration with LLM narratives

## 🎨 UI Features

- **Sidebar**: File upload, status indicators, model settings
- **Chat Interface**: Ask strategic questions, get AI responses
- **Data Visualization**: Charts for sales data and predictions
- **References**: View source citations and document chunks

## 📝 Example Queries

- "What are our company's main strengths based on the annual report?"
- "How do we compare to our competitors in the market?"
- "What are the current trends in AI technology?"
- "Predict our sales for the next quarter"

## 🔒 Environment Variables

See `.env.example` for all required configuration variables.

## 📚 Documentation

For detailed implementation details, see the [Implementation Plan](/.gemini/antigravity/brain/cd54eef4-dc2c-4565-8e2e-2ff30b2338c0/implementation_plan.md).

## 🤝 Contributing

This is a capstone project for an AI Bootcamp.

## 📄 License

MIT License
