# StratAI - Quick Start Guide

## 🚀 Getting Started

### Step 1: Set Up API Keys

Edit the `.env` file and add your API keys:

```bash
OPENAI_API_KEY=your_actual_openai_api_key
TAVILY_API_KEY=your_actual_tavily_api_key
```

### Step 2: Choose Your Deployment Method

#### Option A: Local Development (Recommended for Testing)

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Backend (Terminal 1):**
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```

3. **Start Frontend (Terminal 2):**
   ```bash
   cd frontend
   streamlit run app.py --server.port 8501
   ```

4. **Access the App:**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

#### Option B: Docker Deployment

1. **Build and Run:**
   ```bash
   docker-compose up --build
   ```

2. **Access the App:**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000

3. **Stop Services:**
   ```bash
   docker-compose down
   ```

## 📋 Usage Guide

### Upload Documents

1. Click the sidebar "Upload Documents" section
2. Choose a PDF (annual reports, strategy docs) or CSV (sales data)
3. Click "Process File"
4. Wait for indexing confirmation

### Ask Questions

1. Navigate to the "Chat" tab
2. Type your strategic question, examples:
   - "What are our company's main strengths?"
   - "How do we compare to competitors?"
   - "What are current AI industry trends?"
   - "Predict our sales for next quarter"
3. Click "Analyze"
4. Review the response with citations

### View Forecasts (CSV only)

1. Go to "Data Visualization" tab
2. See 3-month sales forecast
3. Review model performance metrics
4. Explore trend charts

### Check Sources

1. Navigate to "References" tab
2. See all cited sources
3. Internal documents show page numbers
4. External sources include URLs

## 🧪 Testing with Sample Data

Create a simple test CSV:

```csv
Date,Sales
2024-01-01,10000
2024-02-01,12000
2024-03-01,11500
2024-04-01,13000
2024-05-01,14500
2024-06-01,13800
```

Save as `sales_data.csv` and upload to test forecasting.

## 🔧 Troubleshooting

### Backend Connection Issues
- Ensure backend is running on port 8000
- Check `.env` file has valid API keys
- Verify ChromaDB directory is writable

### Frontend Not Loading
- Clear browser cache
- Restart Streamlit with `--server.port 8501`
- Check if port 8501 is available

### Docker Issues
- Rebuild with: `docker-compose up --build --force-recreate`
- Check logs: `docker-compose logs -f`
- Verify `.env` file exists in project root

## 📚 API Documentation

Once the backend is running, visit:
- Interactive API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🎯 Architecture Overview

```
┌─────────────┐
│  Streamlit  │ (Port 8501)
│  Frontend   │
└──────┬──────┘
       │ REST API calls
       ↓
┌─────────────┐     ┌──────────────┐
│   FastAPI   │────→│   ChromaDB   │
│   Backend   │     │  Vector Store│
└──────┬──────┘     └──────────────┘
       │
       ├────→ OpenAI (GPT-4o, Embeddings)
       │
       └────→ Tavily (External Search)
```

## 📝 Project Structure

```
project/
├── backend/
│   ├── core/
│   │   └── config.py         # Settings
│   ├── services/
│   │   ├── ingest.py         # Document processing
│   │   ├── rag_chain.py      # RAG engine
│   │   └── predictor.py      # Forecasting
│   └── main.py               # FastAPI app
├── frontend/
│   └── app.py                # Streamlit UI
├── uploads/                  # Uploaded files
├── vectordb/                 # ChromaDB data
├── requirements.txt
├── .env
├── Dockerfile.backend
├── Dockerfile.frontend
└── docker-compose.yml
```

## 🤝 Support

For issues or questions:
1. Check the main README.md
2. Review API documentation at /docs
3. Check Docker logs if using containers

---

**Ready to analyze your business? Start uploading documents! 📊**
