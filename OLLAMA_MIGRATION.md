# Ollama Migration Guide

## ✅ Migration Complete!

Your StratAI project has been successfully migrated from OpenAI to **Ollama** for local LLM inference.

## 🔧 What Changed

### Configuration Files
- **config.py**: Added support for dual providers (OpenAI/Ollama)
- **.env**: Configured for Ollama by default (llama3.1:8b, nomic-embed-text)
- **requirements.txt**: Added langchain-ollama dependency

### Code Modules
- **ingest.py**: Embeddings now support both OpenAI and Ollama
- **rag_chain.py**: LLM initialization supports both providers

### Provider Settings
```bash
LLM_PROVIDER=ollama           # or "openai"
EMBEDDING_PROVIDER=ollama     # or "openai"
```

## 📋 Prerequisites

### 1. Ollama Models Required

**LLM Model (already installed ✓):**
- llama3.1:8b (4.9 GB)
- llama3.2:3b (2.0 GB) - alternative

**Embedding Model (installing now):**
```bash
ollama pull nomic-embed-text
```

## 🚀 Quick Start

### Option 1: Use Ollama (Default)

Your `.env` file is already configured! Just add your Tavily API key:

```bash
# .env file
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
TAVILY_API_KEY=your_tavily_key_here  # ⚠️ Still required for external search
```

### Option 2: Switch Back to OpenAI

Edit `.env` file:
```bash
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

## 📦 Installation Steps

1. **Update Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Ollama is Running:**
   ```bash
   ollama serve
   ```
   (Usually runs automatically on Mac)

3. **Verify Models:**
   ```bash
   ollama list
   ```
   Should show:
   - llama3.1:8b ✓
   - nomic-embed-text (after pulling)

4. **Add Tavily API Key:**
   Edit `.env` and add your Tavily API key (still needed for external market search)

## 🎯 Running the Application

### Local Development

**Terminal 1 - Backend:**
```bash
cd /Users/afnan/Documents/ai-workshop/project
python -m uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd /Users/afnan/Documents/ai-workshop/project
streamlit run frontend/app.py --server.port 8501
```

Access at: http://localhost:8501

## 🔍 Testing

1. Access the UI
2. Upload a test document (PDF or CSV)
3. Ask a question
4. You should see in backend logs:
   ```
   INFO - Using Ollama LLM: llama3.1:8b
   INFO - Using Ollama embeddings: nomic-embed-text
   ```

## 🎨 Model Options

### LLM Models (Change in .env)

**Recommended:**
- `llama3.1:8b` (Best balance, 4.9GB)
- `llama3.2:3b` (Faster, smaller, 2.0GB)

**Other Options:**
```bash
ollama pull llama3:70b      # More powerful (40GB)
ollama pull mistral         # Alternative model
ollama pull codellama       # For code analysis
```

### Embedding Models

**Recommended:**
- `nomic-embed-text` (Best for RAG, 274MB)

**Alternatives:**
```bash
ollama pull all-minilm      # Smaller, faster
```

## 🔄 Switching Providers

### Use Ollama (Local, Free)
```bash
# .env
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
```

### Use OpenAI (Cloud, Paid)
```bash
# .env
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

### Mix & Match
```bash
# .env
LLM_PROVIDER=ollama           # Local LLM
EMBEDDING_PROVIDER=openai     # OpenAI embeddings
```

## 📊 Performance Comparison

| Aspect | Ollama (Local) | OpenAI (Cloud) |
|--------|---------------|----------------|
| **Cost** | Free | Pay-per-use |
| **Privacy** | 100% local | Cloud-based |
| **Speed** | Depends on hardware | Fast, consistent |
| **Quality** | Good (llama3.1) | Excellent (GPT-4) |
| **Setup** | Models ~5GB | API key only |

## 🐛 Troubleshooting

### "Ollama not found"
```bash
# Install Ollama
brew install ollama
ollama serve
```

### "Model not found"
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### "Connection refused"
Check Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

### Slow responses
- Try smaller model: `llama3.2:3b`
- Check CPU/RAM usage
- Consider using OpenAI for faster results

## 🎓 Benefits of Local LLM

✅ **No API costs** - Unlimited usage  
✅ **Privacy** - Data stays local  
✅ **Offline capable** - No internet needed (except Tavily search)  
✅ **Great for learning** - Perfect for bootcamp projects  

## ⚠️ Note

Tavily API is still required for external market intelligence searches. This is intentional as it provides real-time web data that local models don't have access to.

---

**You're all set! 🎉**

The application now uses Ollama for all LLM inference and embeddings while maintaining the same functionality. Just add your Tavily API key and you're ready to go!
