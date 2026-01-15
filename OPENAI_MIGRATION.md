# 🚀 OpenAI Migration Guide

## ✅ Successfully Switched to OpenAI GPT-4o

Your Strategic Business Analyst Agent is now using **OpenAI GPT-4o** instead of Ollama!

---

## 📊 What Changed?

### Previous Configuration:
- **LLM Provider**: Ollama (Llama 3.1:8b)
- **Location**: Local machine
- **Cost**: Free
- **Speed**: Depends on hardware

### Current Configuration:
- **LLM Provider**: OpenAI (GPT-4o)
- **Location**: Cloud API
- **Cost**: ~$0.005/1K tokens (input), ~$0.015/1K tokens (output)
- **Speed**: 1-3 seconds per response

---

## 🎯 Expected Improvements

### 1. **Better Query Classification** ⭐⭐⭐⭐⭐
- More accurate INTERNAL vs EXTERNAL vs HYBRID detection
- Better understanding of bilingual queries (Thai/English)
- Follows instructions precisely (e.g., returns ONLY "INTERNAL" without explanations)

### 2. **Reduced Hallucinations** 📊
- Better adherence to anti-hallucination rules
- More accurate number citations from source documents
- Less likely to fabricate statistics or comparisons

### 3. **Improved Analysis Quality** 💡
- More structured and professional responses
- Better reasoning about strategic insights
- More actionable recommendations

### 4. **Consistent Formatting** 📝
- Cleaner markdown output
- Better use of emojis and structure
- More executive-friendly reports

---

## 🔄 How to Switch Between Models

Your system is now **fully switchable**! Just update the `.env` file:

### To Use OpenAI (Current):
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o
```

### To Switch Back to Ollama:
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

### Important Notes:
- **Embeddings**: Still using Ollama (`nomic-embed-text`) - free and effective
- **Auto-reload**: Backend automatically reloads when `.env` changes
- **No code changes needed**: The switch is configuration-only

---

## 💰 Cost Estimation

### GPT-4o Pricing:
- **Input**: $0.005 per 1K tokens (~750 words)
- **Output**: $0.015 per 1K tokens

### Typical Query Cost:
- **Simple query** (1 document): ~$0.01
- **Complex analysis** (10 documents): ~$0.05
- **Market research** (hybrid query): ~$0.08

### Monthly Estimate:
- **100 queries/month**: ~$3-5
- **500 queries/month**: ~$15-25
- **1000 queries/month**: ~$30-50

---

## 🧪 Testing the Switch

### Test 1: Classification Accuracy
Run this query to test intent classification:
```
"วิเคราะห์ยอดขายในไตรมาสนี้"
(Expected: INTERNAL)
```

### Test 2: Anti-Hallucination
Upload a sales CSV and ask:
```
"What's our total revenue in Q1 2024?"
```
**Expected behavior**: Should cite EXACT numbers from the CSV

### Test 3: Hybrid Query
```
"Compare our sales performance to market trends"
```
**Expected**: Should retrieve BOTH internal docs AND external market data

---

## 🔍 Monitoring LLM Performance

Check the backend logs to verify OpenAI is being used:

```bash
# In the uvicorn terminal, you should see:
INFO - Using OpenAI LLM: gpt-4o
INFO - LLM classified query intent as: INTERNAL (raw: INTERNAL)
```

---

## 🛠️ Troubleshooting

### Issue: "OpenAI API key not found"
**Solution**: Verify `.env` has `OPENAI_API_KEY` set

### Issue: "Rate limit exceeded"
**Solution**: 
1. Upgrade OpenAI plan
2. Switch back to Ollama temporarily
3. Implement request throttling

### Issue: "Slower than expected"
**Check**:
- Internet connection
- OpenAI API status (status.openai.com)
- Consider switching to `gpt-4o-mini` for faster responses

---

## 🎨 Optional: Add Model Selector to Frontend

Want users to choose the model dynamically? Add this to `frontend/app.py`:

```python
# In the sidebar
model_choice = st.sidebar.radio(
    "Select LLM Provider",
    ["OpenAI (GPT-4o)", "Ollama (Llama 3.1)"],
    index=0
)

# Pass to backend as a query parameter
# (Requires backend modification to support runtime switching)
```

---

## 📈 Recommended Settings

### For Production (Best Quality):
```env
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o
TEMPERATURE=0.7
```

### For Development (Cost Savings):
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
TEMPERATURE=0.7
```

### For Speed (Faster Responses):
```env
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini  # Faster & cheaper
TEMPERATURE=0.5
```

---

## ✅ Next Steps

1. **Test the classification**: Try various queries to see improved accuracy
2. **Monitor costs**: Check OpenAI dashboard for usage
3. **Optimize prompts**: Fine-tune the anti-hallucination rules if needed
4. **Benchmark**: Compare responses between Ollama and OpenAI using the same query

---

## 🔐 Security Reminder

**⚠️ Important**: You've commented out `.env` from `.gitignore`. 

**Before pushing to GitHub:**
```bash
# Option 1: Re-enable .gitignore protection
# Uncomment .env in .gitignore

# Option 2: Use environment secrets
# Move API keys to GitHub Secrets or environment variables

# Option 3: Use .env.example
# Create .env.example with placeholder values
# Keep .env in .gitignore
```

---

## 📞 Support

If you encounter issues:
1. Check backend logs for error messages
2. Verify OpenAI API key is valid
3. Test with a simple query first
4. Compare with Ollama results

---

**Status**: ✅ **ACTIVE - Using OpenAI GPT-4o**

Last updated: 2026-01-15
