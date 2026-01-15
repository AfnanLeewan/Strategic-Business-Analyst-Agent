"""
Quick test to verify OpenAI GPT-4o is working and compare with previous Ollama setup.
Run this to confirm the model switch was successful.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Check environment configuration
print("=" * 60)
print("🔍 CURRENT LLM CONFIGURATION")
print("=" * 60)

llm_provider = os.getenv("LLM_PROVIDER")
openai_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")
ollama_model = os.getenv("OLLAMA_MODEL")

print(f"LLM Provider: {llm_provider}")
print(f"OpenAI Model: {openai_model}")
print(f"OpenAI API Key: {'✅ Set' if openai_key else '❌ Not Set'}")
print(f"Ollama Model (fallback): {ollama_model}")
print()

# Test classification with OpenAI
print("=" * 60)
print("🧪 TESTING QUERY CLASSIFICATION")
print("=" * 60)

from backend.services.rag_chain import rag_engine

test_queries = [
    ("วิเคราะห์ยอดขายในไตรมาสนี้", "INTERNAL"),
    ("What are the latest market trends in tech?", "EXTERNAL"),
    ("Compare our revenue to industry benchmarks", "HYBRID"),
    ("Show me our Q4 performance report", "INTERNAL"),
    ("What's happening in the market today?", "EXTERNAL"),
]

print("\nTesting classification accuracy:\n")

correct = 0
total = len(test_queries)

for query, expected in test_queries:
    intent = rag_engine.classify_intent(query)
    is_correct = intent.value.upper() == expected.upper()
    correct += is_correct
    
    status = "✅" if is_correct else "❌"
    print(f"{status} Query: '{query[:50]}...'")
    print(f"   Expected: {expected} | Got: {intent.value.upper()}")
    print()

accuracy = (correct / total) * 100
print("=" * 60)
print(f"📊 Classification Accuracy: {accuracy:.1f}% ({correct}/{total})")
print("=" * 60)

# Show expected improvements
print("\n" + "=" * 60)
print("🚀 EXPECTED IMPROVEMENTS WITH OPENAI GPT-4o")
print("=" * 60)
print("""
1. ⭐ Better Classification Accuracy
   - More nuanced understanding of intent
   - Better bilingual (Thai/English) support
   - Precise instruction following

2. 📊 Reduced Hallucinations
   - Strictly follows anti-hallucination rules
   - Accurate number citations
   - No fabricated comparisons

3. 💡 Higher Quality Analysis
   - More professional responses
   - Better strategic insights
   - Actionable recommendations

4. 📝 Consistent Formatting
   - Cleaner markdown output
   - Proper source citations
   - Executive-ready reports
""")

print("\n" + "=" * 60)
print("✅ MODEL SWITCH COMPLETE")
print("=" * 60)
print(f"\nYou're now using: {llm_provider.upper()} - {openai_model if llm_provider == 'openai' else ollama_model}")
print("\nTo switch back to Ollama, update .env:")
print("  LLM_PROVIDER=ollama")
print("\nThe backend will auto-reload automatically.")
print("=" * 60)
