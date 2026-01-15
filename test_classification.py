"""
Test script for LLM-based query classification.
Tests both English and Thai queries across different intents.
"""

import sys
import logging
from backend.services.rag_chain import rag_engine

# Configure logging to see classification details
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_classification():
    """Test various query classifications."""
    
    test_cases = [
        # INTERNAL queries
        ("วิเคราะห์ยอดขายของเราในไตรมาสที่ 1", "INTERNAL"),
        ("What's our Q1 revenue?", "INTERNAL"),
        ("Analyze our sales performance", "INTERNAL"),
        ("Show me our company financial reports", "INTERNAL"),
        ("วิเคราะห์ผลประกอบการของบริษัทเรา", "INTERNAL"),
        
        # EXTERNAL queries
        ("What are the latest market trends?", "EXTERNAL"),
        ("แนวโน้มตลาดในปี 2024", "EXTERNAL"),
        ("Tell me about industry news", "EXTERNAL"),
        ("What are competitors doing?", "EXTERNAL"),
        ("ข่าวอุตสาหกรรมล่าสุด", "EXTERNAL"),
        
        # HYBRID queries
        ("Compare our sales to market benchmarks", "HYBRID"),
        ("How do we compare to competitors?", "HYBRID"),
        ("เปรียบเทียบยอดขายของเราและคู่แข่ง", "HYBRID"),
        ("Our market position vs industry", "HYBRID"),
        ("ตำแหน่งของเราในตลาดเมื่อเทียบกับคู่แข่ง", "HYBRID"),
        
        # Edge cases (LLM should handle better than keywords)
        ("How do we stack up against the competition?", "HYBRID"),
        ("What do analysts say about our performance?", "HYBRID"),
        ("Is our growth rate above industry average?", "HYBRID"),
    ]
    
    print("=" * 80)
    print("🧪 Testing LLM-Based Query Classification")
    print("=" * 80)
    
    correct = 0
    total = len(test_cases)
    
    for query, expected in test_cases:
        try:
            intent = rag_engine.classify_intent(query)
            actual = intent.value.upper()
            is_correct = actual == expected
            
            if is_correct:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"\n{status} Query: {query}")
            print(f"   Expected: {expected} | Actual: {actual}")
            
        except Exception as e:
            print(f"\n❌ Query: {query}")
            print(f"   ERROR: {e}")
    
    print("\n" + "=" * 80)
    print(f"📊 Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print("=" * 80)
    
    if correct == total:
        print("🎉 All tests passed!")
    elif correct >= total * 0.9:
        print("✅ Good performance (>90%)")
    elif correct >= total * 0.8:
        print("⚠️  Acceptable performance (>80%)")
    else:
        print("❌ Poor performance (<80%) - needs investigation")

if __name__ == "__main__":
    try:
        test_classification()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
