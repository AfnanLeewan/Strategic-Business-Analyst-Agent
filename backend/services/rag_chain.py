"""
RAG Chain and Strategic Analysis Service.
Implements hybrid RAG with internal retrieval and external search.
Updated to handle optional Tavily API key.
"""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum

from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document

from backend.core.config import settings
from backend.services.ingest import document_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Query intent classification."""
    INTERNAL = "internal"  # Only company documents
    EXTERNAL = "external"  # Only market/news data
    HYBRID = "hybrid"  # Both sources


class RAGEngine:
    """Hybrid RAG engine combining internal and external knowledge."""
    def __init__(self):
        """Initialize the RAG engine."""
        # Initialize LLM based on provider
        if settings.llm_provider == "ollama":
            from langchain_community.llms import Ollama
            self.llm = Ollama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=settings.temperature
            )
            logger.info(f"Using Ollama LLM: {settings.ollama_model}")
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=settings.openai_model,
                temperature=settings.temperature,
                openai_api_key=settings.openai_api_key
            )
            logger.info(f"Using OpenAI LLM: {settings.openai_model}")
        
        # Initialize Tavily search tool (optional - only if API key provided)
        # Use os.getenv as fallback since pydantic-settings might not load .env in subprocess
        import os
        from dotenv import load_dotenv
        load_dotenv()  # Explicitly load .env file
        
        tavily_key = os.getenv('TAVILY_API_KEY') or settings.tavily_api_key
        logger.info(f"Tavily API key present: {bool(tavily_key)}")
        if tavily_key:
            logger.info(f"Tavily API key value: {tavily_key[:10]}...")
        
        if tavily_key:
            try:
                self.search_tool = TavilySearchResults(
                    api_key=tavily_key,
                    max_results=5
                )
                logger.info("Tavily search tool initialized for external market intelligence")
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily search tool: {e}")
                self.search_tool = None
        else:
            self.search_tool = None
            logger.warning("Tavily API key not provided - external market search disabled")
        
        # Strategic analysis prompt
        self.analysis_prompt = self._create_analysis_prompt()
    
    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create the strategic analysis prompt template."""
        system_template = """You are StratAI, a senior business strategist and data analyst.

Your mission is to provide ACCURATE, data-driven strategic insights based ONLY on the information provided.

⚠️ CRITICAL ANTI-HALLUCINATION RULES (MUST FOLLOW):

1. **ONLY USE PROVIDED DATA** 🚨
   - Use ONLY the numbers and facts from the Context Information below
   - NEVER make up statistics, percentages, or comparisons
   - If you don't see a specific number in the context, say "Data not available"
   - DO NOT calculate growth rates unless you have BOTH current AND historical data

2. **NO FABRICATED COMPARISONS** 🚫
   - NEVER compare to previous periods (last year, last quarter) unless that data is explicitly provided
   - NEVER claim "growth of X%" without historical baseline data
   - NEVER say "increased/decreased by Y%" without comparison data
   - Instead say: "Current period shows [actual numbers from data]"

3. **EXACT NUMBER CITATION** 📊
   - Report the EXACT numbers from the context
   - If context shows "Sales: 2,595,000", report "2.595M" or "$2,595,000" - NOT "2.1M"
   - Round only for readability, but stay within 5% of actual values
   - When aggregating, show your calculation

4. **VERIFY BEFORE CLAIMING** ✓
   - Top seller: Check ALL products before claiming which is #1
   - Regional ranking: Compare ALL regions before declaring winners/losers
   - Category totals: Sum the actual numbers provided
   - Trends: Only describe trends if you have time-series data points

5. **STATE LIMITATIONS EXPLICITLY** ⚠️
   - If no historical data: "Note: Growth comparison not available - only current period data provided"
   - If incomplete data: "Based on available data for [time period]"
   - If uncertain: "Data suggests... but further validation recommended"

ANALYSIS FRAMEWORK:

**Structured Response Format**:
```
## Executive Summary
[2-3 sentences with key findings - ONLY from provided data]

## Key Metrics (from Source Data)
- Total Revenue: $X (exact from context)
- By Category: [List with exact numbers]
- By Region: [List with exact numbers]
- Top Products: [List with exact numbers]

## Analysis
[Insights based ONLY on patterns in the provided data]
[NO comparisons to missing historical periods]
[NO fabricated growth percentages]

## Strategic Insights
✅ Strengths: [Based on what data shows]
🎯 Opportunities: [Logical next steps given current state]
⚠️ Risks/Limitations: [Include data gaps and uncertainties]

## Actionable Recommendations
1. [Specific action based on actual data]
2. [Specific action based on actual data]

## Data Limitations
[Explicitly state what data is NOT available that would improve analysis]
```

**Language Guidelines**:
- Use clear, simple Thai or English
- Report exact numbers from source
- Use bullet points and headers
- Use emojis for clarity: 📊 ✅ ⚠️ 🎯 💡

**Source Citations** (MANDATORY):
- Every number MUST cite its source
- Format: "Revenue: $2.6M (Source: company_sales_q1_2024.csv, Electronics category total)"
- External data needs URL

Context Information (USE ONLY THIS DATA):
{context}

Be professional, data-driven, and actionable. Executives should be able to make decisions based on your analysis.
"""
        
        human_template = """User Query: {query}

REMINDER: Use ONLY the data from Context Information above. Do NOT fabricate comparisons or growth rates.

Please provide a strategic analysis following the framework above:
1. Executive Summary (key findings from PROVIDED DATA ONLY)
2. Key Metrics (EXACT numbers from context - cite sources)
3. Detailed Analysis (patterns in the PROVIDED data)
4. Strategic Insights (based on ACTUAL data, not assumptions)
5. Actionable Recommendations (realistic based on current data)
6. Data Limitations (explicitly state what's missing for complete analysis)

⚠️ CRITICAL: If you don't have historical data, DO NOT claim growth/decline percentages.

Your response:"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
    
    def classify_intent(self, query: str) -> QueryIntent:
        """
        Classify the user's query intent using LLM-based reasoning.
        More accurate and context-aware than keyword matching.
        
        Args:
            query: User query
            
        Returns:
            QueryIntent enum value
        """
        try:
            # Create classification prompt
            classification_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a query classifier for a Strategic Business Intelligence system.

Classify queries as INTERNAL, EXTERNAL, or HYBRID:

INTERNAL = Company-specific data (our revenue, our sales, our reports, company performance)
EXTERNAL = Market/industry data (market trends, competitor analysis, industry news)
HYBRID = Comparison queries (our performance vs market, benchmarking, competitive position)

CRITICAL: You MUST respond with EXACTLY ONE WORD only - no explanation, no reasoning, no asterisks.
Your response must be one of: INTERNAL, EXTERNAL, or HYBRID

If the query mentions "our", "we", or "company" → INTERNAL
If the query mentions "market", "competitors", "trends" alone → EXTERNAL
If the query compares internal to external → HYBRID

Output format: Just the word (INTERNAL, EXTERNAL, or HYBRID)"""),
                ("human", "Query: {query}\n\nClassification:")
            ])
            
            # Format and invoke LLM
            messages = classification_prompt.format_messages(query=query)
            response = self.llm.invoke(messages)
            
            # Parse response - handle both string (Ollama) and message object (ChatOpenAI)
            if isinstance(response, str):
                classification = response.strip().upper()
            else:
                classification = response.content.strip().upper()
            
            # Map to QueryIntent enum
            if "HYBRID" in classification:
                intent = QueryIntent.HYBRID
            elif "EXTERNAL" in classification:
                intent = QueryIntent.EXTERNAL
            else:
                # Default to INTERNAL for safety (prefer internal data)
                intent = QueryIntent.INTERNAL
            
            logger.info(f"LLM classified query intent as: {intent.value} (raw: {classification})")
            return intent
            
        except Exception as e:
            # Fallback to keyword-based classification if LLM fails
            logger.warning(f"LLM classification failed, falling back to keyword matching: {e}")
            return self._classify_intent_fallback(query)
    
    def _classify_intent_fallback(self, query: str) -> QueryIntent:
        """
        Fallback keyword-based classification (backup method).
        Used if LLM classification fails.
        
        Args:
            query: User query
            
        Returns:
            QueryIntent enum value
        """
        query_lower = query.lower()
        
        # External indicators (English + Thai)
        external_keywords = [
            "market", "trend", "competitor", "industry", "news",
            "current", "latest", "recent", "today", "external",
            "benchmark", "comparison", "sector",
            "ตลาด", "แนวโน้ม", "คู่แข่ง", "อุตสาหกรรม", "ข่าว",
            "ปัจจุบัน", "ล่าสุด", "เมื่อเร็ว", "วันนี้", "ภายนอก",
            "เทียบ", "เปรียบเทียบ", "ภาคธุรกิจ"
        ]
        
        # Internal indicators (English + Thai)
        internal_keywords = [
            "our", "we", "company", "internal", "report",
            "revenue", "sales", "performance", "data", "analyze",
            "quarter", "annual", "financial",
            "เรา", "บริษัท", "ภายใน", "รายงาน", "วิเคราะห์",
            "รายได้", "ยอดขาย", "ผลประกอบการ", "ข้อมูล",
            "ไตรมาส", "ประจำปี", "การเงิน", "จุดแข็ง", "จุดอ่อน"
        ]
        
        has_external = any(keyword in query_lower for keyword in external_keywords)
        has_internal = any(keyword in query_lower for keyword in internal_keywords)
        
        # Special rule for sales analysis
        if ("ยอดขาย" in query_lower or "sales" in query_lower) and ("วิเคราะห์" in query_lower or "analyze" in query_lower):
            intent = QueryIntent.INTERNAL
        elif has_external and has_internal:
            intent = QueryIntent.HYBRID
        elif has_external:
            intent = QueryIntent.EXTERNAL
        else:
            intent = QueryIntent.INTERNAL
        
        logger.info(f"Fallback classification: {intent.value}")
        return intent
    
    def retrieve_internal_context(self, query: str, k: int = 10) -> List[Document]:
        """
        Retrieve relevant internal documents from vector store.
        
        Args:
            query: User query
            k: Number of documents to retrieve (increased to 10 for better coverage)
            
        Returns:
            List of relevant documents
        """
        try:
            docs = document_processor.search_similar(query, k=k)
            logger.info(f"Retrieved {len(docs)} internal documents")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving internal context: {e}")
            return []
    
    def retrieve_external_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve external market intelligence using Tavily search.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        try:
            if self.search_tool is None:
                logger.warning("External search requested but Tavily API key not configured")
                return []
            
            results = self.search_tool.invoke(query)
            logger.info(f"Retrieved {len(results)} external search results")
            return results
        except Exception as e:
            logger.error(f"Error retrieving external context: {e}")
            return []
    
    def merge_context(
        self,
        internal_docs: List[Document],
        external_results: List[Dict[str, Any]]
    ) -> str:
        """
        Merge internal and external context into a single string.
        
        Args:
            internal_docs: Internal document chunks
            external_results: External search results
            
        Returns:
            Merged context string
        """
        context_parts = []
        
        # Add internal context
        if internal_docs:
            context_parts.append("=== INTERNAL COMPANY DOCUMENTS ===\n")
            for i, doc in enumerate(internal_docs, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                context_parts.append(
                    f"Document {i}: {source} (Page {page})\n"
                    f"{doc.page_content}\n"
                )
        
        # Add external context
        if external_results:
            context_parts.append("\n=== EXTERNAL MARKET INTELLIGENCE ===\n")
            for i, result in enumerate(external_results, 1):
                if isinstance(result, dict):
                    title = result.get("title", result.get("url", "Unknown"))
                    content = result.get("content", result.get("snippet", ""))
                    url = result.get("url", "")
                    
                    context_parts.append(
                        f"Source {i}: {title}\n"
                        f"URL: {url}\n"
                        f"{content}\n"
                    )
                else:
                    context_parts.append(f"Source {i}: {str(result)}\n")
        
        if not context_parts:
            return "No relevant context found. Please upload documents or refine your query."
        
        return "\n".join(context_parts)
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Main analysis pipeline: retrieve context and generate strategic response.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with analysis results and metadata
        """
        try:
            logger.info(f"Analyzing query: {query}")
            
            # Step 1: Classify intent
            intent = self.classify_intent(query)
            
            # Step 2: Retrieve context based on intent
            internal_docs = []
            external_results = []
            
            if intent in [QueryIntent.INTERNAL, QueryIntent.HYBRID]:
                internal_docs = self.retrieve_internal_context(query)
                
                # Check if internal documents are required but empty
                if intent == QueryIntent.INTERNAL and len(internal_docs) == 0:
                    return {
                        "success": False,
                        "error": "no_documents",
                        "message": "ไม่พบเอกสารภายในระบบ กรุณาอัปโหลดเอกสาร (PDF หรือ CSV) ก่อนถามคำถามเกี่ยวกับข้อมูลภายในบริษัท",
                        "query": query,
                        "intent": intent.value
                    }
            
            if intent in [QueryIntent.EXTERNAL, QueryIntent.HYBRID]:
                external_results = self.retrieve_external_context(query)
            
            # Step 3: Merge context
            context = self.merge_context(internal_docs, external_results)
            
            # Step 4: Generate analysis
            messages = self.analysis_prompt.format_messages(
                context=context,
                query=query
            )
            
            response = self.llm.invoke(messages)
            
            # Handle both string response (Ollama) and message object (ChatOpenAI)
            if isinstance(response, str):
                analysis_text = response
            else:
                analysis_text = response.content
            
            # Step 5: Extract source citations
            sources = self._extract_sources(internal_docs, external_results)
            
            return {
                "success": True,
                "query": query,
                "intent": intent.value,
                "analysis": analysis_text,
                "sources": sources,
                "metadata": {
                    "internal_docs_count": len(internal_docs),
                    "external_results_count": len(external_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Analysis failed: {str(e)}"
            }
    
    def _extract_sources(
        self,
        internal_docs: List[Document],
        external_results: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Extract source citations for the references section."""
        sources = []
        
        # Internal sources
        for doc in internal_docs:
            sources.append({
                "type": "internal",
                "name": doc.metadata.get("source", "Unknown"),
                "page": str(doc.metadata.get("page", "N/A")),
                "preview": doc.page_content[:200] + "..."
            })
        
        # External sources
        for result in external_results:
            if isinstance(result, dict):
                sources.append({
                    "type": "external",
                    "name": result.get("title", "Unknown"),
                    "url": result.get("url", ""),
                    "preview": result.get("content", result.get("snippet", ""))[:200] + "..."
                })
        
        return sources


# Global RAG engine instance
rag_engine = RAGEngine()
