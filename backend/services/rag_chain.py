"""
RAG Chain and Strategic Analysis Service.
Implements hybrid RAG with internal retrieval and external search.
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
        
        # Initialize Tavily search tool
        self.search_tool = TavilySearchResults(
            api_key=settings.tavily_api_key,
            max_results=5
        )
        
        # Strategic analysis prompt
        self.analysis_prompt = self._create_analysis_prompt()
    
    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create the strategic analysis prompt template."""
        system_template = """You are StratAI, a senior business strategist and data analyst.

Your mission is to provide clear, data-driven strategic insights that executives can act on immediately.

ANALYSIS FRAMEWORK:

1. **Start with Key Metrics** 📊
   - Always highlight important numbers first
   - Use clear formatting: "Revenue: $X", "Growth: +Y%"
   - Compare current vs historical vs forecast

2. **Structured Response Format**:
   ```
   ## Executive Summary
   [2-3 sentences with key findings]
   
   ## Key Metrics
   - Metric 1: [Number] ([Change])
   - Metric 2: [Number] ([Change])
   
   ## Analysis
   [Detailed insights with evidence]
   
   ## Strategic Insights (SWOT when applicable)
   ✅ Strengths: ...
   🎯 Opportunities: ...
   ⚠️ Risks/Weaknesses: ...
   
   ## Actionable Recommendations
   1. [Specific action] - [Expected outcome]
   2. [Specific action] - [Expected outcome]
   ```

3. **Language Guidelines**:
   - Use clear, simple Thai or English
   - Avoid jargon unless necessary
   - Use bullet points and headers
   - Use emojis for visual clarity: 📊 ✅ ⚠️ 🎯 💡

4. **Source Citations** (CRITICAL):
   - Internal: "Source: [Document/CSV name]"
   - External: "Source: [Outlet]" with URL if available
   - Always cite data sources for numbers

5. **Make it Actionable**:
   - Every insight should have a "So what?" and "What next?"
   - Provide specific, measurable recommendations
   - Include timelines when relevant

Context Information:
{context}

Be professional, data-driven, and actionable. Executives should be able to make decisions based on your analysis.
"""
        
        human_template = """User Query: {query}

Please provide a strategic analysis following the framework above:
1. Executive Summary (key findings)
2. Key Metrics (numbers first!)
3. Detailed Analysis (evidence-based)
4. Strategic Insights (SWOT if applicable)
5. Actionable Recommendations (specific steps)
6. Source Citations (for all claims)

Your response:"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
    
    def classify_intent(self, query: str) -> QueryIntent:
        """
        Classify the user's query intent.
        
        Args:
            query: User query
            
        Returns:
            QueryIntent enum value
        """
        query_lower = query.lower()
        
        # External indicators (English + Thai)
        external_keywords = [
            # English
            "market", "trend", "competitor", "industry", "news",
            "current", "latest", "recent", "today", "external",
            "benchmark", "comparison", "sector",
            # Thai
            "ตลาด", "แนวโน้ม", "คู่แข่ง", "อุตสาหกรรม", "ข่าว",
            "ปัจจุบัน", "ล่าสุด", "เมื่อเร็ว", "วันนี้", "ภายนอก",
            "เทียบ", "เปรียบเทียบ", "ภาคธุรกิจ"
        ]
        
        # Internal indicators (English + Thai)
        internal_keywords = [
            # English
            "our", "we", "company", "internal", "report",
            "revenue", "sales", "performance", "data", "analyze",
            "quarter", "annual", "financial",
            # Thai
            "เรา", "บริษัท", "ภายใน", "รายงาน", "วิเคราะห์",
            "รายได้", "ยอดขาย", "ผลประกอบการ", "ข้อมูล",
            "ไตรมาส", "ประจำปี", "การเงิน", "จุดแข็ง", "จุดอ่อน"
        ]
        
        has_external = any(keyword in query_lower for keyword in external_keywords)
        has_internal = any(keyword in query_lower for keyword in internal_keywords)
        
        # Special logic: if query contains "ยอดขาย" or "sales" + "วิเคราะห์"/"analyze", it's INTERNAL
        if ("ยอดขาย" in query_lower or "sales" in query_lower) and ("วิเคราะห์" in query_lower or "analyze" in query_lower):
            intent = QueryIntent.INTERNAL
        elif has_external and has_internal:
            intent = QueryIntent.HYBRID
        elif has_external:
            intent = QueryIntent.EXTERNAL
        else:
            intent = QueryIntent.INTERNAL
        
        logger.info(f"Query intent classified as: {intent.value}")
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
