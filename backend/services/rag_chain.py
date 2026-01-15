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
        elif settings.llm_provider == "gemini":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                import os
                # Ensure API key is set in environment for the library
                if settings.gemini_api_key:
                    os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key
                
                self.llm = ChatGoogleGenerativeAI(
                    model=settings.gemini_model,
                    temperature=settings.temperature,
                    google_api_key=settings.gemini_api_key,
                    convert_system_message_to_human=True
                )
                logger.info(f"Using Google Gemini LLM: {settings.gemini_model}")
            except ImportError as e:
                logger.error(f"Failed to import Gemini dependencies: {e}")
                logger.warning("Falling back to Ollama due to missing Gemini libraries")
                from langchain_community.llms import Ollama
                self.llm = Ollama(
                    model=settings.ollama_model,
                    base_url=settings.ollama_base_url,
                    temperature=settings.temperature
                )
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

Your mission is to provide helpful, data-driven strategic insights. You should COMBINE the provided data with your general business knowledge to answer the user's question, while ensuring any specific numbers cited come directly from the source.

⚠️ DATA ACCURACY RULES:

1. **DATA FIDELITY**:
   - For specific metrics (revenue, sales, counts), USE ONLY the provided Context Information.
   - Do not invent numbers.
   - If a specific number is missing, you can estimate based on available data IF you explicitly state it's an estimate, or say "Data not available".

2. **SAFE COMPARISONS**:
   - Avoid claiming specific growth % (e.g., "+15%") if you don't have historical data.
   - However, you CAN qualitatively discuss performance (e.g., "Strong performance in X region") based on the current data relative to other regions/products.

3. **USE GENERAL KNOWLEDGE**:
   - You ARE allowed to use your general knowledge to explain *why* certain trends might happen (e.g., seasonality, market factors) even if not in the csv.
   - You ARE allowed to define business terms and suggest standard strategies.

4. **HELPFULNESS**:
   - Answer the user's question DIRECTLY. Do not just list data.
   - If the user asks for a strategy, provide one! Base it on the data you see.
   - "Data Limitations" section is optional - include it only if it critically affects the answer.

ANALYSIS FRAMEWORK:

**Structured Response Format**:
- **Executive Summary**: Direct answer to the user's question.
- **Key Insights**: What does the data tell us? (Connect the dots).
- **Strategic Recommendations**: What should we do? (Use your expert knowledge).
- **Supporting Data**: The key numbers that back up your advice.

**Tone**:
- Professional, encouraging, and insightful.
- Fluent Thai or English (match user's language).

Context Information:
{context}
"""


        
        human_template = """User Query: {query}

Please provide a helpful strategic analysis matching the framework above. 

Remember:
- Be insightful and proactive in your advice.
- Cite specific numbers from the context to back up your points.
- If you use general business knowledge, make it clear it's general advice.

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
