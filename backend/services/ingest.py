"""
Document ingestion and ETL service.
Handles PDF and CSV file processing, chunking, embedding, and vector storage.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from backend.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document ingestion, processing, and vector storage."""
    
    def __init__(self):
        """Initialize the document processor."""
        # Initialize embeddings based on provider
        if settings.embedding_provider == "ollama":
            from langchain_community.embeddings import OllamaEmbeddings
            self.embeddings = OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.ollama_base_url
            )
            logger.info(f"Using Ollama embeddings: {settings.embedding_model}")
        else:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                openai_api_key=settings.openai_api_key
            )
            logger.info(f"Using OpenAI embeddings: {settings.openai_embedding_model}")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Ensure directories exist
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        os.makedirs(settings.upload_dir, exist_ok=True)
        
        # Initialize or load ChromaDB
        self.vectorstore = self._init_vectorstore()
    
    def _init_vectorstore(self) -> FAISS:
        """Initialize or load the FAISS vector store."""
        try:
            # Create empty FAISS index
            # We'll add documents later with add_documents()
            from langchain_community.docstore.in_memory import InMemoryDocstore
            import faiss
            
            # Initialize with a dummy document to get the embedding dimension
            dummy_doc = Document(page_content="initialization")
            dummy_embedding = self.embeddings.embed_query("initialization")
            dimension = len(dummy_embedding)
            
            # Create FAISS index
            index = faiss.IndexFlatL2(dimension)
            vectorstore = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            
            logger.info(f"FAISS vector store initialized (dimension: {dimension})")
            return vectorstore
        except Exception as e:
            logger.error(f"Error initializing FAISS: {e}")
            raise
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Extract text from PDF file and create document chunks.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects with chunks and metadata
        """
        try:
            logger.info(f"Processing PDF: {file_path}")
            
            # Extract text using PyPDF
            reader = PdfReader(file_path)
            text_content = []
            
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text.strip():
                    text_content.append({
                        "text": page_text,
                        "page": page_num
                    })
            
            # Create documents with metadata
            documents = []
            for item in text_content:
                doc = Document(
                    page_content=item["text"],
                    metadata={
                        "source": os.path.basename(file_path),
                        "page": item["page"],
                        "file_type": "pdf"
                    }
                )
                documents.append(doc)
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {len(reader.pages)} pages")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise
    
    def process_csv(self, file_path: str) -> tuple[List[Document], Optional[pd.DataFrame]]:
        """
        Process CSV file for text extraction and data analysis.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (Document chunks, DataFrame for analysis)
        """
        try:
            logger.info(f"Processing CSV: {file_path}")
            
            # Try different encodings to handle various CSV sources
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    used_encoding = encoding
                    logger.info(f"Successfully read CSV with {encoding} encoding")
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            
            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding")
            
            # Data cleaning
            original_shape = df.shape
            df = df.dropna(how='all')  # Remove completely empty rows
            
            # Try to parse date columns
            for col in df.columns:
                if 'date' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass
            
            logger.info(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns (cleaned from {original_shape})")
            
            # Create text representation for RAG
            csv_summary = self._create_csv_summary(df, file_path)
            
            # Create document chunks
            doc = Document(
                page_content=csv_summary,
                metadata={
                    "source": os.path.basename(file_path),
                    "file_type": "csv",
                    "rows": df.shape[0],
                    "columns": df.shape[1]
                }
            )
            
            chunks = self.text_splitter.split_documents([doc])
            logger.info(f"Created {len(chunks)} chunks from CSV summary")
            
            return chunks, df
            
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {e}")
            raise
    
    def _create_csv_summary(self, df: pd.DataFrame, file_path: str) -> str:
        """Create a comprehensive text summary of CSV data for RAG indexing with anti-hallucination safeguards."""
        summary_parts = [
            f"CSV File: {os.path.basename(file_path)}",
            f"\n{'='*80}",
            f"DATASET OVERVIEW",
            f"{'='*80}",
            f"- Total Rows: {df.shape[0]}",
            f"- Total Columns: {df.shape[1]}",
            f"- Columns: {', '.join(df.columns.tolist())}",
        ]
        
        # Add date range if Date column exists
        if 'Date' in df.columns or 'date' in df.columns:
            date_col = 'Date' if 'Date' in df.columns else 'date'
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                summary_parts.append(f"- Date Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                summary_parts.append(f"⚠️ NOTE: This dataset contains ONLY data for the period above. NO historical comparison data available.")
            except:
                pass
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Overall numeric statistics
        if len(numeric_cols) > 0:
            summary_parts.append(f"\n{'='*80}")
            summary_parts.append("OVERALL NUMERIC STATISTICS")
            summary_parts.append(f"{'='*80}")
            for col in numeric_cols:
                total = df[col].sum()
                mean = df[col].mean()
                min_val = df[col].min()
                max_val = df[col].max()
                summary_parts.append(
                    f"{col}:\n"
                    f"  - Total: {total:,.2f}\n"
                    f"  - Average: {mean:,.2f}\n"
                    f"  - Min: {min_val:,.2f}\n"
                    f"  - Max: {max_val:,.2f}"
                )
        
        # Detailed aggregations by categorical columns
        for cat_col in categorical_cols:
            if cat_col.lower() in ['category', 'region', 'product', 'type', 'segment']:
                summary_parts.append(f"\n{'='*80}")
                summary_parts.append(f"BREAKDOWN BY {cat_col.upper()}")
                summary_parts.append(f"{'='*80}")
                
                grouped = df.groupby(cat_col)
                for name, group in grouped:
                    summary_parts.append(f"\n{cat_col}: {name}")
                    summary_parts.append(f"  - Number of records: {len(group)}")
                    
                    for num_col in numeric_cols:
                        total = group[num_col].sum()
                        avg = group[num_col].mean()
                        summary_parts.append(f"  - Total {num_col}: {total:,.2f}")
                        summary_parts.append(f"  - Average {num_col}: {avg:,.2f}")
        
        # Top items by numeric columns
        if 'Product' in df.columns or 'product' in df.columns:
            prod_col = 'Product' if 'Product' in df.columns else 'product'
            for num_col in numeric_cols:
                summary_parts.append(f"\n{'='*80}")
                summary_parts.append(f"TOP PRODUCTS BY {num_col.upper()}")
                summary_parts.append(f"{'='*80}")
                top_products = df.groupby(prod_col)[num_col].sum().sort_values(ascending=False)
                for i, (prod, value) in enumerate(top_products.items(), 1):
                    summary_parts.append(f"{i}. {prod}: {value:,.2f}")
        
        # Add warning about data limitations
        summary_parts.append(f"\n{'='*80}")
        summary_parts.append("⚠️ IMPORTANT DATA LIMITATIONS")
        summary_parts.append(f"{'='*80}")
        summary_parts.append("- This dataset represents a SNAPSHOT for the period shown above")
        summary_parts.append("- NO historical comparison data (previous year, previous quarter) is available")
        summary_parts.append("- Growth rates and trend comparisons CANNOT be calculated from this data alone")
        summary_parts.append("- Any analysis should focus on current period performance and patterns")
        
        # Add sample rows for reference
        summary_parts.append(f"\n{'='*80}")
        summary_parts.append("SAMPLE DATA (First 5 rows for reference)")
        summary_parts.append(f"{'='*80}")
        summary_parts.append(df.head().to_string())
        
        return "\n".join(summary_parts)
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Main ingestion pipeline: process file and store in vector DB.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                chunks = self.process_pdf(file_path)
                df = None
            elif file_ext == '.csv':
                chunks, df = self.process_csv(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Store in vector database
            self.vectorstore.add_documents(chunks)
            logger.info(f"Added {len(chunks)} chunks to vector store")
            
            return {
                "success": True,
                "file_name": os.path.basename(file_path),
                "file_type": file_ext[1:],
                "chunks_created": len(chunks),
                "has_dataframe": df is not None,
                "message": f"Successfully indexed {len(chunks)} chunks"
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to ingest document: {str(e)}"
            }
    
    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for similar documents in the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection."""
        try:
            # For FAISS, count documents in docstore
            count = len(self.vectorstore.docstore._dict) if hasattr(self.vectorstore, 'docstore') else 0
            
            return {
                "total_documents": count,
                "vector_store_type": "FAISS",
                "storage_location": "In-Memory"
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}


# Global document processor instance
document_processor = DocumentProcessor()
