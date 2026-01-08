"""
Configuration management for StratAI backend.
Uses Pydantic Settings for environment variable management.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Provider Configuration
    llm_provider: str = Field(default="ollama", env="LLM_PROVIDER")  # "openai" or "ollama"
    
    # OpenAI Configuration (optional now)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    
    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
    
    # Tavily Search Configuration
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")
    
    # Application Configuration
    backend_host: str = Field(default="0.0.0.0", env="BACKEND_HOST")
    backend_port: int = Field(default=8000, env="BACKEND_PORT")
    frontend_port: int = Field(default=8501, env="FRONTEND_PORT")
    
    # ChromaDB Configuration
    chroma_persist_dir: str = Field(default="./vectordb", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="internal_knowledge", env="CHROMA_COLLECTION_NAME")
    
    # Embedding Configuration
    embedding_provider: str = Field(default="ollama", env="EMBEDDING_PROVIDER")  # "openai" or "ollama"
    embedding_model: str = Field(default="nomic-embed-text", env="EMBEDDING_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    
    # Model Configuration
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # Chunking Configuration
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Upload Configuration
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env


# Global settings instance
settings = Settings()
