"""Test environment variable loading"""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("Environment Variables:")
print(f"TAVILY_API_KEY: {os.getenv('TAVILY_API_KEY', 'NOT FOUND')}")
print(f"OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL', 'NOT FOUND')}")
print(f"OLLAMA_MODEL: {os.getenv('OLLAMA_MODEL', 'NOT FOUND')}")

# Test with pydantic settings
from backend.core.config import settings

print("\nPydantic Settings:")
print(f"tavily_api_key: {settings.tavily_api_key}")
print(f"ollama_base_url: {settings.ollama_base_url}")
print(f"ollama_model: {settings.ollama_model}")
