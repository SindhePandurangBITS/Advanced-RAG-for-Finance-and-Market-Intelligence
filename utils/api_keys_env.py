# src/utils/api_keys_env.py

import os
from dotenv import load_dotenv

def check_and_print_api_keys(required_keys=None, optional_keys=None):
    """
    Loads .env file, checks for required API keys, and prints all keys.
    Raises an error if any required key is missing.
    """
    load_dotenv()
    required_keys = required_keys or [
        "OPENAI_API_KEY", 
        "LANGCHAIN_API_KEY"
    ]
    optional_keys = optional_keys or [
        "COHERE_API_KEY",
        "TRAVILY_API_KEY",
        "LANGCHAIN_ENDPOINT",
        "LANGCHAIN_TRACING_V2"
    ]
    missing = [key for key in required_keys if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required API keys in your .env file: {', '.join(missing)}")
    
    print("---- API Keys and Endpoints ----")
    for key in required_keys + optional_keys:
        print(f"{key:25}= {os.getenv(key)}")
    print("API keys have been set!\n")

if __name__ == "__main__":
    check_and_print_api_keys()
