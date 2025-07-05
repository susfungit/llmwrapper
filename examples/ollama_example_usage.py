#!/usr/bin/env python3
"""
Example usage of LLM Wrapper with Ollama (local LLM inference).

This example demonstrates how to use the Ollama wrapper for local LLM inference.
Ollama must be installed and running on your local machine.

Prerequisites:
1. Install Ollama: https://ollama.ai/
2. Run Ollama: `ollama serve`
3. Pull a model: `ollama pull llama3`

Usage:
    python examples/ollama_example_usage.py
"""

import sys
sys.path.insert(0, '.')

from llmwrapper import get_llm, logger

def main():
    """Demonstrate Ollama wrapper usage."""
    
    print("🦙 Ollama LLM Wrapper Example")
    print("=" * 50)
    
    # Configuration for Ollama
    config = {
        "model": "llama3",                    # Model to use
        "base_url": "http://localhost:11434", # Ollama server URL
        "api_key": None                       # Not needed for local
    }
    
    try:
        # Create Ollama LLM instance
        llm = get_llm("ollama", config)
        print(f"✅ Successfully initialized Ollama with model: {config['model']}")
        
        # List available models
        try:
            models = llm.list_models()
            print(f"📋 Available models: {models}")
        except Exception as e:
            print(f"⚠️  Could not list models: {e}")
        
        # Example 1: Simple chat
        print("\n1. Simple Chat Example:")
        print("-" * 30)
        
        messages = [
            {"role": "user", "content": "Hello! What's your name?"}
        ]
        
        response = llm.chat(messages)
        print(f"🤖 Response: {response}")
        
        # Example 2: System prompt with conversation
        print("\n2. System Prompt Example:")
        print("-" * 30)
        
        messages = [
            {"role": "system", "content": "You are a helpful Python programming assistant."},
            {"role": "user", "content": "Write a simple Python function to calculate factorial."}
        ]
        
        response = llm.chat(messages)
        print(f"🤖 Response: {response}")
        
        # Example 3: Parameters customization
        print("\n3. Custom Parameters Example:")
        print("-" * 30)
        
        messages = [
            {"role": "user", "content": "Tell me a creative story about a robot."}
        ]
        
        response = llm.chat(
            messages,
            temperature=0.8,    # More creative
            max_tokens=100,     # Shorter response
            top_p=0.9
        )
        print(f"🤖 Response: {response}")
        
        # Example 4: Multi-turn conversation
        print("\n4. Multi-turn Conversation:")
        print("-" * 30)
        
        conversation = [
            {"role": "user", "content": "What's the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What's the population of that city?"}
        ]
        
        response = llm.chat(conversation)
        print(f"🤖 Response: {response}")
        
        print("\n✅ All examples completed successfully!")
        
    except ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure Ollama is installed: https://ollama.ai/")
        print("   2. Start Ollama server: `ollama serve`")
        print("   3. Pull a model: `ollama pull llama3`")
        print("   4. Check if server is running: `curl http://localhost:11434/api/tags`")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main() 