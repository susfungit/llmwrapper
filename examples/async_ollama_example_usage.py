#!/usr/bin/env python3
"""
Async example usage of LLM Wrapper with Ollama (local LLM inference).

This example demonstrates how to use the async Ollama wrapper for local LLM inference.
Ollama must be installed and running on your local machine.

Prerequisites:
1. Install Ollama: https://ollama.ai/
2. Run Ollama: `ollama serve`
3. Pull a model: `ollama pull llama3`

Usage:
    python examples/async_ollama_example_usage.py
"""

import asyncio
import sys
sys.path.insert(0, '.')

from llmwrapper import get_async_llm, logger

async def main():
    """Demonstrate async Ollama wrapper usage."""
    
    print("🦙 Async Ollama LLM Wrapper Example")
    print("=" * 50)
    
    # Configuration for Ollama
    config = {
        "model": "llama3",                    # Model to use
        "base_url": "http://localhost:11434", # Ollama server URL
        "api_key": None                       # Not needed for local
    }
    
    llm = None
    
    try:
        # Create async Ollama LLM instance
        llm = get_async_llm("ollama", config)
        print(f"✅ Successfully initialized async Ollama with model: {config['model']}")
        
        # List available models
        try:
            models = await llm.list_models()
            print(f"📋 Available models: {models}")
        except Exception as e:
            print(f"⚠️  Could not list models: {e}")
        
        # Example 1: Simple async chat
        print("\n1. Simple Async Chat Example:")
        print("-" * 30)
        
        messages = [
            {"role": "user", "content": "Hello! What's your name?"}
        ]
        
        response = await llm.chat(messages)
        print(f"🤖 Response: {response}")
        
        # Example 2: Concurrent requests
        print("\n2. Concurrent Requests Example:")
        print("-" * 30)
        
        questions = [
            "What is Python?",
            "What is machine learning?",
            "What is the meaning of life?"
        ]
        
        # Create tasks for concurrent execution
        tasks = []
        for i, question in enumerate(questions):
            messages = [{"role": "user", "content": question}]
            task = llm.chat(messages, max_tokens=50)
            tasks.append(task)
        
        # Execute all tasks concurrently
        print("🚀 Running 3 concurrent requests...")
        responses = await asyncio.gather(*tasks)
        
        for i, (question, response) in enumerate(zip(questions, responses)):
            print(f"   Q{i+1}: {question}")
            print(f"   A{i+1}: {response[:100]}...")
            print()
        
        # Example 3: System prompt with async context manager
        print("\n3. Async Context Manager Example:")
        print("-" * 30)
        
        async with llm:
            messages = [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a simple async function in Python."}
            ]
            
            response = await llm.chat(messages)
            print(f"🤖 Response: {response}")
        
        # Example 4: Error handling and retry
        print("\n4. Error Handling Example:")
        print("-" * 30)
        
        try:
            # This might fail if model doesn't exist
            config_bad = config.copy()
            config_bad["model"] = "nonexistent-model"
            
            llm_bad = get_async_llm("ollama", config_bad)
            messages = [{"role": "user", "content": "Hello"}]
            response = await llm_bad.chat(messages)
            print(f"🤖 Response: {response}")
            
        except Exception as e:
            print(f"⚠️  Expected error with nonexistent model: {e}")
        
        print("\n✅ All async examples completed successfully!")
        
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
        
    finally:
        # Clean up async resources
        if llm:
            await llm.close()
            print("🧹 Cleaned up async resources")


async def benchmark_example():
    """Demonstrate performance benefits of async operations."""
    
    print("\n🏃 Performance Benchmark Example")
    print("=" * 50)
    
    config = {
        "model": "llama3",
        "base_url": "http://localhost:11434",
        "api_key": None
    }
    
    llm = None
    
    try:
        llm = get_async_llm("ollama", config)
        
        # Sequential vs Concurrent comparison
        questions = [
            "What is 2+2?",
            "What is 3+3?", 
            "What is 4+4?"
        ]
        
        # Sequential execution
        import time
        start_time = time.time()
        
        for question in questions:
            messages = [{"role": "user", "content": question}]
            response = await llm.chat(messages, max_tokens=20)
            print(f"Sequential: {question} -> {response.strip()}")
        
        sequential_time = time.time() - start_time
        
        # Concurrent execution
        start_time = time.time()
        
        tasks = []
        for question in questions:
            messages = [{"role": "user", "content": question}]
            task = llm.chat(messages, max_tokens=20)
            tasks.append((question, task))
        
        results = await asyncio.gather(*[task for _, task in tasks])
        
        for (question, _), response in zip(tasks, results):
            print(f"Concurrent: {question} -> {response.strip()}")
        
        concurrent_time = time.time() - start_time
        
        print(f"\n⏱️  Performance Comparison:")
        print(f"   Sequential: {sequential_time:.2f}s")
        print(f"   Concurrent: {concurrent_time:.2f}s")
        print(f"   Speedup: {sequential_time/concurrent_time:.2f}x")
        
    except Exception as e:
        print(f"❌ Benchmark error: {e}")
        
    finally:
        if llm:
            await llm.close()


if __name__ == "__main__":
    # Run main example
    asyncio.run(main())
    
    # Run benchmark example
    asyncio.run(benchmark_example()) 