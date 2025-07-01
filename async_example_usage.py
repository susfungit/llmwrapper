import asyncio
import json
import time
from async_factory import get_async_llm

async def async_single_request_example():
    """Example of making a single async request"""
    print("=== Async Single Request Example ===")
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create async LLM instance
    llm = get_async_llm("openai", config["openai"])
    
    # Make async request
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    start_time = time.time()
    response = await llm.chat(messages)
    end_time = time.time()
    
    print(f"Response: {response}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

async def async_concurrent_requests_example():
    """Example of making concurrent async requests to multiple providers"""
    print("\n=== Async Concurrent Requests Example ===")
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create multiple async LLM instances
    openai_llm = get_async_llm("openai", config["openai"])
    anthropic_llm = get_async_llm("anthropic", config["anthropic"])
    
    # Define the same question for both
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Give a very brief answer."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    # Make concurrent requests
    start_time = time.time()
    
    # Run both requests concurrently
    openai_task = openai_llm.chat(messages)
    anthropic_task = anthropic_llm.chat(messages)
    
    # Wait for both to complete
    openai_response, anthropic_response = await asyncio.gather(
        openai_task, 
        anthropic_task
    )
    
    end_time = time.time()
    
    print(f"OpenAI Response: {openai_response}")
    print(f"Anthropic Response: {anthropic_response}")
    print(f"Total time for both requests: {end_time - start_time:.2f} seconds")

async def async_batch_processing_example():
    """Example of processing multiple questions concurrently"""
    print("\n=== Async Batch Processing Example ===")
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create async LLM instance
    llm = get_async_llm("openai", config["openai"])
    
    # Define multiple questions
    questions = [
        "What is the capital of Spain?",
        "What is 10 * 15?",
        "Name one programming language.",
        "What color is the sky?",
        "What is the largest planet?"
    ]
    
    # Create tasks for all questions
    tasks = []
    for question in questions:
        messages = [
            {"role": "system", "content": "Give a very brief answer."},
            {"role": "user", "content": question}
        ]
        task = llm.chat(messages)
        tasks.append((question, task))
    
    # Execute all tasks concurrently
    start_time = time.time()
    results = await asyncio.gather(*[task for _, task in tasks])
    end_time = time.time()
    
    # Print results
    print("Results:")
    for i, ((question, _), response) in enumerate(zip(tasks, results)):
        print(f"{i+1}. Q: {question}")
        print(f"   A: {response}")
    
    print(f"Total time for {len(questions)} questions: {end_time - start_time:.2f} seconds")

async def async_error_handling_example():
    """Example of error handling in async requests"""
    print("\n=== Async Error Handling Example ===")
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create async LLM instance with potentially invalid config
    try:
        # This might fail if the API key is invalid
        invalid_config = config["openai"].copy()
        invalid_config["api_key"] = "invalid_key"
        llm = get_async_llm("openai", invalid_config)
        
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        response = await llm.chat(messages)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("This demonstrates proper error handling in async operations")

async def compare_sync_vs_async():
    """Compare performance of sync vs async operations"""
    print("\n=== Sync vs Async Performance Comparison ===")
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Note: This example requires importing the sync factory as well
    # from factory import get_llm
    
    print("For a true comparison, you would:")
    print("1. Make 3 sequential sync requests (sync_time)")
    print("2. Make 3 concurrent async requests (async_time)")
    print("3. Compare async_time << sync_time")
    print("Async operations should be significantly faster for I/O bound tasks like API calls")

async def main():
    """Main function to run all async examples"""
    print("LLM Wrapper - Async Examples")
    print("=" * 50)
    
    try:
        await async_single_request_example()
        await async_concurrent_requests_example()
        await async_batch_processing_example()
        await async_error_handling_example()
        await compare_sync_vs_async()
        
    except FileNotFoundError:
        print("Error: config.json not found. Please ensure your configuration file exists.")
    except KeyError as e:
        print(f"Error: Missing configuration key: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Run the async examples
    asyncio.run(main()) 