from llmwrapper.factory import get_llm

config = {
    "api_key": "your-api-key",
    "model": "gpt-4"
}

llm = get_llm("openai", config)
response = llm.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
])

print("LLM Response:", response)