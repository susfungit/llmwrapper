#!/usr/bin/env python3
"""
Example usage of the LLM Wrapper library.

This file demonstrates various ways to securely configure API keys
and interact with different LLM providers.
"""

import os
import json
from pathlib import Path
from factory import get_llm


def example_with_environment_variables():
    """Example using environment variables (recommended approach)."""
    print("=== Example 1: Using Environment Variables ===")
    
    # Check if API key is available in environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    config = {
        "api_key": api_key,
        "model": os.getenv("OPENAI_MODEL", "gpt-4")  # Default to gpt-4
    }
    
    try:
        llm = get_llm("openai", config)
        response = llm.chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the capital of France?"}
        ])
        print(f"✅ OpenAI Response: {response}")
    except Exception as e:
        print(f"❌ Error with OpenAI: {e}")


def example_with_config_file():
    """Example using a JSON configuration file."""
    print("\n=== Example 2: Using Configuration File ===")
    
    config_file = Path("config.json")
    
    # Create example config file if it doesn't exist
    if not config_file.exists():
        example_config = {
            "openai": {
                "api_key": "your-openai-api-key-here",
                "model": "gpt-4"
            },
            "anthropic": {
                "api_key": "your-anthropic-api-key-here",
                "model": "claude-3-sonnet-20240229"
            },
            "gemini": {
                "api_key": "your-gemini-api-key-here",
                "model": "gemini-pro"
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(example_config, f, indent=2)
        
        print(f"📄 Created example config file: {config_file}")
        print("   Please update it with your actual API keys before running.")
        return
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Try OpenAI from config
        if "openai" in config and config["openai"]["api_key"] != "your-openai-api-key-here":
            llm = get_llm("openai", config["openai"])
            response = llm.chat([
                {"role": "user", "content": "Explain what JSON is in one sentence."}
            ])
            print(f"✅ OpenAI Response: {response}")
        else:
            print("❌ OpenAI API key not configured in config.json")
            
    except Exception as e:
        print(f"❌ Error reading config file: {e}")


def example_multiple_providers():
    """Example showing how to use multiple providers with environment variables."""
    print("\n=== Example 3: Multiple Providers ===")
    
    providers_config = {
        "openai": {
            "env_var": "OPENAI_API_KEY",
            "default_model": "gpt-4",
            "model_env": "OPENAI_MODEL"
        },
        "anthropic": {
            "env_var": "ANTHROPIC_API_KEY", 
            "default_model": "claude-3-sonnet-20240229",
            "model_env": "ANTHROPIC_MODEL"
        },
        "gemini": {
            "env_var": "GEMINI_API_KEY",
            "default_model": "gemini-pro", 
            "model_env": "GEMINI_MODEL"
        },
        "grok": {
            "env_var": "XAI_API_KEY",
            "default_model": "grok-beta",
            "model_env": "GROK_MODEL"
        }
    }
    
    test_message = [{"role": "user", "content": "Say hello in a creative way!"}]
    
    for provider_name, provider_info in providers_config.items():
        api_key = os.getenv(provider_info["env_var"])
        
        if not api_key:
            print(f"⏭️  Skipping {provider_name}: {provider_info['env_var']} not set")
            continue
            
        config = {
            "api_key": api_key,
            "model": os.getenv(provider_info["model_env"], provider_info["default_model"])
        }
        
        try:
            llm = get_llm(provider_name, config)
            response = llm.chat(test_message)
            print(f"✅ {provider_name.title()}: {response}")
        except NotImplementedError:
            print(f"⚠️  {provider_name.title()}: Not implemented yet")
        except Exception as e:
            print(f"❌ {provider_name.title()} Error: {e}")


def example_with_fallback_providers():
    """Example showing fallback between providers."""
    print("\n=== Example 4: Provider Fallback ===")
    
    # Try providers in order of preference
    provider_priority = [
        ("openai", "OPENAI_API_KEY", "gpt-4"),
        ("anthropic", "ANTHROPIC_API_KEY", "claude-3-sonnet-20240229"), 
        ("gemini", "GEMINI_API_KEY", "gemini-pro"),
        ("grok", "XAI_API_KEY", "grok-beta")
    ]
    
    message = [{"role": "user", "content": "What is machine learning?"}]
    
    for provider_name, env_var, model in provider_priority:
        api_key = os.getenv(env_var)
        
        if not api_key:
            print(f"⏭️  {provider_name.title()}: API key not available")
            continue
            
        try:
            config = {"api_key": api_key, "model": model}
            llm = get_llm(provider_name, config)
            response = llm.chat(message)
            print(f"✅ Successfully used {provider_name.title()}: {response}")
            break  # Success! Stop trying other providers
            
        except NotImplementedError:
            print(f"⚠️  {provider_name.title()}: Not implemented, trying next...")
            continue
        except Exception as e:
            print(f"❌ {provider_name.title()} failed: {e}, trying next...")
            continue
    else:
        print("❌ All providers failed or no API keys available")


def print_setup_instructions():
    """Print instructions for setting up API keys."""
    print("🔧 Setup Instructions:")
    print("=" * 50)
    print("To use this example, set up your API keys using environment variables:")
    print()
    print("# For OpenAI:")
    print("export OPENAI_API_KEY='your-openai-api-key'")
    print("export OPENAI_MODEL='gpt-4'  # Optional, defaults to gpt-4")
    print()
    print("# For Anthropic:")
    print("export ANTHROPIC_API_KEY='your-anthropic-api-key'")
    print("export ANTHROPIC_MODEL='claude-3-sonnet-20240229'  # Optional")
    print()
    print("# For Google Gemini:")
    print("export GEMINI_API_KEY='your-gemini-api-key'")
    print("export GEMINI_MODEL='gemini-pro'  # Optional")
    print()
    print("# For xAI Grok:")
    print("export XAI_API_KEY='your-xai-api-key'")
    print("export GROK_MODEL='grok-beta'  # Optional")
    print()
    print("🔗 Get API keys from:")
    print("- OpenAI: https://platform.openai.com/api-keys")
    print("- Anthropic: https://console.anthropic.com/")
    print("- Google Gemini: https://makersuite.google.com/app/apikey")
    print("- xAI Grok: https://console.x.ai/")
    print()


if __name__ == "__main__":
    print("🚀 LLM Wrapper Examples")
    print("=" * 50)
    
    # Check if any API keys are available
    available_keys = []
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY"]:
        if os.getenv(key):
            available_keys.append(key)
    
    if not available_keys:
        print_setup_instructions()
        print("\n⚠️  No API keys found in environment variables.")
        print("   Set up at least one API key to see the examples in action.")
    else:
        print(f"✅ Found API keys: {', '.join(available_keys)}")
    
    # Run examples
    example_with_environment_variables()
    example_with_config_file()
    example_multiple_providers()
    example_with_fallback_providers()
    
    print("\n" + "=" * 50)
    print("✨ Examples completed!")