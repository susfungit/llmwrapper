"""
Registry system for LLM providers using decorators.
This provides a clean, extensible way to register and manage LLM providers.
"""

from typing import Dict, Type, Any, Optional, Callable
from .logger import logger


class LLMRegistry:
    """Registry for LLM providers with decorator-based registration."""
    
    def __init__(self):
        self._sync_providers: Dict[str, Dict[str, Any]] = {}
        self._async_providers: Dict[str, Dict[str, Any]] = {}
    
    def register_provider(self, name: str, default_model: str, 
                         provider_type: str = "sync", **kwargs):
        """
        Decorator to register LLM providers.
        
        Args:
            name: Provider name (e.g., "openai", "anthropic")
            default_model: Default model for this provider
            provider_type: "sync" or "async"
            **kwargs: Additional provider-specific configuration
        """
        def decorator(cls):
            provider_info = {
                'class': cls,
                'default_model': default_model,
                'config': kwargs
            }
            
            if provider_type == "sync":
                self._sync_providers[name] = provider_info
            elif provider_type == "async":
                self._async_providers[name] = provider_info
            else:
                raise ValueError(f"Invalid provider_type: {provider_type}")
            
            logger.debug(f"Registered {provider_type} provider: {name} -> {cls.__name__}")
            return cls
        return decorator
    
    def get_sync_provider(self, name: str) -> Dict[str, Any]:
        """Get sync provider information."""
        if name not in self._sync_providers:
            available = list(self._sync_providers.keys())
            raise ValueError(f"Unsupported sync provider: {name}. Available: {available}")
        return self._sync_providers[name]
    
    def get_async_provider(self, name: str) -> Dict[str, Any]:
        """Get async provider information."""
        if name not in self._async_providers:
            available = list(self._async_providers.keys())
            raise ValueError(f"Unsupported async provider: {name}. Available: {available}")
        return self._async_providers[name]
    
    def list_providers(self) -> Dict[str, list]:
        """List all registered providers."""
        return {
            'sync': list(self._sync_providers.keys()),
            'async': list(self._async_providers.keys())
        }
    
    def create_sync_llm(self, provider_name: str, config: dict):
        """Create a sync LLM instance."""
        provider_info = self.get_sync_provider(provider_name)
        provider_class = provider_info['class']
        default_model = provider_info['default_model']
        provider_config = provider_info['config']
        
        # Build constructor arguments
        kwargs = {
            'api_key': config['api_key'],
            'model': config.get('model', default_model)
        }
        
        # Add provider-specific arguments
        for key, default_value in provider_config.items():
            kwargs[key] = config.get(key, default_value)
        
        logger.info(f"Instantiating {provider_class.__name__} with model: {kwargs['model']}")
        return provider_class(**kwargs)
    
    def create_async_llm(self, provider_name: str, config: dict):
        """Create an async LLM instance."""
        provider_info = self.get_async_provider(provider_name)
        provider_class = provider_info['class']
        default_model = provider_info['default_model']
        provider_config = provider_info['config']
        
        # Build constructor arguments
        kwargs = {
            'api_key': config['api_key'],
            'model': config.get('model', default_model)
        }
        
        # Add provider-specific arguments
        for key, default_value in provider_config.items():
            kwargs[key] = config.get(key, default_value)
        
        logger.info(f"Instantiating {provider_class.__name__} with model: {kwargs['model']}")
        return provider_class(**kwargs)


# Global registry instance
llm_registry = LLMRegistry()

# Convenience decorators
def register_sync_provider(name: str, default_model: str, **kwargs):
    """Register a synchronous LLM provider."""
    return llm_registry.register_provider(name, default_model, "sync", **kwargs)

def register_async_provider(name: str, default_model: str, **kwargs):
    """Register an asynchronous LLM provider."""
    return llm_registry.register_provider(name, default_model, "async", **kwargs) 