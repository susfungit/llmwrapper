"""
Security tests for the LLM wrapper library.
Tests all security features including credential masking, input validation, and security logging.
"""

import pytest
import logging
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llmwrapper.security_utils import SecurityUtils
from llmwrapper.logger import logger, SecureFormatter
from llmwrapper.openai_wrapper import OpenAIWrapper
from llmwrapper.ollama_wrapper import OllamaWrapper

class TestSecurityUtils:
    """Test the SecurityUtils class functionality"""

    def test_mask_sensitive_data_dict(self):
        """Test masking sensitive data in dictionaries"""
        data = {
            "api_key": "sk-1234567890abcdef1234567890abcdef",
            "password": "mySecretPassword123",
            "token": "abc123def456ghi789",
            "safe_field": "this is safe"
        }
        
        masked = SecurityUtils.mask_sensitive_data(data)
        
        assert masked["api_key"] == "sk-***"
        assert "***" in masked["password"]  # Should contain masked portion
        assert "***" in masked["token"]    # Should contain masked portion
        assert masked["safe_field"] == "this is safe"

    def test_mask_sensitive_data_nested_dict(self):
        """Test masking sensitive data in nested dictionaries"""
        data = {
            "config": {
                "api_key": "sk-1234567890abcdef1234567890abcdef",
                "safe_value": "no secrets here"
            },
            "credentials": {
                "secret": "topsecret",
                "auth": "bearer_token_123"
            }
        }
        
        masked = SecurityUtils.mask_sensitive_data(data)
        
        assert masked["config"]["api_key"] == "sk-***"
        assert masked["config"]["safe_value"] == "no secrets here"
        assert "***" in str(masked["credentials"]["secret"])
        assert "***" in str(masked["credentials"]["auth"])

    def test_mask_sensitive_data_list(self):
        """Test masking sensitive data in lists"""
        data = [
            {"api_key": "sk-test123456789"},
            {"password": "secret123"},
            {"safe_field": "safe value"}
        ]
        
        masked = SecurityUtils.mask_sensitive_data(data)
        
        assert "***" in str(masked[0]["api_key"])
        assert "***" in str(masked[1]["password"])
        assert masked[2]["safe_field"] == "safe value"

    def test_mask_string_api_keys(self):
        """Test masking API keys in strings"""
        test_cases = [
            ("sk-1234567890abcdef1234567890abcdef", "sk-***"),
            ("sk-ant-api03-abcdef1234567890abcdef1234567890", "sk-ant-***"),
            ("AIzaSyDabcdef1234567890abcdef1234567890", "AIza***"),
            ("xai-abcdef1234567890abcdef1234567890", "xai-***"),
            ("normal text without secrets", "normal text without secrets")
        ]
        
        for original, expected in test_cases:
            result = SecurityUtils._mask_string(original)
            assert result == expected

    def test_validate_api_key_openai(self):
        """Test OpenAI API key validation"""
        # Valid keys
        assert SecurityUtils.validate_api_key("sk-1234567890abcdef1234567890abcdef1234567890abcdef", "openai")
        assert SecurityUtils.validate_api_key("sk-proj-abcdef1234567890abcdef1234567890abcdef1234567890", "openai")
        
        # Invalid keys
        assert not SecurityUtils.validate_api_key("invalid-key", "openai")
        assert not SecurityUtils.validate_api_key("sk-short", "openai")
        assert not SecurityUtils.validate_api_key("", "openai")
        assert not SecurityUtils.validate_api_key(None, "openai")

    def test_validate_api_key_anthropic(self):
        """Test Anthropic API key validation"""
        # Valid keys
        assert SecurityUtils.validate_api_key("sk-ant-api03-abcdef1234567890abcdef1234567890", "anthropic")
        assert SecurityUtils.validate_api_key("sk-ant-sid-abcdef1234567890abcdef1234567890", "anthropic")
        
        # Invalid keys
        assert not SecurityUtils.validate_api_key("sk-wrong-format", "anthropic")
        assert not SecurityUtils.validate_api_key("invalid-key", "anthropic")
        assert not SecurityUtils.validate_api_key("", "anthropic")
        assert not SecurityUtils.validate_api_key(None, "anthropic")

    def test_validate_api_key_gemini(self):
        """Test Gemini API key validation"""
        # Valid keys
        assert SecurityUtils.validate_api_key("AIzaSyDabcdef1234567890abcdef1234567890", "gemini")
        assert SecurityUtils.validate_api_key("AIzaABCDEF1234567890abcdef1234567890", "gemini")
        
        # Invalid keys
        assert not SecurityUtils.validate_api_key("invalid-key", "gemini")
        assert not SecurityUtils.validate_api_key("AIza-short", "gemini")
        assert not SecurityUtils.validate_api_key("", "gemini")
        assert not SecurityUtils.validate_api_key(None, "gemini")

    def test_validate_api_key_grok(self):
        """Test Grok API key validation"""
        # Valid keys
        assert SecurityUtils.validate_api_key("xai-abcdef1234567890abcdef1234567890", "grok")
        assert SecurityUtils.validate_api_key("xai-1234567890abcdef1234567890abcdef", "grok")
        
        # Invalid keys
        assert not SecurityUtils.validate_api_key("invalid-key", "grok")
        assert not SecurityUtils.validate_api_key("xai-short", "grok")
        assert not SecurityUtils.validate_api_key("", "grok")
        assert not SecurityUtils.validate_api_key(None, "grok")

    def test_validate_api_key_ollama(self):
        """Test Ollama API key validation (should be None or empty)"""
        # Valid (None or empty)
        assert SecurityUtils.validate_api_key(None, "ollama")
        assert SecurityUtils.validate_api_key("", "ollama")
        
        # Invalid (any non-None/non-empty value)
        assert not SecurityUtils.validate_api_key("any-key", "ollama")
        assert not SecurityUtils.validate_api_key("sk-test", "ollama")

    def test_validate_api_key_generic_provider(self):
        """Test generic API key validation for unknown providers"""
        # Valid keys (length between 16-200)
        assert SecurityUtils.validate_api_key("a" * 16, "unknown")
        assert SecurityUtils.validate_api_key("a" * 50, "unknown")
        assert SecurityUtils.validate_api_key("a" * 200, "unknown")
        
        # Invalid keys
        assert not SecurityUtils.validate_api_key("short", "unknown")
        assert not SecurityUtils.validate_api_key("a" * 15, "unknown")
        assert not SecurityUtils.validate_api_key("a" * 201, "unknown")
        assert not SecurityUtils.validate_api_key("", "unknown")
        assert not SecurityUtils.validate_api_key(None, "unknown")

    def test_validate_url(self):
        """Test URL validation"""
        # Valid URLs
        assert SecurityUtils.validate_url("https://api.openai.com/v1")
        assert SecurityUtils.validate_url("http://localhost:11434")
        assert SecurityUtils.validate_url("https://secure.example.com")
        assert SecurityUtils.validate_url("http://127.0.0.1:8080")
        
        # Invalid URLs
        assert not SecurityUtils.validate_url("ftp://invalid.com")
        assert not SecurityUtils.validate_url("invalid-url")
        assert not SecurityUtils.validate_url("")
        assert not SecurityUtils.validate_url("javascript:alert('xss')")

    def test_validate_messages_valid(self):
        """Test message validation with valid messages"""
        valid_messages = [
            [{"role": "user", "content": "Hello, world!"}],
            [{"role": "system", "content": "You are helpful"}],
            [{"role": "assistant", "content": "I'm here to help"}],
            [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        ]
        
        for messages in valid_messages:
            assert SecurityUtils.validate_messages(messages)

    def test_validate_messages_invalid(self):
        """Test message validation with invalid messages"""
        invalid_messages = [
            "not a list",
            [{"content": "Missing role"}],
            [{"role": "user"}],  # Missing content
            [{"role": "invalid", "content": "Bad role"}],
            [{"role": "user", "content": 123}],  # Non-string content
            [{"role": "user", "content": "<script>alert('xss')</script>"}],
            [{"role": "user", "content": "eval(malicious_code)"}],
            [{"role": "user", "content": "system('rm -rf /')"}],
        ]
        
        for messages in invalid_messages:
            assert not SecurityUtils.validate_messages(messages)
        
        # Test empty list separately
        assert not SecurityUtils.validate_messages([])

    def test_contains_injection_patterns(self):
        """Test injection pattern detection"""
        dangerous_patterns = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "eval(malicious_code)",
            "exec(dangerous_function)",
            "system('rm -rf /')",
            "<SCRIPT>alert('XSS')</SCRIPT>",  # Case insensitive
        ]
        
        safe_patterns = [
            "Hello, world!",
            "What is 2+2?",
            "Please help me with Python",
            "script in the movie was good",  # Contains 'script' but not dangerous
        ]
        
        for pattern in dangerous_patterns:
            assert SecurityUtils._contains_injection_patterns(pattern)
        
        for pattern in safe_patterns:
            assert not SecurityUtils._contains_injection_patterns(pattern)

    def test_sanitize_config(self):
        """Test configuration sanitization"""
        config = {
            "api_key": "sk-1234567890abcdef1234567890abcdef",
            "password": "secret123",
            "model": "gpt-4",
            "temperature": 0.7
        }
        
        sanitized = SecurityUtils.sanitize_config(config)
        
        assert sanitized["api_key"] == "sk-***"
        assert "***" in str(sanitized["password"])
        assert sanitized["model"] == "gpt-4"
        assert sanitized["temperature"] == 0.7

    @patch('llmwrapper.security_utils.logger')
    def test_log_security_event(self, mock_logger):
        """Test security event logging"""
        event_details = {
            "api_key": "sk-1234567890abcdef1234567890abcdef",
            "provider": "openai",
            "safe_data": "this is safe"
        }
        
        SecurityUtils.log_security_event("TEST_EVENT", event_details)
        
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        assert "SECURITY EVENT - TEST_EVENT" in call_args
        assert "sk-***" in call_args
        assert "this is safe" in call_args


class TestSecureLogging:
    """Test the secure logging functionality"""

    def test_secure_formatter_api_key_masking(self):
        """Test that SecureFormatter masks API keys"""
        formatter = SecureFormatter()
        
        # Create a mock log record
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Initializing with api_key: sk-1234567890abcdef1234567890abcdef",
            args=(), exc_info=None
        )
        
        formatted = formatter.format(record)
        assert "sk-***" in formatted
        assert "sk-1234567890abcdef1234567890abcdef" not in formatted

    def test_secure_formatter_multiple_patterns(self):
        """Test that SecureFormatter masks multiple sensitive patterns"""
        formatter = SecureFormatter()
        
        test_cases = [
            ("api_key: sk-1234567890abcdef1234567890abcdef", "***"),
            ("token: sk-ant-api03-abcdef1234567890abcdef1234567890", "***"),
            ("password: secret123", "***"),
            ("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "***"),
            ("https://user:password@api.example.com", "***"),
        ]
        
        for original, expected_pattern in test_cases:
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg=original, args=(), exc_info=None
            )
            
            formatted = formatter.format(record)
            # Check that some form of masking occurred
            assert "***" in formatted or "sk-***" in formatted or "AIza***" in formatted

    def test_logger_integration(self):
        """Test that the logger uses SecureFormatter"""
        from llmwrapper.logger import SecureFormatter
        
        # Capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(SecureFormatter())  # Use SecureFormatter
        
        # Remove existing handlers and add test handler
        logger.handlers.clear()
        logger.addHandler(handler)
        
        # Log a message with sensitive data
        logger.info("Testing with api_key: sk-1234567890abcdef1234567890abcdef")
        
        # Check that the output is masked
        output = log_stream.getvalue()
        assert "***" in output
        assert "sk-1234567890abcdef1234567890abcdef" not in output

    def test_logging_mixin_token_usage_masking(self):
        """Test that LoggingMixin handles token usage data properly"""
        from llmwrapper.logging_mixin import LoggingMixin
        from llmwrapper.logger import SecureFormatter
        
        mixin = LoggingMixin()
        
        # Mock usage data that might contain sensitive info
        usage_data = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "api_key": "sk-1234567890abcdef1234567890abcdef"  # Shouldn't be here but test masking
        }
        
        # Capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(SecureFormatter())  # Use SecureFormatter
        logger.handlers.clear()
        logger.addHandler(handler)
        
        mixin.log_token_usage("openai", usage_data)
        
        output = log_stream.getvalue()
        # The log should contain the token counts
        assert "10" in output
        assert "20" in output 
        assert "30" in output
        # The API key should not be logged (it's not part of the token usage log format)
        assert "sk-1234567890abcdef1234567890abcdef" not in output


class TestWrapperSecurity:
    """Test security features in wrapper classes"""

    def test_openai_wrapper_api_key_validation(self):
        """Test OpenAI wrapper validates API keys"""
        # Valid API key should work
        with patch('llmwrapper.openai_wrapper.OpenAI'):
            wrapper = OpenAIWrapper("sk-1234567890abcdef1234567890abcdef1234567890abcdef", "gpt-4")
            assert wrapper.model == "gpt-4"
        
        # Invalid API key should raise error
        with pytest.raises(ValueError, match="Invalid OpenAI API key format"):
            OpenAIWrapper("invalid-key", "gpt-4")

    def test_openai_wrapper_message_validation(self):
        """Test OpenAI wrapper validates messages"""
        with patch('llmwrapper.openai_wrapper.OpenAI') as mock_openai:
            # Set up the mock properly
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Hello back!"
            mock_response.usage = None
            
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            wrapper = OpenAIWrapper("sk-1234567890abcdef1234567890abcdef1234567890abcdef", "gpt-4")
            
            # Valid messages should work
            valid_messages = [{"role": "user", "content": "Hello"}]
            
            response = wrapper.chat(valid_messages)
            assert response == "Hello back!"
            
            # Invalid messages should raise error
            invalid_messages = [{"role": "user", "content": "<script>alert('xss')</script>"}]
            with pytest.raises(ValueError, match="Invalid message format"):
                wrapper.chat(invalid_messages)

    @patch('llmwrapper.ollama_wrapper.requests.get')
    def test_ollama_wrapper_url_validation(self, mock_get):
        """Test Ollama wrapper validates URLs"""
        # Mock successful connection
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Valid URL should work
        wrapper = OllamaWrapper("llama3", "http://localhost:11434", None)
        assert wrapper.base_url == "http://localhost:11434"
        
        # Invalid URL should raise error
        with pytest.raises(ValueError, match="Invalid Ollama base URL format"):
            OllamaWrapper("llama3", "invalid-url", None)

    @patch('llmwrapper.ollama_wrapper.requests.get')
    def test_ollama_wrapper_api_key_validation(self, mock_get):
        """Test Ollama wrapper validates API keys (should be None)"""
        # Mock successful connection
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # None API key should work
        wrapper = OllamaWrapper("llama3", "http://localhost:11434", None)
        assert wrapper.api_key is None
        
        # Non-None API key should raise error
        with pytest.raises(ValueError, match="Invalid API key for Ollama"):
            OllamaWrapper("llama3", "http://localhost:11434", "should-be-none")

    @patch('llmwrapper.ollama_wrapper.requests.get')
    @patch('llmwrapper.ollama_wrapper.requests.post')
    def test_ollama_wrapper_parameter_validation(self, mock_post, mock_get):
        """Test Ollama wrapper validates parameters"""
        # Mock successful connection
        mock_get_response = MagicMock()
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        # Mock chat response
        mock_post_response = MagicMock()
        mock_post_response.raise_for_status.return_value = None
        mock_post_response.json.return_value = {"response": "Hello back!"}
        mock_post.return_value = mock_post_response
        
        wrapper = OllamaWrapper("llama3", "http://localhost:11434", None)
        
        # Valid parameters should work
        valid_messages = [{"role": "user", "content": "Hello"}]
        response = wrapper.chat(valid_messages, temperature=0.7, max_tokens=100)
        assert response == "Hello back!"
        
        # Invalid temperature should raise error
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            wrapper.chat(valid_messages, temperature=3.0)
        
        # Invalid max_tokens should raise error
        with pytest.raises(ValueError, match="max_tokens must be between 1 and 32768"):
            wrapper.chat(valid_messages, max_tokens=50000)

    @patch('llmwrapper.ollama_wrapper.requests.get')
    @patch('llmwrapper.ollama_wrapper.requests.post')
    def test_ollama_wrapper_security_logging(self, mock_post, mock_get):
        """Test Ollama wrapper logs security events"""
        # Mock successful connection
        mock_get_response = MagicMock()
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response
        
        wrapper = OllamaWrapper("llama3", "http://localhost:11434", None)
        
        # Mock failed request to test security logging
        mock_post.side_effect = Exception("Connection failed")
        
        with patch.object(wrapper, 'log_security_event') as mock_log_security:
            valid_messages = [{"role": "user", "content": "Hello"}]
            
            with pytest.raises(Exception):
                wrapper.chat(valid_messages)
            
            # Should log security event
            mock_log_security.assert_called()
            call_args = mock_log_security.call_args[0]
            assert call_args[0] == "UNEXPECTED_ERROR"


class TestSecurityIntegration:
    """Test end-to-end security integration"""

    def test_security_event_logging_integration(self):
        """Test that security events are properly logged"""
        # Capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger.handlers.clear()
        logger.addHandler(handler)
        
        # Log a security event
        SecurityUtils.log_security_event("TEST_INTEGRATION", {
            "api_key": "sk-1234567890abcdef1234567890abcdef",
            "provider": "test",
            "action": "validation_test"  # Changed to avoid masking
        })
        
        output = log_stream.getvalue()
        assert "SECURITY EVENT - TEST_INTEGRATION" in output
        assert "sk-***" in output
        assert "validation_test" in output

    def test_comprehensive_data_masking(self):
        """Test comprehensive data masking across different data types"""
        complex_data = {
            "users": [
                {
                    "id": 1,
                    "api_key": "sk-1234567890abcdef1234567890abcdef",
                    "name": "John Doe"
                },
                {
                    "id": 2,
                    "password": "secret123",
                    "email": "jane@example.com"
                }
            ],
            "config": {
                "database": {
                    "host": "localhost",
                    "credentials": {
                        "username": "admin",
                        "password": "dbpassword123"
                    }
                },
                "api_settings": {
                    "token": "bearer_token_12345",
                    "timeout": 30
                }
            }
        }
        
        masked = SecurityUtils.mask_sensitive_data(complex_data)
        
        # Check that sensitive data is masked
        assert masked["users"][0]["api_key"] == "sk-***"
        assert masked["users"][0]["name"] == "John Doe"  # Safe data preserved
        assert "***" in str(masked["users"][1]["password"])
        assert masked["users"][1]["email"] == "jane@example.com"  # Safe data preserved
        assert "***" in str(masked["config"]["database"]["credentials"]["password"])
        assert "***" in str(masked["config"]["api_settings"]["token"])
        assert masked["config"]["api_settings"]["timeout"] == 30  # Safe data preserved

    def test_security_validation_chain(self):
        """Test the complete security validation chain"""
        # Test that all security validations work together
        test_config = {
            "api_key": "sk-1234567890abcdef1234567890abcdef1234567890abcdef",
            "model": "gpt-4",
            "temperature": 0.7
        }
        
        # Validate API key
        assert SecurityUtils.validate_api_key(test_config["api_key"], "openai")
        
        # Validate messages
        test_messages = [{"role": "user", "content": "Hello, world!"}]
        assert SecurityUtils.validate_messages(test_messages)
        
        # Test masking
        masked_config = SecurityUtils.mask_sensitive_data(test_config)
        assert masked_config["api_key"] == "sk-***"
        assert masked_config["model"] == "gpt-4"
        
        # Test security logging
        with patch('llmwrapper.security_utils.logger') as mock_logger:
            SecurityUtils.log_security_event("VALIDATION_SUCCESS", test_config)
            mock_logger.warning.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 