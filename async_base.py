from abc import ABC, abstractmethod

class AsyncBaseLLM(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> str:
        pass 