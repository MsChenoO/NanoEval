from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class ModelInfo:
    """Information about the loaded model"""
    name: str
    architecture: str
    parameters: int
    quantization: str
    context_length: int
    vocab_size: int
    metadata: Dict[str, Any]

@dataclass
class ModelResponse:
    """Structured response from model inference"""
    text: str
    tokens: List[int]
    logprobs: Optional[List[float]] = None
    latency_ms: float = 0
    memory_used_mb: float = 0

class ModelLoader(ABC):
    """Base interface for all local model loading backends"""

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> Any:
        """Load a model from a local path or remote hub"""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate text based on a prompt"""
        pass

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """Retrieve technical specifications of the loaded model"""
        pass

    @abstractmethod
    def unload(self):
        """Free model from memory (CPU/GPU)"""
        pass
