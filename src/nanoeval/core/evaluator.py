from abc import ABC, abstractmethod
from typing import Dict, Any
from nanoeval.core.model_loader import ModelLoader

class Evaluator(ABC):
    """Base class for all safety evaluators"""

    @abstractmethod
    async def evaluate(self, loader: ModelLoader) -> Dict[str, Any]:
        """Run the evaluation logic against the provided model loader"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique identifier for this evaluator"""
        pass
