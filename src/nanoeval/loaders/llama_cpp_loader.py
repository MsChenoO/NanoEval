import time
import os
from typing import Any, Dict, List, Optional
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from nanoeval.core.model_loader import ModelLoader, ModelInfo, ModelResponse

class LlamaCppLoader(ModelLoader):
    """
    Loader for GGUF models using the llama-cpp-python library.
    Ideal for testing quantized models on edge devices.
    """

    def __init__(self):
        if Llama is None:
            raise ImportError(
                "llama-cpp-python not installed. Please install it with: "
                "pip install llama-cpp-python"
            )
        self.model: Optional[Llama] = None
        self._model_path: Optional[str] = None

    def load(self, model_path: str, **kwargs) -> Any:
        """
        Load a GGUF model.
        Common kwargs: n_ctx, n_gpu_layers, n_threads.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model file not found at: {model_path}")

        self._model_path = model_path
        
        # Default optimized settings for local evaluation
        load_params = {
            "model_path": model_path,
            "n_ctx": kwargs.get("n_ctx", 2048),
            "n_gpu_layers": kwargs.get("n_gpu_layers", -1), # -1 uses all available GPU layers
            "verbose": False
        }
        # Update with any user-provided overrides
        load_params.update(kwargs)
        
        self.model = Llama(**load_params)
        return self.model

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate text using the llama.cpp backend"""
        if not self.model:
            raise RuntimeError("Model must be loaded before generation.")

        start_time = time.time()
        
        # Generation params
        gen_params = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "stop": kwargs.get("stop", []),
            "echo": False
        }
        
        output = self.model(**gen_params)
        
        latency = (time.time() - start_time) * 1000
        
        # Extract text and metrics
        # Note: llama-cpp-python returns token counts but not the full sequence easily in this call
        text = output["choices"][0]["text"]
        
        return ModelResponse(
            text=text,
            tokens=[], # llama-cpp doesn't return generated token IDs in the standard completion call
            latency_ms=latency,
            memory_used_mb=0 # Memory tracking for llama.cpp requires OS-level monitoring
        )

    def get_info(self) -> ModelInfo:
        """Extract metadata from the GGUF model"""
        if not self.model:
            raise RuntimeError("Model must be loaded to retrieve info.")
            
        # llama-cpp-python exposes some metadata in the .metadata attribute
        metadata = getattr(self.model, "metadata", {})
        
        return ModelInfo(
            name=os.path.basename(self._model_path or "unknown"),
            architecture=metadata.get("general.architecture", "unknown"),
            parameters=int(metadata.get("general.parameter_count", 0)),
            quantization=self._detect_quantization_from_path(),
            context_length=self.model.n_ctx(),
            vocab_size=self.model.n_vocab(),
            metadata=metadata
        )

    def _detect_quantization_from_path(self) -> str:
        """Heuristic to find quantization level (e.g. Q4_K_M) in filename"""
        if not self._model_path:
            return "unknown"
        filename = os.path.basename(self._model_path).upper()
        # Look for patterns like Q4_K_M, Q8_0, etc.
        import re
        match = re.search(r'Q\d[_A-Z\d]*', filename)
        return match.group(0) if match else "unknown"

    def unload(self):
        """Free model memory"""
        if self.model:
            del self.model
            self.model = None
        self._model_path = None
