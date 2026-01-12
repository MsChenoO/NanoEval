import time
import torch
from typing import Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanoeval.core.model_loader import ModelLoader, ModelInfo, ModelResponse

class HuggingFaceLoader(ModelLoader):
    """Implementation of ModelLoader for Hugging Face Transformers"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_path = None

    def load(self, model_path: str, **kwargs) -> Any:
        """
        Load a model using the transformers library.
        Supports automatic device mapping and optional quantization.
        """
        self._model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Merge default loading args with user overrides
        load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "trust_remote_code": True,
        }
        load_kwargs.update(kwargs)
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        return self.model

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate text with basic performance and memory tracking"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before generation.")

        start_time = time.time()
        
        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        # Reset memory tracking if on CUDA
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()

        # Generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                do_sample=kwargs.get("do_sample", True),
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )

        latency = (time.time() - start_time) * 1000
        
        # Calculate memory usage
        mem_used = 0
        if torch.cuda.is_available():
            mem_used = (torch.cuda.max_memory_allocated() - mem_before) / (1024 * 1024)

        # Process output
        generated_tokens = outputs.sequences[0]
        # Only take the newly generated part
        new_tokens = generated_tokens[input_length:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return ModelResponse(
            text=response_text,
            tokens=new_tokens.tolist(),
            latency_ms=latency,
            memory_used_mb=mem_used
        )

    def get_info(self) -> ModelInfo:
        """Extract technical specifications from the loaded model and config"""
        if not self.model:
            raise RuntimeError("Model must be loaded to retrieve info.")
            
        config = self.model.config
        return ModelInfo(
            name=self._model_path,
            architecture=config.model_type,
            parameters=self.model.num_parameters(),
            quantization=self._detect_quantization_level(),
            context_length=getattr(config, "max_position_embeddings", 2048),
            vocab_size=config.vocab_size,
            metadata=config.to_dict()
        )

    def _detect_quantization_level(self) -> str:
        """Internal helper to identify if the model is running in 4/8-bit"""
        if hasattr(self.model, "is_quantized") and self.model.is_quantized:
            # Check for BitsAndBytes quantization
            if hasattr(self.model, "quantization_method"):
                return self.model.quantization_method
            return "unknown-quantized"
        return "none"

    def unload(self):
        """Clean up resources to free memory"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        
        self.model = None
        self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
