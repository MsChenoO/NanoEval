import yaml
import asyncio
from typing import List, Dict, Any, Optional
from nanoeval.core.model_loader import ModelLoader
from nanoeval.loaders.huggingface_loader import HuggingFaceLoader
from nanoeval.loaders.llama_cpp_loader import LlamaCppLoader
from nanoeval.core.evaluator import Evaluator
from nanoeval.evaluators.distillation.safety_preservation import SafetyPreservationEvaluator

class SmallModelEvaluationPipeline:
    """Orchestrator for small model safety evaluations"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path) if config_path else {}
        self.loader = self._create_loader()
        self.evaluators: List[Evaluator] = [] 

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _create_loader(self) -> ModelLoader:
        """Initialize the configured model loader backend"""
        loader_type = self.config.get('loader', 'huggingface')
        if loader_type == 'huggingface':
            return HuggingFaceLoader()
        elif loader_type in ['gguf', 'llama_cpp']:
            return LlamaCppLoader()
        
        raise ValueError(f"Unsupported loader type: {loader_type}")

    def register_evaluator(self, evaluator: Evaluator):
        """Add an evaluator to the pipeline"""
        self.evaluators.append(evaluator)

    async def evaluate_model(self, model_path: str) -> Dict[str, Any]:
        """Run all registered safety evaluations on a single model"""
        print(f"[*] Starting evaluation for: {model_path}")
        self.loader.load(model_path)
        model_info = self.loader.get_info()
        
        results = {}
        for evaluator in self.evaluators:
            print(f"  Running Evaluator: {evaluator.name}...")
            results[evaluator.name] = await evaluator.evaluate(self.loader)
        
        self.loader.unload()
        return {
            "model_info": model_info,
            "results": results,
            "overall_score": self._calculate_overall_score(results)
        }

    async def evaluate_model_pair(self, teacher_path: str, student_path: str) -> Dict[str, Any]:
        """Compare teacher and student models for distillation safety preservation"""
        print(f"[*] Comparing Distillation Safety: {teacher_path} -> {student_path}")
        
        # We need two loaders. self.loader is for the teacher (or primary).
        teacher_loader = self.loader
        student_loader = self._create_loader()
        
        print("  Loading Teacher...")
        teacher_loader.load(teacher_path)
        
        print("  Loading Student...")
        student_loader.load(student_path)
        
        # Instantiate the specific evaluator for comparison
        # In a real app, this path should be configurable
        preservation_eval = SafetyPreservationEvaluator("benchmarks/safety_critical_prompts.jsonl")
        
        print("  Running Safety Preservation Audit...")
        results = await preservation_eval.evaluate_pair(teacher_loader, student_loader)
        
        # Cleanup
        teacher_loader.unload()
        student_loader.unload()
        
        return {
            "teacher_path": teacher_path,
            "student_path": student_path,
            "results": results
        }

    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Aggregate scores from all evaluators"""
        return 0.0 # Placeholder
