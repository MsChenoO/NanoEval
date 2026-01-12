import unittest
from unittest.mock import MagicMock, patch
import asyncio
import json
import os
from nanoeval.core.pipeline import SmallModelEvaluationPipeline
from nanoeval.evaluators.standard.refusal_rate import RefusalRateEvaluator
from nanoeval.core.model_loader import ModelResponse, ModelInfo

class TestFullPipeline(unittest.TestCase):
    @patch("nanoeval.loaders.huggingface_loader.AutoTokenizer")
    @patch("nanoeval.loaders.huggingface_loader.AutoModelForCausalLM")
    def test_pipeline_flow(self, mock_model_class, mock_tokenizer_class):
        # 1. Setup Mocks for Model Loading
        mock_model = MagicMock()
        mock_model.num_parameters.return_value = 3000000000
        mock_model.config.model_type = "llama"
        mock_model.config.vocab_size = 32000
        mock_model.config.to_dict.return_value = {}
        mock_model.device = "cpu"
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "I'm sorry, I cannot fulfill this request."
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # 2. Setup Pipeline
        pipeline = SmallModelEvaluationPipeline()
        
        # 3. Setup and Register Evaluator
        # Ensure the benchmarks directory exists for the test
        os.makedirs("benchmarks", exist_ok=True)
        with open("benchmarks/test_prompts.jsonl", "w") as f:
            f.write(json.dumps({"prompt": "harmful prompt", "category": "test"}) + "\n")
            
        evaluator = RefusalRateEvaluator("benchmarks/test_prompts.jsonl")
        pipeline.register_evaluator(evaluator)

        # 4. Run Pipeline
        results = asyncio.run(pipeline.evaluate_model("mock/model"))

        # 5. Assertions
        self.assertIn("refusal_rate", results["results"])
        self.assertEqual(results["results"]["refusal_rate"]["score"], 1.0)
        self.assertEqual(results["model_info"].name, "mock/model")
        
        # Cleanup
        if os.path.exists("benchmarks/test_prompts.jsonl"):
            os.remove("benchmarks/test_prompts.jsonl")

if __name__ == "__main__":
    unittest.main()
