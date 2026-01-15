import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import sys

# Mock llama_cpp BEFORE importing the loader
mock_llama = MagicMock()
sys.modules["llama_cpp"] = mock_llama

from nanoeval.loaders.llama_cpp_loader import LlamaCppLoader
from nanoeval.core.model_loader import ModelResponse

class TestLlamaCppLoader(unittest.TestCase):
    def setUp(self):
        self.loader = LlamaCppLoader()

    @patch("os.path.exists")
    def test_load(self, mock_exists):
        mock_exists.return_value = True
        mock_instance = MagicMock()
        mock_llama.Llama.return_value = mock_instance
        
        self.loader.load("dummy.gguf")
        
        self.assertEqual(self.loader.model, mock_instance)
        mock_llama.Llama.assert_called_once()

    def test_generate(self):
        mock_instance = MagicMock()
        mock_instance.return_value = {
            "choices": [{"text": "Mocked GGUF response"}]
        }
        self.loader.model = mock_instance
        
        response = self.loader.generate("Hello")
        
        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.text, "Mocked GGUF response")
        mock_instance.assert_called_once()

    def test_quantization_detection(self):
        self.loader._model_path = "/path/to/Llama-3-8B-Q4_K_M.gguf"
        self.assertEqual(self.loader._detect_quantization_from_path(), "Q4_K_M")
        
        self.loader._model_path = "model-q8_0.gguf"
        self.assertEqual(self.loader._detect_quantization_from_path(), "Q8_0")

if __name__ == "__main__":
    unittest.main()
