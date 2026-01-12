import unittest
from unittest.mock import MagicMock, patch
import torch
from nanoeval.loaders.huggingface_loader import HuggingFaceLoader
from nanoeval.core.model_loader import ModelResponse, ModelInfo

class TestHuggingFaceLoader(unittest.TestCase):
    def setUp(self):
        self.loader = HuggingFaceLoader()

    @patch("nanoeval.loaders.huggingface_loader.AutoTokenizer")
    @patch("nanoeval.loaders.huggingface_loader.AutoModelForCausalLM")
    def test_load(self, mock_model_class, mock_tokenizer_class):
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        model = self.loader.load("mock/model")

        self.assertEqual(model, mock_model)
        self.assertEqual(self.loader.model, mock_model)
        self.assertEqual(self.loader.tokenizer, mock_tokenizer)
        mock_model_class.from_pretrained.assert_called_once()

    @patch("nanoeval.loaders.huggingface_loader.AutoTokenizer")
    @patch("nanoeval.loaders.huggingface_loader.AutoModelForCausalLM")
    def test_generate(self, mock_model_class, mock_tokenizer_class):
        # Setup mocks
        mock_model = MagicMock()
        self.loader.model = mock_model
        self.loader.tokenizer = MagicMock()
        self.loader.model.device = "cpu"
        
        # Mock tokenizer encoding
        mock_inputs = MagicMock()
        mock_inputs.input_ids = torch.tensor([[1, 2, 3]])
        mock_inputs.to.return_value = mock_inputs
        self.loader.tokenizer.return_value = mock_inputs
        
        # Mock model generation
        mock_output = MagicMock()
        mock_output.sequences = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_output
        
        # Mock tokenizer decoding
        self.loader.tokenizer.decode.return_value = "Mocked response"

        response = self.loader.generate("Hello world")

        self.assertIsInstance(response, ModelResponse)
        self.assertEqual(response.text, "Mocked response")
        self.assertEqual(response.tokens, [4, 5])
        mock_model.generate.assert_called_once()

    def test_get_info_error(self):
        # Should raise error if model is not loaded
        with self.assertRaises(RuntimeError):
            self.loader.get_info()

if __name__ == "__main__":
    unittest.main()
