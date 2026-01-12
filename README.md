# 🛡️ NanoEval

**The Safety Certification for Distilled & Edge-Deployed AI**

[![Status](https://img.shields.io/badge/Status-In--Development-orange)](#)
[![Target](https://img.shields.io/badge/Target-Small--Models--1B--13B-blue)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)

**NanoEval** is one of the first comprehensive safety framework purpose-built for small language models (SLMs). While large-scale models receive massive security scrutiny, the millions of distilled, quantized, and fine-tuned models being deployed to edge devices often lack rigorous safety validation.

NanoEval bridges this gap by providing an automated, high-speed testing suite designed for rapid iteration cycles, enabling developers to catch safety regressions before deployment.

---

## 🎯 Why NanoEval?

### Small Models Are Everywhere, But Safety Testing Isn't

### Safety risks:

*   **Distillation Drift:** Does the student model still refuse harmful requests as effectively as its teacher?
*   **Quantization Noise:** Does compressing to INT4/INT8 amplify latent biases?
*   **Edge Isolation:** Can your model maintain safety offline without cloud-based guardrails?
*   **Fine-Tuning Regression:** Did domain-specific training break the base model's safety alignment?

---

### Our Solution: Purpose-Built Safety Framework for Small Models

### ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Distillation Audit** | Compare Teacher vs. Student models to measure refusal rate preservation. |
| **Quantization Analysis** | Benchmark safety degradation across FP16, INT8, and INT4 levels. |
| **Regression Testing** | Automated "Before vs. After" safety checks for fine-tuned versions. |
| **Edge-Ready Validation** | Simulation of offline, resource-constrained safety performance. |
| **CI/CD Integration** | Fast, lightweight evaluations (<5 min) designed for automated pipelines. |

---

### Target Users

1. **Model Distillation Teams** - Validate distilled models maintain safety properties
2. **Edge AI Companies** - Certify models for deployment on devices
3. **Enterprise ML Teams** - Test fine-tuned models for regressions
4. **Open Source Developers** - Community-driven safety evaluation
5. **Mobile App Developers** - Pre-deployment safety checks for on-device AI
6. **IoT/Embedded AI** - Safety validation for resource-constrained deployment

---

## 🏗️ Architecture

NanoEval is designed to run locally, ensuring your proprietary models and data never leave your infrastructure.

```
NanoEval/
├── core/                # Inference & Orchestration
├── evaluators/          # Specialized Test Engines (Distillation, Bias, PII)
├── loaders/             # Support for HF, GGUF, MLX, and ONNX
└── benchmarks/          # Optimized safety-critical datasets for small models
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MsChenoO/NanoEval.git
cd NanoEval

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install NanoEval and dependencies
pip install -e .

# Verify installation
nanoeval --version
```
---

## 📜 License

Apache License 2.0

```