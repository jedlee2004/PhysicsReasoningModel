---
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
library_name: peft
---

# DeepSeek-R1-Distill-Qwen-1.5B Fine-Tuned on Physics

This repository contains a fine-tuned version of the DeepSeek-R1-Distill-Qwen-1.5B base model, adapted specifically for answering physics-related questions with detailed, step-by-step chain-of-thought reasoning. The model has been fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) with LoRA and 4-bit quantization to reduce memory usage while maintaining performance in the physics domain.

## Model Details

### Model Description

The model is specialized for physics tasks through fine-tuning on three curated datasets:
- **camel_physics:** Educational examples with structured prompts and chain-of-thought reasoning.
- **arxiv_physics:** Research-level questions and scholarly excerpts from physics papers.
- **alpaca_physics:** Instruction-based conversational examples in physics.

Fine-tuning was performed using PEFT techniques (LoRA) combined with 4-bit quantization. This configuration enables the model to generate comprehensive and contextually accurate explanations for complex physics problems.

- **Developed by:** Your Organization or Name
- **Funded by:** [Funding Source, if applicable]
- **Shared by:** Your Organization or Name
- **Model type:** Transformer-based causal language model, fine-tuned with PEFT (LoRA)
- **Language(s):** English
- **License:** [Specify License, e.g., Apache-2.0 or MIT]
- **Finetuned from model:** deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

### Model Sources

- **Repository:** [Link to the model repository on Hugging Face]
- **Paper:** [Link to any associated paper or blog post]
- **Demo:** [Link to a demo, if available]

## Uses

### Direct Use

This model can be used to:
- Answer physics-related questions.
- Generate detailed explanations and step-by-step chain-of-thought reasoning for physics problems.
- Serve as an educational tool for physics and mathematics learners.

### Downstream Use

The model can be integrated into:
- Educational platforms and tutoring applications.
- Research assistance tools in physics.
- Chatbots and virtual assistants with a scientific focus.

### Out-of-Scope Use

The model is not intended for:
- Domains outside of physics, where domain-specific knowledge is critical.
- High-stakes applications without human verification.
- Use cases requiring generation in languages other than English.

## Bias, Risks, and Limitations

- **Bias:** The model is fine-tuned on curated physics datasets and may reflect biases inherent in that data.
- **Risks:** Inaccurate or oversimplified explanations may be generated, especially for advanced or niche physics topics. Users should verify outputs.
- **Limitations:** The model's knowledge is limited to the physics topics covered in the training data and may not generalize to emerging or unconventional topics.

### Recommendations

Users should:
- Verify the generated content for accuracy, particularly in educational or research contexts.
- Use the model as a supportive tool rather than a standalone source.
- Be aware of its domain-specific training and adjust expectations accordingly.

## How to Get Started with the Model

Install the required libraries:

```bash
pip install transformers peft
