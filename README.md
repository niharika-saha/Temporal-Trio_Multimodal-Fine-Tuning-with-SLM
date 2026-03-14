# Temporal-Trio_Multimodal-Fine-Tuning-with-SLM

## Overview

This project fine-tunes a vision-language model to generate captions for mobile UI screenshots using the **RICO Screen2Words** dataset.

The model learns to generate natural language descriptions of mobile app interfaces from screenshots. This is useful for tasks such as:

- UI accessibility
- Automated documentation
- Interface understanding
- Screen summarization

The base model used is **Salesforce/blip-image-captioning-base**, fine-tuned on the RICO dataset.

---

## Hugging Face Model

**[nikilovesml/temporal_trio-rico-screen2words-blip-caption](https://huggingface.co/nikilovesml/temporal_trio-rico-screen2words-blip-caption)**

This repository contains:

- Fine-tuned BLIP model weights
- Processor configuration
- Inference instructions
- Example usage

---

## Dataset

**RICO Screen2Words**

| Field    | Description                            |
| -------- | -------------------------------------- |
| image    | UI screenshot                          |
| captions | List containing the screen description |

Example caption:

```
"display page of messages and other options"
```

The RICO Screen2Words was selected because the task is image captioning, where the model maps a UI screenshot directly to a short natural-language description. This task requires visual recognition of interface elements and language generation, allowing smaller vision–language models to be effectively fine-tuned on T4 GPUs without heavy reasoning or structured data extraction.

---

## Model Architecture

Base model: **Salesforce/blip-image-captioning-base**

BLIP combines:

- **Vision Transformer (ViT)** for image encoding
- **Transformer decoder** for text generation

```
UI Screenshot
      ↓
Vision Transformer Encoder
      ↓
Multimodal Transformer
      ↓
Text Decoder
      ↓
Generated Caption
```

BLIP was chosen because it is:

- Optimized for image captioning
- Relatively lightweight (~250M parameters)
- Compatible with single-GPU training

---

## Why BLIP Was Selected

| Model    | Result                    |
| -------- | ------------------------- |
| BLIP     | Best caption quality      |
| ViT-GPT2 | Unstable captions         |
| BLIP-2   | Exceeded T4 memory limits |
| CLIP-FLAN-T5 | Poor caption alignment |

BLIP provided the best balance between performance, compute efficiency, and training stability.

---

## Training Setup

- **GPU:** NVIDIA T4 (16GB VRAM)
- **Platform:** Google Colab

The pipeline is designed to run fully within T4 compute limits.

---

## Training Configuration

| Parameter     | Value   | Intention                  |
| ------------- | ------- | ------------------------- |
| Batch size    | 4       | Fits in T4 memory         |
| Epochs        | 3       | Sufficient convergence    |
| Learning rate | 5e-5    | Standard fine-tuning rate |
| FP16          | Enabled | Reduces GPU memory        |
| Optimizer     | AdamW   | Default for transformers  |

- **Training library:** Transformers
- **Training interface:** Trainer

---

## Installation

```bash
pip install transformers datasets accelerate peft bitsandbytes torchvision huggingface_hub
```

---

## Training Pipeline

### Load Dataset

```python
from datasets import load_dataset

dataset = load_dataset("rootsautomation/RICO-Screen2Words")
```

### Load Model

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```

### Preprocessing

```python
def preprocess(example):
    image = example["image"]
    caption = example["captions"][0]

    inputs = processor(
        images=image,
        text=caption,
        padding="max_length",
        truncation=True
    )

    inputs["labels"] = inputs["input_ids"]
    return inputs
```

### Train

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn
)

trainer.train()
```

---

## Evaluation

Example output after training:

```
Ground Truth : display page of messages and other options
Prediction   : display page of messages with other options
```

---

## Inference

### Load Model from Hugging Face

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained(
    "nikilovesml/temporal_trio-rico-screen2words-blip-caption"
)
model = BlipForConditionalGeneration.from_pretrained(
    "nikilovesml/temporal_trio-rico-screen2words-blip-caption"
)
```

### Run Inference

```python
from PIL import Image

image = dataset["test"][0]["image"]

inputs = processor(image, return_tensors="pt")
output = model.generate(**inputs)
caption = processor.decode(output[0], skip_special_tokens=True)

print(caption)
```

---

## Example Predictions

| Ground Truth                               | Prediction                                  |
| ------------------------------------------ | ------------------------------------------- |
| display page of messages and other options | display page of messages with other options |
| create account details for a chat app      | create account page for chat application    |
| page displaying a search bar               | search page of application                  |

---

## Reproducibility

To reproduce the full training pipeline:

1. Clone this repository
2. Install dependencies
3. Run `train.ipynb`

The project was designed to run entirely on **T4 GPU compute**, ensuring accessibility for students and researchers with limited hardware.

---

## Conclusion

This project demonstrates that a lightweight vision-language model can be effectively fine-tuned to understand mobile UI screens. Using BLIP with the RICO Screen2Words dataset provides a practical approach to generating screen descriptions while remaining compatible with modest GPU resources such as the T4.
