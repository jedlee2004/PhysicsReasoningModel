
import os
from typing import Any, Dict, Tuple, Union
import torch
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling, 
    set_seed,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TrainerCallback,
    BatchEncoding
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatasetSample:
    context: str
    question: str
    answer: str

@dataclass
class Config:
    # ---Configuration interface for model fine-tuning. ---
    seed: int  # Random seed for reproducibility
    model_id: str  # Model ID (e.g., Hugging Face model repo ID)
    local_model_dir: str  # Local directory to store model files
    dataset_name: str  # Name of the dataset to use
    dataset_dir: str  # Local directory to cache the dataset
    output_dir: str  # Directory to save the trained model
    epochs: int  # Number of training epochs
    batch_size: int  # Batch size per device
    gradient_accumulation: int  # Gradient accumulation steps
    save_steps: int  # Number of steps between saving checkpoints
    save_total: int  # Maximum number of checkpoints to keep
    logging_steps: int  # Number of steps between logging
    learning_rate: float  # Learning rate for training
    warmup_steps: int  # Number of warmup steps for the learning rate scheduler
    max_grad_norm: float  # Maximum gradient norm for gradient clipping

class LoadDatasetsAndModel:
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # should be 14B
    model: AutoModelForCausalLM
    tokenizer: any = None

    def __init__(self):
        self.local_model_dir: str = "model"
        os.makedirs(self.local_model_dir, exist_ok=True)

    def download_and_save_model(self):
        config_path: str = os.path.join(self.local_model_dir, "config.json")
        if not os.path.exists(config_path):
            print("Local model not found. Downloading model and tokenizer from Hugging Face Hub...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            os.makedirs(self.local_model_dir, exist_ok=True)
            tokenizer.save_pretrained(self.local_model_dir)

            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    # llm_int8_threshold=6.0,
                    # llm_int8_enable_fp32_cpu_offload=True
                )            
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )

            except Exception as e:
                print(f"Quantization failed: {e}. Loading model without quantization.")
                # Fallback: Load the model without quantization
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    torch_dtype=torch.float16
                )

            model.save_pretrained(self.local_model_dir)
            self.model = model

            # Free memory of models downloaded and saved
            torch.cuda.empty_cache()
            print(f"Model and tokenizer saved to '{self.local_model_dir}'")
        else:
            print(f"Model already exists in '{self.local_model_dir}'.")

    def load_deepseek_model(self, local_model_dir: str = "model") -> tuple[Union[Any, AutoModelForCausalLM], Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
        # Check if the directory exists
        if not os.path.exists(local_model_dir):
            raise FileNotFoundError(
                f"Model directory {local_model_dir} does not exist. "
                f"Please ensure the model is downloaded and placed in this directory."
                )

        # Check if config.json exists in the directory
        config_path = os.path.join(local_model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {local_model_dir}. Ensure the model files are correct.")

        # Load the model from the local directory
        try:
            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

            # Load the model
            model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                local_model_dir,
                quantization_config=BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                ),
                device_map={"": 0} if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16
            )

            # Prepare the model for k-bit training
            model = prepare_model_for_kbit_training(model)
            print("DeepSeek model loaded and prepared for k-bit training.")

            lora_config = LoraConfig(
                r=8,                   # LoRA rank (adjust as needed)
                lora_alpha=16,         # Scaling factor for LoRA weights
                target_modules=["q_proj", "v_proj"],  # Target linear layers (adjust if needed)
                lora_dropout=0.1,      # Dropout rate for LoRA layers
                bias="none",           # Bias handling
                task_type="CAUSAL_LM"   # Task type for causal language modeling 
            )

            peftModel = get_peft_model(model, lora_config)

            return peftModel, tokenizer
        except ValueError as e:
            raise ValueError(f"Failed to load the DeepSeek model from {local_model_dir}: {e}")

class Preprocessing:
    alpaca_physics: Dataset = None
    arxiv_physics: Dataset = None
    camel_physics: Dataset = None
    tokenizer: PreTrainedTokenizer = None

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        try:
            self.camel_physics = load_from_disk("datasets/camel_physics")
            self.arxiv_physics = load_from_disk("datasets/arxiv_physics")
            self.alpaca_physics = load_from_disk("datasets/alpaca_physics")
            if self.alpaca_physics is None or self.arxiv_physics is None or self.camel_physics is None:
                raise ValueError(f"Failed to load datasets from disk")
        except Exception as e:
            error_message: str = f"Error loading dataset: {e}"
            print(error_message)
            raise ValueError(error_message)

    def format_prompt_camel(self, sample: Dict[str, str]) -> Dict[str, str]:
        prompt_template = """Below is an instruction that describes a task, paired with input that provides further context.
        Write a comprehensive response that fully addresses the request.
        Before answering, think carefully about the question and create a step-by-step chain of thought to ensure accuracy.

        ### Instruction:
        You are a physicist and mathematician with advanced knowledge in this domain.
        Your role is: {role_1}.
        The question pertains to the sub-topic: {sub_topic}.

        ### Question:
        {topic}

        ### Chain-of-Thought:
        <think>
        {message_1}
        </think>

        ### Final Answer:
        {message_2}"""
        
        try:
            sample["prompt"] = prompt_template.format(
                topic=sample.get("topic;", "No topic provided"),
                role_1=sample.get("role_1", "No role provided"),
                sub_topic=sample.get("sub_topic", "No sub-topic provided"),
                message_1=sample.get("message_1", "No chain-of-thought provided"),
                message_2=sample.get("message_2", "No final answer provided")
            )
            return sample
        except Exception as e:
            error_message = f"Error templating sample: {e}"
            print(error_message)
            raise ValueError(error_message)
        
    def format_prompt_alpaca(self, sample: Dict[str, str]) -> Dict[str, str]:
        prompt_template = """Below is an instruction that describes a task, paired with input that provides further context.
        Write a comprehensive response that fully addresses the request.
        Before answering, think carefully about the question and create a step-by-step chain of thought to ensure a logical and accurate response.

        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Chain-of-Thought:
        <think>
        [Insert your reasoning here]
        </think>

        ### Final Answer:
        {output}"""

        try:
            sample["prompt"] = prompt_template.format(
                instruction=sample.get("instruction", "No instruction provided"),
                input=sample.get("input", "No input provided"),
                output=sample.get("output", "No output provided")
            )
            return sample
        except Exception as e:
            error_message = f"Error templating sample: {e}"
            print(error_message)
            raise ValueError(error_message)

    def format_prompt_arxiv(self, sample: Dict[str, str]) -> Dict[str, str]:
        prompt_template = """Below is an instruction that describes a task, paired with input that provides further context.
        Write a detailed response that fully addresses the question.
        Before answering, carefully analyze the question and develop a step-by-step chain of thought to arrive at a logical and accurate response.

        ### Question:
        {question}

        ### Chain-of-Thought:
        <think>
        [Insert your reasoning here]
        </think>

        ### Final Answer:
        {answer}"""
        
        try:
            sample["prompt"] = prompt_template.format(
                question=sample.get("question", "No question provided"),
                answer=sample.get("answer", "No answer provided")
            )
            return sample
        except Exception as e:
            error_message = f"Error templating sample: {e}"
            print(error_message)
            raise ValueError(error_message)            

    def tokenize_prompt(self, example) -> BatchEncoding:
        # Tokenize the prompt field
        tokenized: BatchEncoding = self.tokenizer(example["prompt"], truncation=True, padding="max_length", max_length=1024)
        return tokenized

    def preprocess_all_datasets(self) -> DatasetDict:
        preprocessed_path: str = "datasets/preprocessed"
        if os.path.exists(preprocessed_path):
            try:
                # Load each dataset folder
                camel_path = os.path.join(preprocessed_path, "camel")
                arxiv_path = os.path.join(preprocessed_path, "arxiv")
                alpaca_path = os.path.join(preprocessed_path, "alpaca")
                
                camel_obj = DatasetDict.load_from_disk(camel_path)
                arxiv_obj = DatasetDict.load_from_disk(arxiv_path)
                alpaca_obj = DatasetDict.load_from_disk(alpaca_path)
                
                # If the loaded object is a DatasetDict with a "train" key, extract it;
                # otherwise, assume the object is already a plain Dataset.
                # camel_ds = camel_obj["train"] if isinstance(camel_obj, DatasetDict) and "train" in camel_obj else camel_obj
                # arxiv_ds = arxiv_obj["train"] if isinstance(arxiv_obj, DatasetDict) and "train" in arxiv_obj else arxiv_obj
                # alpaca_ds = alpaca_obj["train"] if isinstance(alpaca_obj, DatasetDict) and "train" in alpaca_obj else alpaca_obj
                camel_ds: Dataset = None
                arxiv_ds: Dataset = None
                alpaca_ds: Dataset = None

                if isinstance(camel_obj, DatasetDict) and "train" in camel_obj:
                    camel_ds = camel_obj["train"]
                else:
                    camel_ds = camel_obj

                if isinstance(arxiv_obj, DatasetDict) and "train" in arxiv_obj:
                    arxiv_ds = arxiv_obj["train"]
                else:
                    arxiv_ds = arxiv_obj

                if isinstance(alpaca_obj, DatasetDict) and "train" in alpaca_obj:
                    alpaca_ds = alpaca_obj["train"]
                else:
                    alpaca_ds = alpaca_obj
                
                combined_train = concatenate_datasets([camel_ds, arxiv_ds, alpaca_ds])
                print("Loaded preprocessed datasets from disk.")
                return DatasetDict({"train": combined_train})
            except Exception as e:
                raise ValueError(f"Error loading preprocessed datasets: {e}")

        # Else, process and save the datasets if no preprocessed folder exists.
        try:
            camel_ds = self.camel_physics.map(lambda x: self.tokenize_prompt(self.format_prompt_camel(x)))
            arxiv_ds = self.arxiv_physics.map(lambda x: self.tokenize_prompt(self.format_prompt_arxiv(x)))
            alpaca_ds = self.alpaca_physics.map(lambda x: self.tokenize_prompt(self.format_prompt_alpaca(x)))

            combined_train = concatenate_datasets([camel_ds["train"], arxiv_ds["train"], alpaca_ds["train"]])

            DatasetDict({"train": camel_ds}).save_to_disk(os.path.join(preprocessed_path, "camel"))
            DatasetDict({"train": arxiv_ds}).save_to_disk(os.path.join(preprocessed_path, "arxiv"))
            DatasetDict({"train": alpaca_ds}).save_to_disk(os.path.join(preprocessed_path, "alpaca"))

            print("Preprocessed datasets saved to disk.")
            return DatasetDict({"train": combined_train})
        except Exception as e:
            raise ValueError(f"Error tokenizing and combining datasets: {e}")

class CustomProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Called when logs are reported
        print(f"Step {state.global_step}: {logs}")

    def on_step_end(self, args, state, control, **kwargs):
        # Called at the end of each training step
        # You can add custom logging or update external progress trackers here.
        print(f"Completed step {state.global_step}")

class TrainModel:
    device_map = "cuda"

    def __init__(self, config: Config, tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> None:
        self.config: Config = config
        self.model_id: str = config.model_id
        self.local_model_dir: str = config.local_model_dir
        self.tokenizer: AutoTokenizer = tokenizer
        self.model: AutoModelForCausalLM = model
        
    def compute_metrics(self, eval_prediction: Tuple[Any, Any]) -> Dict[str, float]:
        logits, labels = eval_prediction
        # Compute predictions by taking the argmax over the logits.
        predictions = np.argmax(logits, axis=-1)
        
        # Create a mask to ignore positions with label == -100 (commonly used for masked tokens).
        mask = labels != -100
        # Count correct predictions only where mask is True.
        correct = (predictions == labels) * mask
        accuracy = float(np.sum(correct)) / float(np.sum(mask)) if np.sum(mask) > 0 else 0.0
        return {"accuracy": accuracy}
    
    def train(self: 'TrainModel', tokenized_datasets:DatasetDict) -> None:
        # Step 6: Create a data collator for causal language modeling.
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False, seed=42)

        # Step 7 Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total,
            logging_dir="./logs",
            logging_strategy="steps",
            logging_steps=self.config.logging_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            fp16=torch.cuda.is_available(),
            max_grad_norm=self.config.max_grad_norm,
            load_best_model_at_end=False,
            push_to_hub=False,
            # deepspeed="ds_config.json",
            disable_tqdm=False,
        )
    
        # Step 8 Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            # eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else None,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[CustomProgressCallback()]
        )

        # Step 9 Start training
        trainer.train()

        # Step 10 Save final model and tokenizer
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"Training complete and model saved to {self.config.output_dir}")

    def run(self, tokenized_datasets: DatasetDict) -> None:
        torch.cuda.empty_cache()
        self.train(tokenized_datasets)


if __name__ == "__main__":
    model: AutoModelForCausalLM = None
    tokenizer: PreTrainedTokenizer = None
    tokenized_datasets: DatasetDict = None

    # Check if CUDA is available
    device_map = "cuda"
    print(f"Using device: {device_map}")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 1. Loading model and tokenizer
    try:
        # Download and save the model if not already present locally.
        loadData = LoadDatasetsAndModel()
        loadData.download_and_save_model()

        # Load the tokenizer and model from the local directory.
        model, tokenizer = loadData.load_deepseek_model()
        print(f"Model and tokenizer loaded from '{loadData.local_model_dir}'.")
    except Exception as e:
        print(f"Error loading model with load_deepseek_model: {e}")

    # 2. Preprocess Data for fine tuning
    try:
        # tokenizer.pad_token = tokenizer.eos_token
        if tokenizer is None:
            raise ValueError("Tokenizer was not initialized. Check model loading logic.")
        preprocessing = Preprocessing(tokenizer)
        tokenized_datasets = preprocessing.preprocess_all_datasets()
        # if tokenized_datasets is None or "train" not in tokenized_datasets or "validation" not in tokenized_datasets:
        #     raise ValueError("Preprocessed datasets must contain 'train' and 'validation' splits.")
    except Exception as e:
        errorMessage: str = f"Error preprocessing datasets for fine tuning model: {e}"
        print(errorMessage)
        raise ValueError(errorMessage)

    # 3. Fine tune the model / Train
    try:
        config: Config = Config(
            seed=42,
            model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            local_model_dir="./model",
            dataset_name="combined",
            dataset_dir="./datasets",
            output_dir="./output",
            epochs=3,
            batch_size=1,
            gradient_accumulation=1,
            save_steps=100,
            save_total=3,
            logging_steps=50,
            learning_rate=2e-4,
            warmup_steps=100,
            max_grad_norm=1.0
        )
        set_seed(config.seed)
        model.gradient_checkpointing_enable()
        train_model = TrainModel(config, tokenizer, model)
        train_model.run(tokenized_datasets)
    except Exception as e:
        print(f"Error training model: ${e}")
        raise ValueError(e)