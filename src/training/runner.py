"""QLoRA/DoRA training runner with PEFT integration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import structlog
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TrainingArguments,
)

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
from trl import SFTTrainer

from src.config.settings import TrainingSettings

logger = structlog.get_logger(__name__)


class TrainingRunner:
    """Orchestrate QLoRA/DoRA fine-tuning of causal language models."""

    def __init__(self, settings: TrainingSettings) -> None:
        """Initialize the training runner.

        Args:
            settings: Training configuration settings.
        """
        self.settings = settings
        self.model: PreTrainedModel | None = None
        self.tokenizer: Any = None

    def setup_model(self, use_qlora: bool = True, use_dora: bool = False) -> PreTrainedModel:
        """Load and configure the base model with quantization and LoRA adapters.

        Args:
            use_qlora: Enable 4-bit QLoRA quantization.
            use_dora: Use DoRA instead of standard LoRA.

        Returns:
            Configured PEFT model ready for training.
        """
        logger.info(
            "Setting up model",
            model_id=self.settings.base_model_id,
            qlora=use_qlora,
            dora=use_dora,
        )

        bnb_config = None
        if use_qlora:
            if not BNB_AVAILABLE:
                raise RuntimeError(
                    "QLoRA requires bitsandbytes which is not installed. "
                    "Install with: pip install bitsandbytes (Linux/CUDA only). "
                    "For macOS, set use_qlora: false in your config."
                )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=self.settings.huggingface_token or None,
            trust_remote_code=False,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.base_model_id,
            token=self.settings.huggingface_token or None,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        lora_config = LoraConfig(
            r=self.settings.lora_rank,
            lora_alpha=self.settings.lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_dora=use_dora,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        logger.info("Model setup complete")
        return self.model

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        output_dir: str = "./output",
        callbacks: list[Any] | None = None,
        formatting_func: Callable | None = None,
    ) -> dict[str, Any]:
        """Run the fine-tuning training loop.

        Args:
            train_dataset: Tokenized training dataset.
            eval_dataset: Optional tokenized evaluation dataset.
            output_dir: Directory for saving outputs and checkpoints.
            callbacks: Optional list of training callbacks.
            formatting_func: Optional function to format raw dataset rows into text.

        Returns:
            Training metrics dictionary.

        Raises:
            RuntimeError: If model has not been set up.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call setup_model() before train()")

        # Pre-tokenized datasets need remove_unused_columns=False to keep
        # input_ids/attention_mask/labels.  Raw text datasets need True so the
        # text column is dropped after tokenisation by SFTTrainer.
        has_text_col = "text" in train_dataset.column_names
        has_input_ids = "input_ids" in train_dataset.column_names
        remove_unused = not has_input_ids

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.settings.epochs,
            per_device_train_batch_size=self.settings.batch_size,
            gradient_accumulation_steps=self.settings.gradient_accumulation_steps,
            learning_rate=self.settings.learning_rate,
            weight_decay=self.settings.weight_decay,
            warmup_ratio=self.settings.warmup_ratio,
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            evaluation_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=100 if eval_dataset is not None else None,
            bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            report_to="none",
            remove_unused_columns=remove_unused,
        )

        sft_kwargs: dict[str, Any] = dict(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            max_seq_length=self.settings.max_seq_length,
            callbacks=callbacks or [],
        )
        if formatting_func is not None:
            sft_kwargs["formatting_func"] = formatting_func
        elif "text" in train_dataset.column_names:
            sft_kwargs["dataset_text_field"] = "text"

        trainer = SFTTrainer(**sft_kwargs)

        logger.info("Starting training", epochs=self.settings.epochs)
        result = trainer.train()
        metrics: dict[str, Any] = result.metrics
        logger.info("Training complete", metrics=metrics)

        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved", path=output_dir)

        return metrics

    def save_merged(self, output_dir: str) -> None:
        """Merge LoRA adapters and save the full model.

        Args:
            output_dir: Directory to save the merged model.

        Raises:
            RuntimeError: If model has not been set up.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call setup_model() before save_merged()")

        logger.info("Merging adapters and saving", output_dir=output_dir)
        merged = self.model.merge_and_unload()
        merged.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Merged model saved")


__all__: list[str] = ["TrainingRunner"]
