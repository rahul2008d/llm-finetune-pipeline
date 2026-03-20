"""Model loading utilities: quantized base models, LoRA adapters, and tokenizers."""
from __future__ import annotations

from typing import Any

import structlog
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

from src.config.training import LoRAConfig as LoRAAppConfig
from src.config.training import ModelConfig, QuantizationConfig

logger = structlog.get_logger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class ModelLoader:
    """Load quantized base models, apply LoRA/DoRA adapters, and configure tokenizers."""

    def load_base_model(
        self,
        config: ModelConfig,
        quant_config: QuantizationConfig,
        *,
        no_cuda: bool = False,
    ) -> PreTrainedModel:
        """Load a causal-LM, optionally with 4-bit quantization.

        Args:
            config: Model identification and loading settings.
            quant_config: BitsAndBytes quantization parameters.
            no_cuda: When True, force CPU and skip quantization.

        Returns:
            Model ready for adapter application.
        """
        device_map = "cpu" if no_cuda else "auto"

        if quant_config.load_in_4bit:
            if not BNB_AVAILABLE:
                raise RuntimeError(
                    "4-bit quantization requires bitsandbytes which is not installed. "
                    "Install with: pip install bitsandbytes (Linux/CUDA only). "
                    "Set quantization.load_in_4bit: false for CPU/MPS training."
                )
            compute_dtype = _DTYPE_MAP[quant_config.bnb_4bit_compute_dtype]
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
            )
        else:
            bnb_config = None

        logger.info(
            "Loading base model",
            model=config.model_name_or_path,
            quant_method=quant_config.method,
            load_in_4bit=quant_config.load_in_4bit,
            device_map=device_map,
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=_DTYPE_MAP[config.torch_dtype],
            attn_implementation=config.attn_implementation,
            use_cache=config.use_cache,
        )

        if quant_config.load_in_4bit:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            model = prepare_model_for_kbit_training(model)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "Base model loaded",
            model=config.model_name_or_path,
            total_params=total_params,
        )

        return model

    def apply_lora(
        self,
        model: PreTrainedModel,
        lora_config: LoRAAppConfig,
    ) -> Any:
        """Wrap a base model with LoRA/DoRA adapters via PEFT.

        Args:
            model: Base model (typically quantized).
            lora_config: LoRA adapter configuration.

        Returns:
            PeftModel with trainable adapter parameters.
        """
        peft_config = LoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias=lora_config.bias,
            task_type=TaskType[lora_config.task_type],
            use_dora=lora_config.use_dora,
            use_rslora=lora_config.use_rslora,
            modules_to_save=lora_config.modules_to_save,
        )

        logger.info(
            "Applying LoRA adapters",
            r=lora_config.r,
            alpha=lora_config.lora_alpha,
            dora=lora_config.use_dora,
            rslora=lora_config.use_rslora,
            targets=lora_config.target_modules,
        )

        peft_model = get_peft_model(model, peft_config)

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        pct = 100.0 * trainable / total if total > 0 else 0.0

        logger.info(
            "LoRA adapters applied",
            trainable_params=trainable,
            total_params=total,
            trainable_pct=round(pct, 4),
        )

        peft_model.print_trainable_parameters()
        return peft_model

    def load_tokenizer(
        self,
        config: ModelConfig,
    ) -> PreTrainedTokenizerBase:
        """Load and configure a tokenizer for causal-LM fine-tuning.

        Args:
            config: Model configuration with tokenizer settings.

        Returns:
            Configured tokenizer with pad token and correct max length.
        """
        logger.info("Loading tokenizer", model=config.model_name_or_path)

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token", pad_token=tokenizer.pad_token)

        tokenizer.padding_side = "right"
        tokenizer.model_max_length = config.max_seq_length

        logger.info(
            "Tokenizer loaded",
            vocab_size=tokenizer.vocab_size,
            max_length=tokenizer.model_max_length,
            padding_side=tokenizer.padding_side,
        )

        return tokenizer


__all__: list[str] = ["ModelLoader"]
