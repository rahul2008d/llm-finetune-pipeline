"""Unit tests for training.model_loader – ModelLoader class."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from peft import TaskType

from src.config.training import LoRAConfig, ModelConfig, QuantizationConfig
from src.training.model_loader import ModelLoader, _DTYPE_MAP


# ── fixtures ────────────────────────────────────────────────────


@pytest.fixture()
def loader() -> ModelLoader:
    return ModelLoader()


@pytest.fixture()
def model_config() -> ModelConfig:
    return ModelConfig(model_name_or_path="meta-llama/Llama-3.1-8B-Instruct")


@pytest.fixture()
def quant_config() -> QuantizationConfig:
    return QuantizationConfig()


@pytest.fixture()
def lora_config() -> LoRAConfig:
    return LoRAConfig()


@pytest.fixture()
def dora_config() -> LoRAConfig:
    return LoRAConfig(use_dora=True)


@pytest.fixture()
def rslora_config() -> LoRAConfig:
    return LoRAConfig(use_rslora=True)


def _fake_model(total_params: int = 1000, trainable: int = 1000) -> MagicMock:
    """Build a mock model with controllable parameter counts."""
    param_trainable = MagicMock()
    param_trainable.numel.return_value = trainable
    param_trainable.requires_grad = True

    param_frozen = MagicMock()
    param_frozen.numel.return_value = total_params - trainable
    param_frozen.requires_grad = False

    model = MagicMock()
    if trainable == total_params:
        model.parameters.return_value = [param_trainable]
    else:
        model.parameters.return_value = [param_trainable, param_frozen]
    return model


# ═══════════════════════════════════════════════════════════════
# DTYPE_MAP
# ═══════════════════════════════════════════════════════════════


class TestDtypeMap:
    def test_float16_maps_correctly(self) -> None:
        assert _DTYPE_MAP["float16"] is torch.float16

    def test_bfloat16_maps_correctly(self) -> None:
        assert _DTYPE_MAP["bfloat16"] is torch.bfloat16

    def test_float32_maps_correctly(self) -> None:
        assert _DTYPE_MAP["float32"] is torch.float32

    def test_all_keys_present(self) -> None:
        assert set(_DTYPE_MAP) == {"float16", "bfloat16", "float32"}


# ═══════════════════════════════════════════════════════════════
# load_base_model
# ═══════════════════════════════════════════════════════════════


class TestLoadBaseModel:
    @patch("src.training.model_loader.prepare_model_for_kbit_training")
    @patch("src.training.model_loader.AutoModelForCausalLM")
    @patch("src.training.model_loader.BitsAndBytesConfig")
    def test_constructs_bnb_config_defaults(
        self,
        mock_bnb_cls: MagicMock,
        mock_auto: MagicMock,
        mock_prep: MagicMock,
        loader: ModelLoader,
        model_config: ModelConfig,
        quant_config: QuantizationConfig,
    ) -> None:
        mock_model = _fake_model()
        mock_auto.from_pretrained.return_value = mock_model
        mock_prep.return_value = mock_model

        loader.load_base_model(model_config, quant_config)

        mock_bnb_cls.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    @patch("src.training.model_loader.prepare_model_for_kbit_training")
    @patch("src.training.model_loader.AutoModelForCausalLM")
    @patch("src.training.model_loader.BitsAndBytesConfig")
    def test_calls_from_pretrained_with_correct_args(
        self,
        mock_bnb_cls: MagicMock,
        mock_auto: MagicMock,
        mock_prep: MagicMock,
        loader: ModelLoader,
        model_config: ModelConfig,
        quant_config: QuantizationConfig,
    ) -> None:
        mock_model = _fake_model()
        mock_auto.from_pretrained.return_value = mock_model
        mock_prep.return_value = mock_model
        bnb_instance = mock_bnb_cls.return_value

        loader.load_base_model(model_config, quant_config)

        mock_auto.from_pretrained.assert_called_once_with(
            "meta-llama/Llama-3.1-8B-Instruct",
            quantization_config=bnb_instance,
            device_map="auto",
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )

    @patch("src.training.model_loader.prepare_model_for_kbit_training")
    @patch("src.training.model_loader.AutoModelForCausalLM")
    @patch("src.training.model_loader.BitsAndBytesConfig")
    def test_enables_gradient_checkpointing(
        self,
        mock_bnb_cls: MagicMock,
        mock_auto: MagicMock,
        mock_prep: MagicMock,
        loader: ModelLoader,
        model_config: ModelConfig,
        quant_config: QuantizationConfig,
    ) -> None:
        mock_model = _fake_model()
        mock_auto.from_pretrained.return_value = mock_model
        mock_prep.return_value = mock_model

        loader.load_base_model(model_config, quant_config)

        mock_model.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    @patch("src.training.model_loader.prepare_model_for_kbit_training")
    @patch("src.training.model_loader.AutoModelForCausalLM")
    @patch("src.training.model_loader.BitsAndBytesConfig")
    def test_calls_prepare_model_for_kbit_training(
        self,
        mock_bnb_cls: MagicMock,
        mock_auto: MagicMock,
        mock_prep: MagicMock,
        loader: ModelLoader,
        model_config: ModelConfig,
        quant_config: QuantizationConfig,
    ) -> None:
        mock_model = _fake_model()
        mock_auto.from_pretrained.return_value = mock_model
        mock_prep.return_value = mock_model

        loader.load_base_model(model_config, quant_config)

        mock_prep.assert_called_once_with(mock_model)

    @patch("src.training.model_loader.prepare_model_for_kbit_training")
    @patch("src.training.model_loader.AutoModelForCausalLM")
    @patch("src.training.model_loader.BitsAndBytesConfig")
    def test_returns_prepared_model(
        self,
        mock_bnb_cls: MagicMock,
        mock_auto: MagicMock,
        mock_prep: MagicMock,
        loader: ModelLoader,
        model_config: ModelConfig,
        quant_config: QuantizationConfig,
    ) -> None:
        raw_model = _fake_model()
        prepared_model = _fake_model()
        mock_auto.from_pretrained.return_value = raw_model
        mock_prep.return_value = prepared_model

        result = loader.load_base_model(model_config, quant_config)

        assert result is prepared_model

    @patch("src.training.model_loader.prepare_model_for_kbit_training")
    @patch("src.training.model_loader.AutoModelForCausalLM")
    @patch("src.training.model_loader.BitsAndBytesConfig")
    def test_fp4_quant_type(
        self,
        mock_bnb_cls: MagicMock,
        mock_auto: MagicMock,
        mock_prep: MagicMock,
        loader: ModelLoader,
        model_config: ModelConfig,
    ) -> None:
        qc = QuantizationConfig(bnb_4bit_quant_type="fp4")
        mock_model = _fake_model()
        mock_auto.from_pretrained.return_value = mock_model
        mock_prep.return_value = mock_model

        loader.load_base_model(model_config, qc)

        _, kwargs = mock_bnb_cls.call_args
        assert kwargs["bnb_4bit_quant_type"] == "fp4"

    @patch("src.training.model_loader.prepare_model_for_kbit_training")
    @patch("src.training.model_loader.AutoModelForCausalLM")
    @patch("src.training.model_loader.BitsAndBytesConfig")
    def test_float16_compute_dtype(
        self,
        mock_bnb_cls: MagicMock,
        mock_auto: MagicMock,
        mock_prep: MagicMock,
        loader: ModelLoader,
        model_config: ModelConfig,
    ) -> None:
        qc = QuantizationConfig(bnb_4bit_compute_dtype="float16")
        mock_model = _fake_model()
        mock_auto.from_pretrained.return_value = mock_model
        mock_prep.return_value = mock_model

        loader.load_base_model(model_config, qc)

        _, kwargs = mock_bnb_cls.call_args
        assert kwargs["bnb_4bit_compute_dtype"] is torch.float16

    @patch("src.training.model_loader.prepare_model_for_kbit_training")
    @patch("src.training.model_loader.AutoModelForCausalLM")
    @patch("src.training.model_loader.BitsAndBytesConfig")
    def test_trust_remote_code_forwarded(
        self,
        mock_bnb_cls: MagicMock,
        mock_auto: MagicMock,
        mock_prep: MagicMock,
        loader: ModelLoader,
        quant_config: QuantizationConfig,
    ) -> None:
        mc = ModelConfig(
            model_name_or_path="custom/model",
            trust_remote_code=True,
        )
        mock_model = _fake_model()
        mock_auto.from_pretrained.return_value = mock_model
        mock_prep.return_value = mock_model

        loader.load_base_model(mc, quant_config)

        _, kwargs = mock_auto.from_pretrained.call_args
        assert kwargs["trust_remote_code"] is True

    @patch("src.training.model_loader.prepare_model_for_kbit_training")
    @patch("src.training.model_loader.AutoModelForCausalLM")
    @patch("src.training.model_loader.BitsAndBytesConfig")
    def test_sdpa_attn_implementation(
        self,
        mock_bnb_cls: MagicMock,
        mock_auto: MagicMock,
        mock_prep: MagicMock,
        loader: ModelLoader,
        quant_config: QuantizationConfig,
    ) -> None:
        mc = ModelConfig(
            model_name_or_path="x/y",
            attn_implementation="sdpa",
        )
        mock_model = _fake_model()
        mock_auto.from_pretrained.return_value = mock_model
        mock_prep.return_value = mock_model

        loader.load_base_model(mc, quant_config)

        _, kwargs = mock_auto.from_pretrained.call_args
        assert kwargs["attn_implementation"] == "sdpa"

    @patch("src.training.model_loader.prepare_model_for_kbit_training")
    @patch("src.training.model_loader.AutoModelForCausalLM")
    @patch("src.training.model_loader.BitsAndBytesConfig")
    def test_double_quant_disabled(
        self,
        mock_bnb_cls: MagicMock,
        mock_auto: MagicMock,
        mock_prep: MagicMock,
        loader: ModelLoader,
        model_config: ModelConfig,
    ) -> None:
        qc = QuantizationConfig(bnb_4bit_use_double_quant=False)
        mock_model = _fake_model()
        mock_auto.from_pretrained.return_value = mock_model
        mock_prep.return_value = mock_model

        loader.load_base_model(model_config, qc)

        _, kwargs = mock_bnb_cls.call_args
        assert kwargs["bnb_4bit_use_double_quant"] is False


# ═══════════════════════════════════════════════════════════════
# apply_lora
# ═══════════════════════════════════════════════════════════════


class TestApplyLora:
    @patch("src.training.model_loader.get_peft_model")
    def test_builds_lora_config_defaults(
        self,
        mock_get_peft: MagicMock,
        loader: ModelLoader,
        lora_config: LoRAConfig,
    ) -> None:
        base = _fake_model(total_params=1000, trainable=100)
        peft_model = _fake_model(total_params=1000, trainable=100)
        peft_model.print_trainable_parameters = MagicMock()
        mock_get_peft.return_value = peft_model

        loader.apply_lora(base, lora_config)

        peft_cfg = mock_get_peft.call_args[0][1]
        assert peft_cfg.r == 64
        assert peft_cfg.lora_alpha == 128
        assert peft_cfg.lora_dropout == pytest.approx(0.05)
        assert peft_cfg.bias == "none"
        assert peft_cfg.use_dora is False
        assert peft_cfg.use_rslora is False

    @patch("src.training.model_loader.get_peft_model")
    def test_passes_target_modules(
        self,
        mock_get_peft: MagicMock,
        loader: ModelLoader,
        lora_config: LoRAConfig,
    ) -> None:
        base = _fake_model()
        peft_model = _fake_model()
        peft_model.print_trainable_parameters = MagicMock()
        mock_get_peft.return_value = peft_model

        loader.apply_lora(base, lora_config)

        peft_cfg = mock_get_peft.call_args[0][1]
        assert set(peft_cfg.target_modules) == {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        }

    @patch("src.training.model_loader.get_peft_model")
    def test_dora_enabled(
        self,
        mock_get_peft: MagicMock,
        loader: ModelLoader,
        dora_config: LoRAConfig,
    ) -> None:
        base = _fake_model()
        peft_model = _fake_model()
        peft_model.print_trainable_parameters = MagicMock()
        mock_get_peft.return_value = peft_model

        loader.apply_lora(base, dora_config)

        peft_cfg = mock_get_peft.call_args[0][1]
        assert peft_cfg.use_dora is True

    @patch("src.training.model_loader.get_peft_model")
    def test_rslora_enabled(
        self,
        mock_get_peft: MagicMock,
        loader: ModelLoader,
        rslora_config: LoRAConfig,
    ) -> None:
        base = _fake_model()
        peft_model = _fake_model()
        peft_model.print_trainable_parameters = MagicMock()
        mock_get_peft.return_value = peft_model

        loader.apply_lora(base, rslora_config)

        peft_cfg = mock_get_peft.call_args[0][1]
        assert peft_cfg.use_rslora is True

    @patch("src.training.model_loader.get_peft_model")
    def test_task_type_set_to_causal_lm(
        self,
        mock_get_peft: MagicMock,
        loader: ModelLoader,
        lora_config: LoRAConfig,
    ) -> None:
        base = _fake_model()
        peft_model = _fake_model()
        peft_model.print_trainable_parameters = MagicMock()
        mock_get_peft.return_value = peft_model

        loader.apply_lora(base, lora_config)

        peft_cfg = mock_get_peft.call_args[0][1]
        assert peft_cfg.task_type is TaskType.CAUSAL_LM

    @patch("src.training.model_loader.get_peft_model")
    def test_returns_peft_model(
        self,
        mock_get_peft: MagicMock,
        loader: ModelLoader,
        lora_config: LoRAConfig,
    ) -> None:
        base = _fake_model()
        peft_model = _fake_model()
        peft_model.print_trainable_parameters = MagicMock()
        mock_get_peft.return_value = peft_model

        result = loader.apply_lora(base, lora_config)

        assert result is peft_model

    @patch("src.training.model_loader.get_peft_model")
    def test_calls_print_trainable_parameters(
        self,
        mock_get_peft: MagicMock,
        loader: ModelLoader,
        lora_config: LoRAConfig,
    ) -> None:
        base = _fake_model()
        peft_model = _fake_model()
        peft_model.print_trainable_parameters = MagicMock()
        mock_get_peft.return_value = peft_model

        loader.apply_lora(base, lora_config)

        peft_model.print_trainable_parameters.assert_called_once()

    @patch("src.training.model_loader.get_peft_model")
    def test_custom_rank_and_alpha(
        self,
        mock_get_peft: MagicMock,
        loader: ModelLoader,
    ) -> None:
        cfg = LoRAConfig(r=16, lora_alpha=32)
        base = _fake_model()
        peft_model = _fake_model()
        peft_model.print_trainable_parameters = MagicMock()
        mock_get_peft.return_value = peft_model

        loader.apply_lora(base, cfg)

        peft_cfg = mock_get_peft.call_args[0][1]
        assert peft_cfg.r == 16
        assert peft_cfg.lora_alpha == 32

    @patch("src.training.model_loader.get_peft_model")
    def test_modules_to_save_forwarded(
        self,
        mock_get_peft: MagicMock,
        loader: ModelLoader,
    ) -> None:
        cfg = LoRAConfig(modules_to_save=["embed_tokens", "lm_head"])
        base = _fake_model()
        peft_model = _fake_model()
        peft_model.print_trainable_parameters = MagicMock()
        mock_get_peft.return_value = peft_model

        loader.apply_lora(base, cfg)

        peft_cfg = mock_get_peft.call_args[0][1]
        assert peft_cfg.modules_to_save == ["embed_tokens", "lm_head"]

    @patch("src.training.model_loader.get_peft_model")
    def test_custom_target_modules(
        self,
        mock_get_peft: MagicMock,
        loader: ModelLoader,
    ) -> None:
        cfg = LoRAConfig(target_modules=["q_proj", "v_proj"])
        base = _fake_model()
        peft_model = _fake_model()
        peft_model.print_trainable_parameters = MagicMock()
        mock_get_peft.return_value = peft_model

        loader.apply_lora(base, cfg)

        peft_cfg = mock_get_peft.call_args[0][1]
        assert set(peft_cfg.target_modules) == {"q_proj", "v_proj"}

    @patch("src.training.model_loader.get_peft_model")
    def test_lora_dropout_forwarded(
        self,
        mock_get_peft: MagicMock,
        loader: ModelLoader,
    ) -> None:
        cfg = LoRAConfig(lora_dropout=0.1)
        base = _fake_model()
        peft_model = _fake_model()
        peft_model.print_trainable_parameters = MagicMock()
        mock_get_peft.return_value = peft_model

        loader.apply_lora(base, cfg)

        peft_cfg = mock_get_peft.call_args[0][1]
        assert peft_cfg.lora_dropout == pytest.approx(0.1)


# ═══════════════════════════════════════════════════════════════
# load_tokenizer
# ═══════════════════════════════════════════════════════════════


class TestLoadTokenizer:
    @patch("src.training.model_loader.AutoTokenizer")
    def test_calls_from_pretrained(
        self,
        mock_tok_cls: MagicMock,
        loader: ModelLoader,
        model_config: ModelConfig,
    ) -> None:
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok_cls.from_pretrained.return_value = mock_tok

        loader.load_tokenizer(model_config)

        mock_tok_cls.from_pretrained.assert_called_once_with(
            "meta-llama/Llama-3.1-8B-Instruct",
            trust_remote_code=False,
        )

    @patch("src.training.model_loader.AutoTokenizer")
    def test_sets_pad_token_when_none(
        self,
        mock_tok_cls: MagicMock,
        loader: ModelLoader,
        model_config: ModelConfig,
    ) -> None:
        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "</s>"
        mock_tok_cls.from_pretrained.return_value = mock_tok

        loader.load_tokenizer(model_config)

        assert mock_tok.pad_token == "</s>"

    @patch("src.training.model_loader.AutoTokenizer")
    def test_preserves_existing_pad_token(
        self,
        mock_tok_cls: MagicMock,
        loader: ModelLoader,
        model_config: ModelConfig,
    ) -> None:
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok_cls.from_pretrained.return_value = mock_tok

        loader.load_tokenizer(model_config)

        assert mock_tok.pad_token == "<pad>"

    @patch("src.training.model_loader.AutoTokenizer")
    def test_sets_padding_side_right(
        self,
        mock_tok_cls: MagicMock,
        loader: ModelLoader,
        model_config: ModelConfig,
    ) -> None:
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok_cls.from_pretrained.return_value = mock_tok

        loader.load_tokenizer(model_config)

        assert mock_tok.padding_side == "right"

    @patch("src.training.model_loader.AutoTokenizer")
    def test_sets_model_max_length(
        self,
        mock_tok_cls: MagicMock,
        loader: ModelLoader,
        model_config: ModelConfig,
    ) -> None:
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok_cls.from_pretrained.return_value = mock_tok

        loader.load_tokenizer(model_config)

        assert mock_tok.model_max_length == 4096

    @patch("src.training.model_loader.AutoTokenizer")
    def test_custom_max_seq_length(
        self,
        mock_tok_cls: MagicMock,
        loader: ModelLoader,
    ) -> None:
        mc = ModelConfig(model_name_or_path="x/y", max_seq_length=2048)
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok_cls.from_pretrained.return_value = mock_tok

        loader.load_tokenizer(mc)

        assert mock_tok.model_max_length == 2048

    @patch("src.training.model_loader.AutoTokenizer")
    def test_returns_tokenizer(
        self,
        mock_tok_cls: MagicMock,
        loader: ModelLoader,
        model_config: ModelConfig,
    ) -> None:
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok_cls.from_pretrained.return_value = mock_tok

        result = loader.load_tokenizer(model_config)

        assert result is mock_tok

    @patch("src.training.model_loader.AutoTokenizer")
    def test_trust_remote_code_forwarded(
        self,
        mock_tok_cls: MagicMock,
        loader: ModelLoader,
    ) -> None:
        mc = ModelConfig(
            model_name_or_path="custom/model",
            trust_remote_code=True,
        )
        mock_tok = MagicMock()
        mock_tok.pad_token = "<pad>"
        mock_tok_cls.from_pretrained.return_value = mock_tok

        loader.load_tokenizer(mc)

        _, kwargs = mock_tok_cls.from_pretrained.call_args
        assert kwargs["trust_remote_code"] is True


# ═══════════════════════════════════════════════════════════════
# integration-style: full pipeline
# ═══════════════════════════════════════════════════════════════


class TestFullPipeline:
    @patch("src.training.model_loader.AutoTokenizer")
    @patch("src.training.model_loader.get_peft_model")
    @patch("src.training.model_loader.prepare_model_for_kbit_training")
    @patch("src.training.model_loader.AutoModelForCausalLM")
    @patch("src.training.model_loader.BitsAndBytesConfig")
    def test_load_apply_lora_load_tokenizer(
        self,
        mock_bnb: MagicMock,
        mock_auto: MagicMock,
        mock_prep: MagicMock,
        mock_peft: MagicMock,
        mock_tok_cls: MagicMock,
    ) -> None:
        # Setup mocks
        base_model = _fake_model(total_params=8_000_000, trainable=8_000_000)
        prepared = _fake_model(total_params=8_000_000, trainable=0)
        peft_model = _fake_model(total_params=8_000_000, trainable=400_000)
        peft_model.print_trainable_parameters = MagicMock()

        mock_auto.from_pretrained.return_value = base_model
        mock_prep.return_value = prepared
        mock_peft.return_value = peft_model

        mock_tok = MagicMock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "</s>"
        mock_tok_cls.from_pretrained.return_value = mock_tok

        mc = ModelConfig(model_name_or_path="meta-llama/Llama-3.1-8B-Instruct")
        qc = QuantizationConfig()
        lc = LoRAConfig()

        loader = ModelLoader()

        # Full pipeline
        model = loader.load_base_model(mc, qc)
        peft_result = loader.apply_lora(model, lc)
        tokenizer = loader.load_tokenizer(mc)

        assert model is prepared
        assert peft_result is peft_model
        assert tokenizer is mock_tok
        assert mock_tok.pad_token == "</s>"
        assert mock_tok.padding_side == "right"


# ═══════════════════════════════════════════════════════════════
# __all__ export
# ═══════════════════════════════════════════════════════════════


class TestModuleExports:
    def test_model_loader_in_all(self) -> None:
        from src.training import model_loader

        assert "ModelLoader" in model_loader.__all__

    def test_importable_from_training_package(self) -> None:
        from src.training import ModelLoader as ML

        assert ML is ModelLoader
