"""SageMaker inference handler for fine-tuned LLM serving."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore[assignment, misc]
    AutoTokenizer = None  # type: ignore[assignment, misc]

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None  # type: ignore[assignment, misc]


# ── Default generation parameters ──────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "do_sample": True,
    "repetition_penalty": 1.1,
}

_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "max_new_tokens": (1, 4096),
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "top_k": (0, 1000),
    "repetition_penalty": (1.0, 5.0),
}


# ── SageMaker contract functions ───────────────────────────────


def model_fn(model_dir: str) -> tuple:
    """Load model and tokenizer from model_dir.

    - Detect if adapter (adapter_config.json present) or merged model.
    - Load in float16/bfloat16 for GPU, float32 for CPU.
    - Set model to eval mode.
    - Set pad_token if not set.
    - Log: model size, device, dtype.

    Args:
        model_dir: Path to the model directory.

    Returns:
        Tuple of (model, tokenizer).
    """
    import os
    from pathlib import Path

    model_path = Path(model_dir)

    # Determine device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Check if this is an adapter model
    is_adapter = (model_path / "adapter_config.json").exists()

    if is_adapter:
        # Load adapter config to find base model
        adapter_config_path = model_path / "adapter_config.json"
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "")

        logger.info(
            "Loading adapter model",
            base_model=base_model_name,
            adapter_path=str(model_path),
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=False,
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    else:
        logger.info("Loading merged model", path=str(model_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Set eval mode
    model.eval()

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Log model info
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded",
        device=device,
        dtype=str(dtype),
        parameters=f"{param_count / 1e9:.2f}B",
        is_adapter=is_adapter,
    )

    return (model, tokenizer)


def input_fn(request_body: str, content_type: str) -> dict:
    """Parse and validate input request.

    Supports content types: application/json, text/plain.
    Validates all parameters with bounds checking.

    Args:
        request_body: Raw request body string.
        content_type: MIME type of the request.

    Returns:
        Parsed dict with defaults applied.

    Raises:
        ValueError: If content type is unsupported or input is invalid.
    """
    if content_type == "application/json":
        try:
            data = json.loads(request_body)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        if isinstance(data, str):
            data = {"prompt": data}

        if "prompt" not in data:
            raise ValueError("Missing required field: 'prompt'")

    elif content_type == "text/plain":
        data = {"prompt": request_body}
    else:
        raise ValueError(
            f"Unsupported content type: {content_type}. "
            "Use application/json or text/plain."
        )

    # Apply defaults and validate bounds
    result: dict[str, Any] = {"prompt": data["prompt"]}

    for param, default in _DEFAULTS.items():
        value = data.get(param, default)
        if param in _PARAM_BOUNDS:
            low, high = _PARAM_BOUNDS[param]
            if isinstance(value, (int, float)):
                value = max(low, min(high, value))
        result[param] = value

    # Handle stop sequences
    result["stop_sequences"] = data.get("stop_sequences", [])

    return result


def predict_fn(input_data: dict, model_and_tokenizer: tuple) -> dict:
    """Generate text from the model.

    - Tokenize input prompt.
    - Generate with torch.no_grad() and autocast.
    - Handle stop_sequences by checking generated text.
    - Decode output, strip prompt from response.

    Args:
        input_data: Parsed input dict from input_fn.
        model_and_tokenizer: Tuple of (model, tokenizer) from model_fn.

    Returns:
        Dict with generated_text, num_input_tokens, num_output_tokens,
        latency_ms, and finish_reason.
    """
    model, tokenizer = model_and_tokenizer
    prompt = input_data["prompt"]

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    num_input_tokens = input_ids.shape[1]

    # Generation params
    gen_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": input_data["max_new_tokens"],
        "temperature": input_data["temperature"],
        "top_p": input_data["top_p"],
        "top_k": input_data["top_k"],
        "do_sample": input_data["do_sample"],
        "repetition_penalty": input_data["repetition_penalty"],
        "pad_token_id": tokenizer.pad_token_id,
    }

    # Generate
    start_time = time.perf_counter()

    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                outputs = model.generate(**gen_kwargs)
        else:
            outputs = model.generate(**gen_kwargs)

    latency_ms = (time.perf_counter() - start_time) * 1000

    # Decode — strip prompt from response
    generated_ids = outputs[0][num_input_tokens:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    num_output_tokens = len(generated_ids)

    # Determine finish reason
    finish_reason = "length"
    if num_output_tokens < input_data["max_new_tokens"]:
        finish_reason = "stop"

    # Handle stop sequences
    stop_sequences = input_data.get("stop_sequences", [])
    for stop_seq in stop_sequences:
        if stop_seq in generated_text:
            generated_text = generated_text[: generated_text.index(stop_seq)]
            finish_reason = "stop_sequence"
            break

    return {
        "generated_text": generated_text,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": num_output_tokens,
        "latency_ms": round(latency_ms, 2),
        "finish_reason": finish_reason,
    }


def output_fn(prediction: dict, accept_type: str) -> str:
    """Serialize prediction to JSON.

    Includes request_id for tracing (generates UUID if not present).

    Args:
        prediction: Prediction dict from predict_fn.
        accept_type: Requested MIME type for response.

    Returns:
        JSON string of the prediction with request_id.

    Raises:
        ValueError: If accept type is not supported.
    """
    if accept_type != "application/json":
        raise ValueError(
            f"Unsupported accept type: {accept_type}. Use application/json."
        )

    output = dict(prediction)
    if "request_id" not in output:
        output["request_id"] = str(uuid.uuid4())

    return json.dumps(output)


__all__: list[str] = ["model_fn", "input_fn", "predict_fn", "output_fn"]
