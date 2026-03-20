# Phase 9: Model Registry & Artifact Management

## Context for Copilot

I have an existing LLM fine-tuning pipeline. Training is validated end-to-end. Now I need model registry, artifact packaging, and inference handlers for deployment.

**Existing files that are relevant (do NOT modify these):**
- `src/training/trainer.py` — `FineTuneTrainer`, `TrainingResult`
- `src/training/model_loader.py` — `ModelLoader`
- `src/training/merger.py` — already exists (check before creating)
- `src/config/training.py` — `TrainingJobConfig`, `ModelConfig`, `QuantizationConfig`, `LoRAConfig`
- `src/evaluation/evaluator.py` — `ModelEvaluator`
- `src/evaluation/report.py` — `ReportGenerator`
- `src/monitoring/model_card.py` — `ModelCardGenerator`
- `src/utils/s3.py` — `S3Client`

**Existing serving files (check what exists first):**
- `src/serving/endpoint.py` — may have `SageMakerEndpointHandler`
- `src/serving/bedrock.py` — may have `BedrockImporter`

**Rules for ALL files:**
- Absolute imports: `from src.serving...`, `from src.config.training import ...`
- Type hints on every function
- Docstrings on every class and public method
- Structured logging via `structlog.get_logger(__name__)`
- Handle missing dependencies gracefully (try/except ImportError)
- Do NOT delete or overwrite existing methods — only ADD
- Write unit tests for every new class/function

---

## Prompt 34 — SageMaker Model Registry

Check if `src/serving/model_registry.py` already exists. If it does, enhance it. If not, create it.

```python
"""SageMaker Model Registry integration for versioned model management."""
```

### Class: `ModelRegistryManager`

```python
class ModelRegistryManager:
    """Manage model versions in SageMaker Model Registry."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with boto3 SageMaker client."""

    def register_model(
        self,
        model_s3_uri: str,
        model_package_group_name: str,
        training_result: "TrainingResult",
        eval_results: dict[str, Any] | None = None,
        model_card_content: str | None = None,
        inference_image_uri: str | None = None,
        supported_instance_types: list[str] | None = None,
        approval_status: str = "PendingManualApproval",
    ) -> str:
        """Register a trained model in the registry.

        1. Create ModelPackageGroup if not exists
        2. Create ModelPackage with:
           - InferenceSpecification (container image, supported instance types)
           - ModelMetrics (eval results as JSON)
           - CustomerMetadataProperties (training config hash, dataset_id, git SHA)
           - ModelApprovalStatus: PendingManualApproval (default)
        3. Return model_package_arn
        """

    def approve_model(self, model_package_arn: str) -> None:
        """Update approval status to 'Approved'."""

    def reject_model(self, model_package_arn: str, reason: str) -> None:
        """Update approval status to 'Rejected' with reason."""

    def get_latest_approved(self, group_name: str) -> dict[str, Any]:
        """Query for latest model with Approved status.
        
        Return dict with: model_package_arn, model_data_url, creation_time, metrics.
        Return empty dict if no approved model found.
        """

    def list_versions(self, group_name: str) -> list[dict[str, Any]]:
        """List all model versions with status and metrics."""

    def get_model_lineage(self, model_package_arn: str) -> dict[str, Any]:
        """Return lineage info: training config, dataset_id, code commit, eval results."""
```

### Tests: `tests/unit/test_serving/test_model_registry.py`

Use moto or mock boto3:
- `test_register_model_creates_group_and_package` — verify both API calls made
- `test_register_model_returns_arn` — verify ARN string returned
- `test_approve_model` — verify UpdateModelPackage called with Approved
- `test_reject_model` — verify rejection reason passed
- `test_get_latest_approved` — mock list response, verify filtering
- `test_list_versions` — verify all versions returned
- `test_register_model_with_eval_results` — verify metrics included

---

## Prompt 35 — Artifact Packaging

Check if `src/serving/artifact_packager.py` exists. If not, create it.

```python
"""Package model artifacts for SageMaker and Bedrock deployment."""
```

### Class: `ArtifactPackager`

```python
class ArtifactPackager:
    """Package model artifacts for different deployment targets."""

    def package_for_sagemaker(
        self,
        model_path: str,
        output_path: str,
        inference_code_path: str = "src/serving/inference.py",
    ) -> str:
        """Create model.tar.gz for SageMaker deployment.

        Contents:
          model/  (safetensors weights, config.json, tokenizer files)
          code/inference.py  (SageMaker inference handler)
          code/requirements.txt  (serving dependencies)

        Upload to S3 if output_path starts with s3://.
        Return the output path (local or S3 URI).
        """

    def package_for_bedrock(
        self,
        model_path: str,
        output_s3_uri: str,
    ) -> str:
        """Upload model artifacts to S3 in Bedrock-expected structure.

        Verify before upload:
        - config.json exists and has valid architecture field
        - tokenizer files exist
        - safetensors files exist and sum to < 50GB
        - Supported architectures: LlamaForCausalLM, MistralForCausalLM, etc.

        S3 structure:
          s3://bucket/bedrock-models/{model_name}/
            config.json
            tokenizer.json
            tokenizer_config.json
            special_tokens_map.json
            *.safetensors

        Return S3 URI.
        """

    def verify_artifact(self, path: str, target: str) -> dict[str, Any]:
        """Verify artifact integrity for given target (sagemaker or bedrock).

        Check:
        - All required files present
        - safetensors file integrity (load and check shapes)
        - Total size within limits

        Return: {"is_valid": bool, "issues": list[str], "size_gb": float}
        """
```

### Tests: `tests/unit/test_serving/test_artifact_packager.py`

- `test_package_for_sagemaker_creates_tarball` — verify tar.gz created with correct structure
- `test_package_for_bedrock_validates_architecture` — verify config.json checked
- `test_verify_artifact_valid` — verify returns is_valid=True for correct artifacts
- `test_verify_artifact_missing_files` — verify returns issues list
- `test_verify_artifact_too_large` — verify size check works
- `test_package_for_sagemaker_includes_inference_code` — verify code/ directory in tar

---

## Prompt 36 — SageMaker Inference Handler

Check if `src/serving/inference.py` already exists and what it contains. If it's empty or a stub, replace it. If it has real code, enhance it.

```python
"""SageMaker inference handler for fine-tuned LLM serving."""
```

### Functions (SageMaker contract):

```python
def model_fn(model_dir: str) -> tuple:
    """Load model and tokenizer from model_dir.

    - Detect if adapter (adapter_config.json present) or merged model
    - Load in float16/bfloat16 for GPU, float32 for CPU
    - Set model to eval mode
    - Set pad_token if not set
    - Log: model size, device, dtype
    - Return (model, tokenizer)
    """


def input_fn(request_body: str, content_type: str) -> dict:
    """Parse and validate input request.

    Support content types: application/json, text/plain
    Parse: {prompt, max_new_tokens, temperature, top_p, top_k,
            do_sample, repetition_penalty, stop_sequences}
    Validate all parameters with bounds checking.
    Return parsed dict with defaults applied.
    """


def predict_fn(input_data: dict, model_and_tokenizer: tuple) -> dict:
    """Generate text from the model.

    - Tokenize input prompt
    - Generate with torch.no_grad() and torch.cuda.amp.autocast()
    - Handle stop_sequences by checking generated text
    - Decode output, strip prompt from response
    - Return: {generated_text, num_input_tokens, num_output_tokens,
               latency_ms, finish_reason}
    """


def output_fn(prediction: dict, accept_type: str) -> str:
    """Serialize prediction to JSON.

    Include request_id for tracing (generate UUID if not present).
    Support accept types: application/json
    """
```

### Tests: `tests/unit/test_serving/test_inference.py`

- `test_input_fn_json` — verify JSON parsing with all fields
- `test_input_fn_defaults` — verify defaults applied for missing fields
- `test_input_fn_text_plain` — verify plain text creates prompt-only input
- `test_input_fn_invalid_json` — verify error handling
- `test_predict_fn_returns_expected_keys` — mock model, verify output dict
- `test_output_fn_json` — verify JSON serialization
- `test_output_fn_includes_request_id` — verify UUID present

---

## Summary of Phase 9 files

### New files to CREATE (if they don't exist):
1. `src/serving/model_registry.py`
2. `src/serving/artifact_packager.py`
3. `tests/unit/test_serving/test_model_registry.py`
4. `tests/unit/test_serving/test_artifact_packager.py`
5. `tests/unit/test_serving/test_inference.py`

### Files to ENHANCE (if they exist) or CREATE (if they don't):
1. `src/serving/inference.py`

### Files to NOT modify:
- `src/training/trainer.py`
- `src/training/model_loader.py`
- `src/config/training.py`
- `src/cli.py`
