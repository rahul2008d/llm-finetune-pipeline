# Phase 7-8: Evaluation, Experiment Tracking & Benchmarking

## Context for Copilot

I have an existing LLM fine-tuning pipeline with these modules already built:

**Existing evaluation files:**
- `src/evaluation/metrics.py` — `MetricsComputer` class with `compute_perplexity`, `compute_rouge`, `compute_bleu`, `compute_accuracy`
- `src/evaluation/benchmarks.py` — `BenchmarkRunner` class with `run_benchmark`, `run_all`
- `src/evaluation/report.py` — `ReportGenerator` class with `generate_json_report`, `generate_markdown_report`

**Existing monitoring files:**
- `src/monitoring/cloudwatch.py` — `CloudWatchMetrics` class with `put_metric`, `put_training_metrics`, `put_inference_metrics`
- `src/monitoring/alerting.py` — `AlertManager` class with `send_alert`, `send_training_alert`, `send_drift_alert`
- `src/monitoring/drift.py` — `DriftDetector` class with `detect_distribution_drift`, `detect_performance_drift`, `compute_embedding_drift`

**Existing training integration:**
- `src/training/trainer.py` — `FineTuneTrainer` with MLflow integration (graceful fallback if MLflow unavailable)
- `src/training/trainer.py` — `TrainingResult` Pydantic model with: `run_id`, `experiment_name`, `final_train_loss`, `final_eval_loss`, `best_eval_loss`, `total_steps`, `training_time_seconds`, `estimated_cost_usd`, `adapter_s3_uri`, `metrics`
- `src/config/training.py` — `TrainingJobConfig` (full production config)

**Rules for ALL files in this task:**
- Use absolute imports: `from src.evaluation...`, `from src.config.training import ...`
- Type hints on every function signature
- Docstrings on every class and every public method
- Structured logging via `structlog.get_logger(__name__)`
- Handle missing optional dependencies gracefully (try/except ImportError)
- Do NOT delete or overwrite existing methods in existing files — only ADD new methods/classes
- Do NOT modify `src/training/trainer.py` or `src/config/training.py`

---

## Prompt 27 — MLflow Experiment Tracker

Create a NEW file `src/monitoring/mlflow_tracker.py`:

```python
"""MLflow experiment tracking wrapper for the fine-tuning pipeline."""
```

### Class: `ExperimentTracker`

```python
class ExperimentTracker:
    """Centralized MLflow tracking for training, evaluation, and model registration."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        """Configure MLflow with S3 artifact store.
        
        If MLflow is unavailable or tracking_uri is empty, all methods become no-ops
        and log a warning on first call.
        """

    def start_run(self, run_name: str, config: TrainingJobConfig) -> Any:
        """Start an MLflow run and log ALL config parameters.
        
        - mlflow.start_run(run_name=run_name)
        - Log all config parameters flattened: model.torch_dtype, lora.r, etc.
        - Log tags: git_sha (from env or subprocess), dataset_id, instance_type, method (qlora/dora)
        - Log the full config YAML as an artifact
        - Return the MLflow run object (or a no-op context if MLflow unavailable)
        """

    def log_training_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log training step metrics.
        
        Required metrics: train_loss, eval_loss, learning_rate, epoch,
        gpu_memory_mb, gradient_norm, estimated_cost_usd
        """

    def log_eval_metrics(self, metrics: dict[str, float]) -> None:
        """Log evaluation results: perplexity, bleu, rouge, custom metrics."""

    def log_model_artifact(self, model_path: str, artifact_name: str) -> None:
        """Log model artifacts (adapter weights, merged model) to MLflow."""

    def log_model_card(self, model_card_content: str) -> None:
        """Serialize model card markdown and log as artifact."""

    def end_run(self, status: str = "FINISHED") -> None:
        """End the active MLflow run with given status."""

    def compare_runs(
        self,
        experiment_name: str,
        metric: str = "eval_loss",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query MLflow for top runs by metric.
        
        Returns sorted list of dicts with: run_id, run_name, params, metrics.
        Returns empty list if MLflow unavailable.
        """
```

### Tests: `tests/unit/test_monitoring/test_mlflow_tracker.py`

Write tests that mock MLflow:
- `test_start_run_logs_params` — verify mlflow.log_params called with flattened config
- `test_log_training_metrics` — verify mlflow.log_metrics called with step
- `test_log_eval_metrics` — verify metrics are logged
- `test_compare_runs_returns_sorted` — mock mlflow.search_runs, verify sort order
- `test_graceful_when_mlflow_unavailable` — patch mlflow import to fail, verify no crash
- `test_end_run_sets_status` — verify mlflow.end_run called with status

---

## Prompt 28 — CloudWatch Custom Metrics Publisher

Enhance the EXISTING `src/monitoring/cloudwatch.py` by ADDING a new class. Do NOT modify the existing `CloudWatchMetrics` class.

### New class to ADD: `TrainingMetricsPublisher`

```python
class TrainingMetricsPublisher:
    """Publish training metrics to CloudWatch asynchronously.
    
    Metrics are batched and published in a background thread
    to avoid blocking the training loop.
    """

    def __init__(
        self,
        experiment_name: str,
        job_name: str,
        instance_type: str,
        method: str,  # "qlora" or "dora"
        model_family: str,  # "llama3", "mistral", etc.
        region: str = "us-east-1",
    ) -> None:
        """Initialize publisher with dimensions for all metrics.
        
        Namespace: LLMFineTuning/{experiment_name}
        Dimensions: JobName, InstanceType, Method, ModelFamily
        Start background thread for async publishing.
        """

    def publish_training_step(self, step: int, metrics: dict[str, float]) -> None:
        """Queue metrics for async publishing.
        
        Expected metrics: TrainLoss, EvalLoss, LearningRate, GradientNorm,
        GPUMemoryUtilization, ThroughputSamplesPerSec.
        Batch up to 1000 metrics per API call.
        """

    def publish_job_summary(self, result: "TrainingResult") -> None:
        """Publish final job summary metrics.
        
        Metrics: TotalTrainingTimeMinutes, FinalEvalLoss, EstimatedCostUSD,
        TotalSteps, BestCheckpointStep.
        """

    def create_dashboard(self, experiment_name: str) -> str:
        """Create CloudWatch Dashboard and return dashboard URL.
        
        Widgets:
        - Loss curves (train + eval overlay)
        - Learning rate schedule
        - GPU memory over time
        - Cost accumulation over time
        - Gradient norm over time
        """

    def stop(self) -> None:
        """Flush remaining metrics and stop background thread."""
```

### Tests: `tests/unit/test_monitoring/test_cloudwatch_publisher.py`

- `test_publish_training_step_queues_metrics` — verify metrics added to queue
- `test_publish_job_summary` — verify summary metrics published
- `test_create_dashboard_returns_url` — mock boto3, verify dashboard JSON structure
- `test_async_publish_does_not_block` — verify publish returns immediately
- `test_stop_flushes_queue` — verify remaining metrics published on stop

---

## Prompt 29 — Model Card Generator

Create a NEW file `src/monitoring/model_card.py`:

```python
"""Automated model card generation in Hugging Face format."""
```

### Class: `ModelCardGenerator`

```python
class ModelCardGenerator:
    """Generate standardized model cards for fine-tuned models."""

    def generate(
        self,
        config: "TrainingJobConfig",
        result: "TrainingResult",
        eval_results: dict[str, Any] | None = None,
    ) -> str:
        """Generate a complete model card in Markdown format.
        
        Sections (Hugging Face standard):
        1. Model Details: name, version, base model, fine-tuning method
        2. Training Details: all hyperparameters, dataset name+version,
           hardware used (instance type), training duration, cost
        3. Adapter Details: rank, alpha, target modules, trainable params %,
           DoRA/RSLoRA flags
        4. Performance: eval loss, perplexity, benchmark results table
           (if eval_results provided)
        5. Intended Use: extracted from config if present, otherwise generic
        6. Limitations: documented by user or auto-detected
        7. Ethical Considerations: PII scan results, data provenance
        8. Lineage: dataset_id or dataset_path, code commit SHA (from git),
           config hash (SHA256 of serialized config)
        9. How to Use: Python code snippet for loading the adapter with PEFT
        
        Returns: Markdown string
        """

    def save(self, content: str, output_path: str) -> None:
        """Save model card to file (local or S3).
        
        - If output_path starts with s3://, upload via S3Client
        - Otherwise write to local file as README.md
        """

    def save_json(self, config: "TrainingJobConfig", result: "TrainingResult",
                  eval_results: dict | None, output_path: str) -> None:
        """Save model card metadata as JSON for programmatic access."""

    def log_to_mlflow(self, content: str) -> None:
        """Log model card as MLflow artifact if MLflow is available."""
```

### Tests: `tests/unit/test_monitoring/test_model_card.py`

- `test_generate_returns_markdown` — verify output is valid markdown with all 9 sections
- `test_generate_without_eval_results` — verify Performance section says "No evaluation results"
- `test_generate_includes_lora_details` — verify rank, alpha, target_modules in output
- `test_save_local` — verify file written with correct content
- `test_save_json` — verify JSON output matches expected schema
- `test_log_to_mlflow` — mock mlflow, verify artifact logged
- `test_lineage_includes_git_sha` — verify git SHA appears (mock subprocess)

---

## Prompt 30 — Evaluation Framework

Create a NEW file `src/evaluation/evaluator.py`:

```python
"""Unified model evaluation framework."""
```

### Class: `ModelEvaluator`

```python
class ModelEvaluator:
    """Evaluate fine-tuned models with multiple strategies."""

    def __init__(self, model_path: str, device_map: str = "auto") -> None:
        """Load model (merged or adapter) for evaluation.
        
        - Support local paths and S3 URIs
        - If path contains adapter_config.json, load as PEFT adapter
        - Otherwise load as full model
        - Set model to eval mode
        - Log: model size, device, dtype
        """

    def evaluate_perplexity(
        self, dataset: Any, batch_size: int = 8, stride: int = 512,
    ) -> float:
        """Compute perplexity on eval dataset using sliding window.
        
        Handle long sequences with stride-based evaluation.
        Return float perplexity value.
        """

    def evaluate_generation(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = False,
    ) -> list[dict[str, Any]]:
        """Generate responses for a list of prompts.
        
        Return list of dicts with: prompt, generated_text, num_tokens,
        latency_ms, tokens_per_second.
        """

    def evaluate_benchmarks(
        self,
        benchmarks: list[str],
        num_fewshot: int = 5,
    ) -> dict[str, Any]:
        """Run lm-evaluation-harness benchmarks.
        
        Supported: mmlu, hellaswag, arc_easy, arc_challenge, winogrande,
        truthfulqa_mc, gsm8k.
        
        Falls back gracefully if lm-eval not installed.
        Return results per benchmark with scores.
        """

    def evaluate_custom_task(
        self,
        eval_dataset_path: str,
        metrics: list[str],
        max_new_tokens: int = 512,
    ) -> dict[str, float]:
        """Run custom evaluation task.
        
        Load dataset from path, generate responses, score against references.
        Metrics: exact_match, bleu, rouge_l, f1_token.
        """

    def run_full_evaluation(self, eval_config: dict[str, Any]) -> dict[str, Any]:
        """Run all configured evaluations.
        
        eval_config should match configs/evaluation/eval_default.yaml schema.
        Aggregate all results into single dict.
        Log everything to MLflow if available.
        Return complete evaluation results.
        """
```

### Tests: `tests/unit/test_evaluation/test_evaluator.py`

- `test_evaluate_perplexity_returns_float` — mock model, verify positive float returned
- `test_evaluate_generation_returns_expected_format` — verify dict keys in output
- `test_evaluate_benchmarks_graceful_without_lm_eval` — patch import, verify no crash
- `test_evaluate_custom_task` — mock generation, verify metrics computed
- `test_run_full_evaluation` — verify all sub-evaluations called based on config

---

## Prompt 31 — Custom Metrics Library

Enhance the EXISTING `src/evaluation/metrics.py` by ADDING new functions. Do NOT modify or remove existing methods in `MetricsComputer`.

### New standalone functions to ADD (outside the class, as module-level functions):

```python
def compute_exact_match(
    predictions: list[str],
    references: list[str],
    normalize: bool = True,
) -> float:
    """Compute exact match accuracy.
    
    If normalize=True: lowercase, strip whitespace, remove articles (a, an, the).
    Return float between 0.0 and 1.0.
    Handle empty inputs: return 0.0 with warning.
    """


def compute_f1_token_overlap(prediction: str, reference: str) -> float:
    """Compute token-level F1 score (SQuAD-style).
    
    1. Tokenize both strings by whitespace
    2. Compute precision = matching_tokens / prediction_tokens
    3. Compute recall = matching_tokens / reference_tokens
    4. F1 = 2 * precision * recall / (precision + recall)
    Handle empty inputs: return 0.0.
    """


def compute_batch_f1(predictions: list[str], references: list[str]) -> float:
    """Average F1 across a batch of prediction/reference pairs."""


def compute_coherence_score(texts: list[str]) -> float:
    """Measure internal consistency of generated texts.
    
    Based on sentence-level perplexity variance.
    Lower variance = more coherent.
    Return normalized score between 0.0 and 1.0.
    Requires a tokenizer — accept optional tokenizer param, skip if not provided.
    """


def compute_diversity(texts: list[str]) -> dict[str, float]:
    """Compute lexical diversity metrics.
    
    Returns dict with:
    - distinct_1: unique unigrams / total unigrams
    - distinct_2: unique bigrams / total bigrams
    - distinct_3: unique trigrams / total trigrams
    - self_bleu: average BLEU of each text against all others (inverse diversity)
    
    Handle empty inputs: return all 0.0 with warning.
    """


def compute_toxicity(texts: list[str]) -> dict[str, Any]:
    """Score texts for toxicity.
    
    Strategy: use a lightweight keyword-based approach as default.
    If a toxicity model is available (e.g., detoxify), use it.
    
    Returns dict with:
    - mean_toxicity: float
    - max_toxicity: float
    - num_flagged: int
    - flagged_indices: list[int]
    
    Handle missing toxicity model gracefully.
    """


def compute_repetition_rate(texts: list[str], n: int = 3) -> float:
    """Compute the rate of repeated n-grams across generated texts.
    
    Higher = more repetitive.
    Return float between 0.0 and 1.0.
    """
```

### Tests: `tests/unit/test_evaluation/test_custom_metrics.py`

Write tests for EACH function:
- `test_exact_match_identical_strings` — returns 1.0
- `test_exact_match_with_normalization` — "The cat" vs "the cat" returns 1.0
- `test_exact_match_empty_input` — returns 0.0
- `test_f1_token_overlap_partial` — "hello world" vs "hello there" returns expected F1
- `test_f1_token_overlap_no_overlap` — returns 0.0
- `test_f1_token_overlap_empty` — returns 0.0
- `test_batch_f1_averages_correctly` — verify average of individual F1 scores
- `test_diversity_all_same` — repeated text returns low distinct scores
- `test_diversity_all_different` — varied text returns high distinct scores
- `test_diversity_empty` — returns all 0.0
- `test_toxicity_clean_text` — returns low scores
- `test_toxicity_handles_missing_model` — no crash without detoxify
- `test_repetition_rate_no_repeats` — returns 0.0 for unique text
- `test_repetition_rate_high_repeats` — returns high value for repeated text
- `test_coherence_score_returns_float` — verify type
- `test_all_metrics_handle_empty_lists` — every function returns gracefully on []

---

## Prompt 32 — A/B Model Comparison Engine

Create a NEW file `src/evaluation/comparator.py`:

```python
"""Side-by-side model comparison with statistical testing."""
```

### Class: `ModelComparator`

```python
@dataclass
class MetricComparison:
    """Comparison of a single metric between two models."""
    metric_name: str
    model_a_value: float
    model_b_value: float
    difference: float
    relative_change_pct: float
    p_value: float | None  # from bootstrap test
    winner: str  # "model_a", "model_b", or "tie"


@dataclass
class GenerationComparison:
    """Side-by-side generation example."""
    prompt: str
    model_a_output: str
    model_b_output: str


@dataclass
class ComparisonReport:
    """Complete comparison between two models."""
    model_a_name: str
    model_b_name: str
    metrics_comparison: dict[str, MetricComparison]
    generation_examples: list[GenerationComparison]
    recommendation: str  # auto-generated summary


class ModelComparator:
    """Compare two models on identical evaluation data."""

    def compare(
        self,
        model_a_path: str,
        model_b_path: str,
        eval_dataset: Any,
        metrics: list[str],
        num_generation_examples: int = 20,
    ) -> ComparisonReport:
        """Full comparison of two models.
        
        1. Load both models via ModelEvaluator
        2. Run identical evaluation on both
        3. Compute paired bootstrap confidence intervals (1000 resamples)
        4. Generate comparison table with win/loss/tie per metric
        5. Generate side-by-side generation examples (random sample)
        6. Auto-generate recommendation text
        """

    def compare_against_base(
        self,
        finetuned_path: str,
        base_model_name: str,
        eval_dataset: Any,
        metrics: list[str] | None = None,
    ) -> ComparisonReport:
        """Compare fine-tuned model against its base.
        
        Default metrics: perplexity, rouge_l, exact_match.
        Useful for measuring improvement from fine-tuning.
        """

    @staticmethod
    def _bootstrap_p_value(
        scores_a: list[float],
        scores_b: list[float],
        n_resamples: int = 1000,
        seed: int = 42,
    ) -> float:
        """Compute p-value via paired bootstrap resampling."""
```

### Tests: `tests/unit/test_evaluation/test_comparator.py`

- `test_compare_returns_report` — mock both models, verify ComparisonReport structure
- `test_bootstrap_p_value` — verify returns float between 0 and 1
- `test_compare_against_base` — verify it calls compare with correct args
- `test_generation_examples_count` — verify correct number of examples
- `test_recommendation_generated` — verify recommendation string is non-empty

---

## Prompt 33 — Evaluation Config & Report Generator

### Part A: Create config file `configs/evaluation/eval_default.yaml`

```yaml
# Default evaluation configuration
model_path: ""  # set at runtime via CLI --model-path

benchmarks:
  - mmlu
  - hellaswag
  - arc_challenge
  - truthfulqa_mc

custom_tasks: []
  # Example:
  # - name: domain_qa
  #   dataset_path: s3://bucket/eval-data/domain_qa.jsonl
  #   metrics: [exact_match, rouge_l, f1]

generation_eval:
  prompts_path: ""  # path to JSONL file with test prompts
  max_new_tokens: 512
  temperature: 0.1
  top_p: 0.9
  do_sample: false

perplexity:
  dataset_split: "test"
  stride: 512
  batch_size: 8

compare_against_base: true

thresholds:
  max_perplexity: 15.0
  min_rouge_l: 0.3
  min_exact_match: 0.5
  max_toxicity: 0.1
```

### Part B: Enhance EXISTING `src/evaluation/report.py`

ADD a new method to the existing `ReportGenerator` class. Do NOT modify existing methods.

```python
    @staticmethod
    def generate_full_evaluation_report(
        eval_results: dict[str, Any],
        config: dict[str, Any],
        comparison: "ComparisonReport | None" = None,
        output_path: str = "results/eval_report.md",
    ) -> str:
        """Generate comprehensive Markdown evaluation report.
        
        Sections:
        1. Executive Summary — pass/fail against thresholds from config
        2. Benchmark Results — table with scores per benchmark
        3. Custom Task Results — table per custom task (if any)
        4. Generation Quality — sample outputs with quality scores
        5. Perplexity Analysis — perplexity score with context
        6. Base Model Comparison — ComparisonReport table (if provided)
        7. Toxicity & Safety — toxicity scores summary
        8. Recommendations — auto-generated based on results vs thresholds
        
        Save to output_path (local or S3).
        Log to MLflow as artifact if available.
        Return the Markdown string.
        """
```

### Tests: `tests/unit/test_evaluation/test_full_report.py`

- `test_full_report_contains_all_sections` — verify section headers present
- `test_full_report_pass_fail_thresholds` — verify pass/fail logic
- `test_full_report_without_comparison` — verify no crash when comparison=None
- `test_full_report_saves_to_file` — verify file written
- `test_full_report_with_empty_results` — verify graceful output

---

## Summary of files to create/modify

### New files to CREATE:
1. `src/monitoring/mlflow_tracker.py`
2. `src/monitoring/model_card.py`
3. `src/evaluation/evaluator.py`
4. `src/evaluation/comparator.py`
5. `configs/evaluation/eval_default.yaml`
6. `tests/unit/test_monitoring/test_mlflow_tracker.py`
7. `tests/unit/test_monitoring/test_model_card.py`
8. `tests/unit/test_monitoring/test_cloudwatch_publisher.py`
9. `tests/unit/test_evaluation/test_evaluator.py`
10. `tests/unit/test_evaluation/test_custom_metrics.py`
11. `tests/unit/test_evaluation/test_comparator.py`
12. `tests/unit/test_evaluation/test_full_report.py`

### Existing files to ENHANCE (add to, not replace):
1. `src/monitoring/cloudwatch.py` — add `TrainingMetricsPublisher` class
2. `src/evaluation/metrics.py` — add standalone metric functions
3. `src/evaluation/report.py` — add `generate_full_evaluation_report` method

### Files to NOT modify:
- `src/training/trainer.py`
- `src/training/model_loader.py`
- `src/config/training.py`
- `src/cli.py`
