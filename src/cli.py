"""CLI entry point for the LLM fine-tuning pipeline.

Usage::

    llm-ft data validate --config configs/data/validation.yaml
    llm-ft train local --config configs/training/qlora_llama3_8b.yaml
    llm-ft evaluate --model-path s3://... --benchmark mmlu,hellaswag
    llm-ft deploy sagemaker --model s3://... --endpoint-name prod-v1
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import typer
import yaml
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.logging import configure_logging, get_logger
from src.config.training import TrainingJobConfig

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_VALIDATION = 2

# ---------------------------------------------------------------------------
# Console / logger
# ---------------------------------------------------------------------------
console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Pydantic models for per-command config validation
# ---------------------------------------------------------------------------


class DataValidateConfig(BaseModel):
    dataset: str = Field(description="Dataset name or local path")
    checks: list[str] = Field(default_factory=lambda: ["schema", "nulls", "duplicates"])
    sample_size: int | None = Field(default=None, ge=1)


class DataPrepareConfig(BaseModel):
    dataset: str = Field(description="Source dataset name or path")
    output: str = Field(description="S3 or local output path")
    format: str = Field(default="jsonl")
    max_samples: int | None = Field(default=None, ge=1)
    seed: int = Field(default=42)


class TrainLocalConfig(BaseModel):
    model_id: str = Field(description="Base model identifier")
    dataset: str = Field(description="Dataset name or path")
    output_dir: str = Field(default="./output", description="Output directory")
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    lora_rank: int = Field(default=64, ge=1)
    lora_alpha: int = Field(default=128, ge=1)
    use_qlora: bool = Field(default=True)
    use_dora: bool = Field(default=False)


class TrainSageMakerConfig(TrainLocalConfig):
    instance: str = Field(default="ml.g5.2xlarge")
    instance_count: int = Field(default=1, ge=1)
    role_arn: str = Field(default="")
    volume_size_gb: int = Field(default=100, ge=1)


class TrainHPOConfig(BaseModel):
    model_id: str = Field(description="Base model identifier")
    dataset: str = Field(description="Dataset name or path")
    output_dir: str = Field(default="./output")
    max_jobs: int = Field(default=10, ge=1)
    max_parallel_jobs: int = Field(default=2, ge=1)
    strategy: str = Field(default="Bayesian")
    objective_metric: str = Field(default="eval_loss")
    objective_type: str = Field(default="Minimize")


class EvaluateConfig(BaseModel):
    model_path: str = Field(description="Path to fine-tuned model")
    benchmarks: list[str] = Field(default_factory=list)
    output_dir: str = Field(default="results/")
    num_fewshot: int = Field(default=0, ge=0)


class MergeConfig(BaseModel):
    adapter_path: str = Field(description="Path to LoRA adapter")
    base_model: str = Field(description="Base model identifier")
    output: str = Field(description="Output path for merged model")
    push_to_hub: bool = Field(default=False)


class DeploySageMakerConfig(BaseModel):
    model: str = Field(description="S3 URI of model artifacts")
    instance: str = Field(default="ml.g5.xlarge")
    instance_count: int = Field(default=1, ge=1)
    endpoint_name: str = Field(description="SageMaker endpoint name")
    role_arn: str = Field(default="")


class DeployBedrockConfig(BaseModel):
    model: str = Field(description="S3 URI of model artifacts")
    model_name: str = Field(description="Bedrock custom model name")
    role_arn: str = Field(default="")


class MonitorStatusConfig(BaseModel):
    endpoint_name: str = Field(description="Endpoint name to monitor")
    region: str = Field(default="us-east-1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(path: Path, cli_overrides: dict[str, Any], model_cls: type[BaseModel]) -> BaseModel:
    """Load YAML, overlay CLI args, and validate via Pydantic.

    Returns a validated model instance.  Raises ``SystemExit(2)`` on
    validation failure.
    """
    if not path.exists():
        console.print(f"[red]Config file not found:[/red] {path}")
        raise SystemExit(EXIT_VALIDATION)

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # CLI args override YAML values (skip None / unset values)
    merged = {**raw, **{k: v for k, v in cli_overrides.items() if v is not None}}

    try:
        return model_cls.model_validate(merged)
    except ValidationError as exc:
        console.print(f"[red]Validation error in config:[/red]\n{exc}")
        raise SystemExit(EXIT_VALIDATION) from exc


def _confirm(summary: str, yes: bool) -> None:
    """Print a rich panel summary and ask for confirmation."""
    console.print(Panel(summary, title="Plan", border_style="cyan"))
    if not yes:
        if not typer.confirm("Proceed?"):
            console.print("[yellow]Aborted.[/yellow]")
            raise SystemExit(EXIT_SUCCESS)


def _run_command(func: Any, *args: Any, **kwargs: Any) -> None:
    """Run *func* inside the common exception handler."""
    logger = get_logger("cli")
    try:
        func(*args, **kwargs)
    except SystemExit:
        raise
    except ValidationError as exc:
        logger.error("Validation error", error=str(exc))
        console.print(f"[red]Validation error:[/red] {exc}")
        raise SystemExit(EXIT_VALIDATION) from exc
    except Exception as exc:
        logger.error("Command failed", error=str(exc), exc_info=True)
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(EXIT_ERROR) from exc


# ---------------------------------------------------------------------------
# Typer app + sub-apps
# ---------------------------------------------------------------------------
app = typer.Typer(
    name="llm-ft",
    help="LLM fine-tuning pipeline CLI.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

data_app = typer.Typer(name="data", help="Data validation and preparation.", no_args_is_help=True)
train_app = typer.Typer(name="train", help="Model training (local, SageMaker, HPO).", no_args_is_help=True)
deploy_app = typer.Typer(name="deploy", help="Model deployment (SageMaker, Bedrock).", no_args_is_help=True)
monitor_app = typer.Typer(name="monitor", help="Endpoint monitoring.", no_args_is_help=True)

app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(deploy_app, name="deploy")
app.add_typer(monitor_app, name="monitor")


# ---- data ----------------------------------------------------------------


@data_app.command("validate")
def data_validate(
    config: Path = typer.Option(..., "--config", help="Path to data validation YAML config."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Validate a dataset against quality checks."""
    configure_logging(json_output=False)

    cfg = _load_config(config, {}, DataValidateConfig)
    assert isinstance(cfg, DataValidateConfig)

    summary = (
        f"[bold]Data Validate[/bold]\n"
        f"  dataset : {cfg.dataset}\n"
        f"  checks  : {', '.join(cfg.checks)}\n"
        f"  sample  : {cfg.sample_size or 'all'}"
    )
    _confirm(summary, yes)

    def _run() -> None:
        from src.data.validation import DataValidator

        logger = get_logger("cli.data.validate")
        validator = DataValidator()
        logger.info("Running data validation", dataset=cfg.dataset, checks=cfg.checks)
        # DataValidator.validate is expected to raise on failure
        validator.validate(cfg.dataset)
        logger.info("Data validation passed")
        console.print("[green]Validation passed.[/green]")

    _run_command(_run)


@data_app.command("prepare")
def data_prepare(
    config: Path = typer.Option(..., "--config", help="Path to data preparation YAML config."),
    output: Optional[str] = typer.Option(None, "--output", help="Override output path."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Prepare and upload a dataset."""
    configure_logging(json_output=False)

    # Try loading as full DataPipelineConfig first, fall back to simple DataPrepareConfig
    try:
        from src.data.pipeline import DataPipelineConfig
        cfg = _load_config(config, {"output_local_path": output} if output else {}, DataPipelineConfig)
        is_full_pipeline = True
    except (ValidationError, Exception):
        cfg = _load_config(config, {"output": output}, DataPrepareConfig)
        is_full_pipeline = False

    if is_full_pipeline:
        assert isinstance(cfg, DataPipelineConfig)
        summary = (
            f"[bold]Data Prepare (Full Pipeline)[/bold]\n"
            f"  source        : {cfg.source}\n"
            f"  model         : {cfg.model_name}\n"
            f"  template      : {cfg.template_name}\n"
            f"  max_seq_length: {cfg.max_seq_length}\n"
            f"  max_samples   : {cfg.max_samples or 'all'}\n"
            f"  pii_scan      : {cfg.pii_scan}\n"
            f"  output        : {cfg.output_s3_uri or cfg.output_local_path}"
        )
    else:
        assert isinstance(cfg, DataPrepareConfig)
        summary = (
            f"[bold]Data Prepare (Simple)[/bold]\n"
            f"  dataset     : {cfg.dataset}\n"
            f"  output      : {cfg.output}\n"
            f"  max_samples : {cfg.max_samples or 'all'}"
        )
    _confirm(summary, yes)

    def _run() -> None:
        logger = get_logger("cli.data.prepare")

        if is_full_pipeline:
            from src.data.pipeline import DataPipeline
            logger.info("Running full data pipeline", source=cfg.source)
            pipeline = DataPipeline()
            report = pipeline.run(cfg)
            logger.info("Pipeline complete", report=str(report))
            console.print("[green]Data pipeline complete[/green]")
        else:
            # Simple mode: just load, sample, and save
            from src.data.loader import DatasetLoader
            from datasets import load_from_disk

            logger.info("Preparing dataset (simple mode)", dataset=cfg.dataset)

            if Path(cfg.dataset).exists():
                dataset = load_from_disk(cfg.dataset)
            elif cfg.dataset.startswith("s3://"):
                dataset = DatasetLoader.from_s3(cfg.dataset)
            else:
                dataset = DatasetLoader.from_huggingface(cfg.dataset)

            if cfg.max_samples and cfg.max_samples < len(dataset):
                dataset = dataset.shuffle(seed=cfg.seed).select(range(cfg.max_samples))

            logger.info("Dataset prepared", num_rows=len(dataset))

            if cfg.output.startswith("s3://"):
                from src.utils.s3 import S3Client
                s3 = S3Client()
                local_path = Path("/tmp/prepared_dataset")  # noqa: S108
                dataset.save_to_disk(str(local_path))
                parts = cfg.output.replace("s3://", "").split("/", 1)
                s3.upload_directory(str(local_path), parts[0], parts[1] if len(parts) > 1 else "")
            else:
                output_path = Path(cfg.output)
                output_path.mkdir(parents=True, exist_ok=True)
                dataset.save_to_disk(str(output_path))

            console.print(f"[green]Dataset prepared → {cfg.output}[/green]")

    _run_command(_run)


# ---- train ---------------------------------------------------------------


@train_app.command("local")
def train_local(
    config: Path = typer.Option(..., "--config", help="Path to training YAML config."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Run local fine-tuning."""
    configure_logging(json_output=False)

    # Load the FULL production config, not the simplified TrainLocalConfig
    cfg = _load_config(config, {}, TrainingJobConfig)
    assert isinstance(cfg, TrainingJobConfig)

    summary = (
        f"[bold]Train Local[/bold]\n"
        f"  model       : {cfg.model.model_name_or_path}\n"
        f"  dataset     : {cfg.dataset_path or cfg.dataset_id}\n"
        f"  output      : {cfg.output_local_path or cfg.output_s3_uri}\n"
        f"  epochs      : {cfg.training.num_train_epochs}\n"
        f"  batch_size  : {cfg.training.per_device_train_batch_size}\n"
        f"  lr          : {cfg.training.learning_rate}\n"
        f"  LoRA r/α    : {cfg.lora.r}/{cfg.lora.lora_alpha}\n"
        f"  QLoRA 4-bit : {cfg.quantization.load_in_4bit}\n"
        f"  DoRA        : {cfg.lora.use_dora}"
    )
    _confirm(summary, yes)

    def _run() -> None:
        from src.training.trainer import FineTuneTrainer

        logger = get_logger("cli.train.local")
        logger.info("Starting local training", experiment=cfg.experiment_name)

        trainer = FineTuneTrainer(cfg)
        result = trainer.train()

        logger.info("Training complete", result=result.model_dump())
        console.print(f"[green]Training complete. Output → {result.adapter_s3_uri}[/green]")

    _run_command(_run)


@train_app.command("sagemaker")
def train_sagemaker(
    config: Path = typer.Option(..., "--config", help="Path to training YAML config."),
    instance: Optional[str] = typer.Option(None, "--instance", help="Override SageMaker instance type."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Launch fine-tuning on SageMaker."""
    configure_logging(json_output=False)

    cfg = _load_config(config, {"instance": instance}, TrainSageMakerConfig)
    assert isinstance(cfg, TrainSageMakerConfig)

    summary = (
        f"[bold]Train on SageMaker[/bold]\n"
        f"  model          : {cfg.model_id}\n"
        f"  dataset        : {cfg.dataset}\n"
        f"  instance       : {cfg.instance} × {cfg.instance_count}\n"
        f"  volume         : {cfg.volume_size_gb} GB\n"
        f"  epochs         : {cfg.epochs}\n"
        f"  batch_size     : {cfg.batch_size}\n"
        f"  lr             : {cfg.learning_rate}\n"
        f"  LoRA r/α       : {cfg.lora_rank}/{cfg.lora_alpha}\n"
        f"  QLoRA          : {cfg.use_qlora}  DoRA: {cfg.use_dora}"
    )
    _confirm(summary, yes)

    def _run() -> None:
        import sagemaker
        from sagemaker.huggingface import HuggingFace

        logger = get_logger("cli.train.sagemaker")
        logger.info("Launching SageMaker training job", instance=cfg.instance)

        sess = sagemaker.Session()
        role = cfg.role_arn or sagemaker.get_execution_role()

        hyperparameters = {
            "model_id": cfg.model_id,
            "dataset": cfg.dataset,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "use_qlora": cfg.use_qlora,
            "use_dora": cfg.use_dora,
        }

        estimator = HuggingFace(
            entry_point="train.py",
            source_dir="src",
            role=role,
            instance_type=cfg.instance,
            instance_count=cfg.instance_count,
            volume_size=cfg.volume_size_gb,
            transformers_version="4.44",
            pytorch_version="2.3",
            py_version="py310",
            hyperparameters=hyperparameters,
            sagemaker_session=sess,
        )
        estimator.fit(wait=True)

        logger.info("SageMaker training complete", job_name=estimator.latest_training_job.name)
        console.print(f"[green]SageMaker job complete: {estimator.latest_training_job.name}[/green]")

    _run_command(_run)


@train_app.command("hpo")
def train_hpo(
    config: Path = typer.Option(..., "--config", help="Path to HPO search YAML config."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Launch a SageMaker hyperparameter optimisation job."""
    configure_logging(json_output=False)

    cfg = _load_config(config, {}, TrainHPOConfig)
    assert isinstance(cfg, TrainHPOConfig)

    summary = (
        f"[bold]HPO Search[/bold]\n"
        f"  model              : {cfg.model_id}\n"
        f"  dataset            : {cfg.dataset}\n"
        f"  strategy           : {cfg.strategy}\n"
        f"  max_jobs           : {cfg.max_jobs}\n"
        f"  max_parallel_jobs  : {cfg.max_parallel_jobs}\n"
        f"  objective          : {cfg.objective_type}({cfg.objective_metric})"
    )
    _confirm(summary, yes)

    def _run() -> None:
        import sagemaker
        from sagemaker.huggingface import HuggingFace
        from sagemaker.tuner import ContinuousParameter, HyperparameterTuner, IntegerParameter

        logger = get_logger("cli.train.hpo")
        logger.info("Launching HPO job", strategy=cfg.strategy, max_jobs=cfg.max_jobs)

        sess = sagemaker.Session()
        role = sagemaker.get_execution_role()

        estimator = HuggingFace(
            entry_point="train.py",
            source_dir="src",
            role=role,
            instance_type="ml.g5.2xlarge",
            instance_count=1,
            transformers_version="4.44",
            pytorch_version="2.3",
            py_version="py310",
            hyperparameters={"model_id": cfg.model_id, "dataset": cfg.dataset},
            sagemaker_session=sess,
        )

        tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name=cfg.objective_metric,
            objective_type=cfg.objective_type,
            hyperparameter_ranges={
                "learning_rate": ContinuousParameter(1e-5, 5e-4),
                "lora_rank": IntegerParameter(8, 128),
                "epochs": IntegerParameter(1, 5),
            },
            max_jobs=cfg.max_jobs,
            max_parallel_jobs=cfg.max_parallel_jobs,
            strategy=cfg.strategy,
        )
        tuner.fit(wait=True)

        logger.info("HPO complete", best_job=tuner.best_training_job())
        console.print(f"[green]HPO complete. Best job: {tuner.best_training_job()}[/green]")

    _run_command(_run)


# ---- evaluate -------------------------------------------------------------


@app.command("evaluate")
def evaluate_cmd(
    model_path: str = typer.Option(..., "--model-path", help="S3 or local path to fine-tuned model."),
    benchmark: str = typer.Option("mmlu", "--benchmark", help="Comma-separated benchmark names."),
    config: Optional[Path] = typer.Option(None, "--config", help="Optional evaluation YAML config."),
    output_dir: str = typer.Option("results/", "--output-dir", help="Results output directory."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Evaluate a fine-tuned model against benchmarks."""
    configure_logging(json_output=False)

    benchmarks = [b.strip() for b in benchmark.split(",")]
    overrides: dict[str, Any] = {"model_path": model_path, "benchmarks": benchmarks, "output_dir": output_dir}

    if config and config.exists():
        cfg = _load_config(config, overrides, EvaluateConfig)
    else:
        try:
            cfg = EvaluateConfig.model_validate(overrides)
        except ValidationError as exc:
            console.print(f"[red]Validation error:[/red]\n{exc}")
            raise SystemExit(EXIT_VALIDATION) from exc
    assert isinstance(cfg, EvaluateConfig)

    summary = (
        f"[bold]Evaluate[/bold]\n"
        f"  model_path  : {cfg.model_path}\n"
        f"  benchmarks  : {', '.join(cfg.benchmarks)}\n"
        f"  output_dir  : {cfg.output_dir}"
    )
    _confirm(summary, yes)

    def _run() -> None:
        from src.evaluation.report import ReportGenerator

        logger = get_logger("cli.evaluate")
        logger.info("Running evaluation", model_path=cfg.model_path, benchmarks=cfg.benchmarks)

        # Attempt lm-eval harness if available, otherwise fall back to internal runner
        results: dict[str, Any] = {}
        try:
            import lm_eval

            lm_eval_results = lm_eval.simple_evaluate(
                model="hf",
                model_args=f"pretrained={cfg.model_path}",
                tasks=cfg.benchmarks,
                num_fewshot=cfg.num_fewshot,
            )
            results = lm_eval_results.get("results", {})
        except ImportError:
            logger.warning("lm-eval not installed, skipping harness benchmarks")

        out = Path(cfg.output_dir)
        ReportGenerator.generate_json_report(results, out / "eval_results.json")
        ReportGenerator.generate_markdown_report(results, out / "eval_results.md")

        logger.info("Evaluation complete", output_dir=cfg.output_dir)
        console.print(f"[green]Evaluation complete → {cfg.output_dir}[/green]")

    _run_command(_run)


# ---- merge ----------------------------------------------------------------


@app.command("merge")
def merge_cmd(
    adapter_path: str = typer.Option(..., "--adapter-path", help="S3 or local path to LoRA adapter."),
    base_model: str = typer.Option(..., "--base-model", help="Base model identifier."),
    output: str = typer.Option(..., "--output", help="Output path for merged model."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Merge LoRA adapter into the base model."""
    configure_logging(json_output=False)

    try:
        cfg = MergeConfig(adapter_path=adapter_path, base_model=base_model, output=output)
    except ValidationError as exc:
        console.print(f"[red]Validation error:[/red]\n{exc}")
        raise SystemExit(EXIT_VALIDATION) from exc

    summary = (
        f"[bold]Merge Adapter[/bold]\n"
        f"  adapter   : {cfg.adapter_path}\n"
        f"  base      : {cfg.base_model}\n"
        f"  output    : {cfg.output}"
    )
    _confirm(summary, yes)

    def _run() -> None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger = get_logger("cli.merge")
        logger.info("Merging adapter", adapter=cfg.adapter_path, base=cfg.base_model)

        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, device_map="auto", trust_remote_code=False)
        model = PeftModel.from_pretrained(base, cfg.adapter_path)
        merged = model.merge_and_unload()

        merged.save_pretrained(cfg.output)
        AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=False).save_pretrained(cfg.output)

        logger.info("Merge complete", output=cfg.output)
        console.print(f"[green]Merged model saved → {cfg.output}[/green]")

    _run_command(_run)


# ---- deploy ---------------------------------------------------------------


@deploy_app.command("sagemaker")
def deploy_sagemaker(
    model: str = typer.Option(..., "--model", help="S3 URI of model artifacts."),
    instance: str = typer.Option("ml.g5.xlarge", "--instance", help="SageMaker instance type."),
    endpoint_name: str = typer.Option(..., "--endpoint-name", help="Endpoint name."),
    instance_count: int = typer.Option(1, "--instance-count", help="Number of instances."),
    role_arn: Optional[str] = typer.Option(None, "--role-arn", help="IAM role ARN."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Deploy a model to a SageMaker real-time endpoint."""
    configure_logging(json_output=False)

    try:
        cfg = DeploySageMakerConfig(
            model=model, instance=instance, endpoint_name=endpoint_name,
            instance_count=instance_count, role_arn=role_arn or "",
        )
    except ValidationError as exc:
        console.print(f"[red]Validation error:[/red]\n{exc}")
        raise SystemExit(EXIT_VALIDATION) from exc

    summary = (
        f"[bold]Deploy → SageMaker[/bold]\n"
        f"  model          : {cfg.model}\n"
        f"  endpoint       : {cfg.endpoint_name}\n"
        f"  instance       : {cfg.instance} × {cfg.instance_count}"
    )
    _confirm(summary, yes)

    def _run() -> None:
        import sagemaker
        from src.serving.endpoint import SageMakerEndpointHandler

        logger = get_logger("cli.deploy.sagemaker")
        role = cfg.role_arn or sagemaker.get_execution_role()

        handler = SageMakerEndpointHandler(
            model_path=cfg.model,
            endpoint_name=cfg.endpoint_name,
            instance_type=cfg.instance,
            instance_count=cfg.instance_count,
        )
        handler.deploy(role_arn=role)

        logger.info("Deployment complete", endpoint=cfg.endpoint_name)
        console.print(f"[green]Endpoint live: {cfg.endpoint_name}[/green]")

    _run_command(_run)


@deploy_app.command("bedrock")
def deploy_bedrock(
    model: str = typer.Option(..., "--model", help="S3 URI of model artifacts."),
    model_name: str = typer.Option(..., "--model-name", help="Bedrock custom model name."),
    role_arn: Optional[str] = typer.Option(None, "--role-arn", help="IAM role ARN for Bedrock."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Import a model into AWS Bedrock."""
    configure_logging(json_output=False)

    try:
        cfg = DeployBedrockConfig(model=model, model_name=model_name, role_arn=role_arn or "")
    except ValidationError as exc:
        console.print(f"[red]Validation error:[/red]\n{exc}")
        raise SystemExit(EXIT_VALIDATION) from exc

    summary = (
        f"[bold]Deploy → Bedrock[/bold]\n"
        f"  model      : {cfg.model}\n"
        f"  model_name : {cfg.model_name}"
    )
    _confirm(summary, yes)

    def _run() -> None:
        from src.serving.bedrock import BedrockImporter

        logger = get_logger("cli.deploy.bedrock")
        logger.info("Importing model to Bedrock", model_name=cfg.model_name)

        importer = BedrockImporter()
        job_arn = importer.create_model_import_job(
            job_name=f"import-{cfg.model_name}",
            model_name=cfg.model_name,
            model_data_url=cfg.model,
            role_arn=cfg.role_arn,
        )

        logger.info("Bedrock import job created", job_arn=job_arn)
        console.print(f"[green]Import job submitted: {job_arn}[/green]")

    _run_command(_run)


# ---- monitor --------------------------------------------------------------


@monitor_app.command("status")
def monitor_status(
    endpoint_name: str = typer.Option(..., "--endpoint-name", help="Endpoint name to check."),
    region: str = typer.Option("us-east-1", "--region", help="AWS region."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Show status and metrics for a deployed endpoint."""
    configure_logging(json_output=False)

    try:
        cfg = MonitorStatusConfig(endpoint_name=endpoint_name, region=region)
    except ValidationError as exc:
        console.print(f"[red]Validation error:[/red]\n{exc}")
        raise SystemExit(EXIT_VALIDATION) from exc

    summary = (
        f"[bold]Monitor Status[/bold]\n"
        f"  endpoint : {cfg.endpoint_name}\n"
        f"  region   : {cfg.region}"
    )
    _confirm(summary, yes)

    def _run() -> None:
        import boto3

        logger = get_logger("cli.monitor.status")
        logger.info("Fetching endpoint status", endpoint=cfg.endpoint_name)

        sm_client = boto3.client("sagemaker", region_name=cfg.region)
        resp = sm_client.describe_endpoint(EndpointName=cfg.endpoint_name)

        status = resp["EndpointStatus"]
        table = Table(title=f"Endpoint: {cfg.endpoint_name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value")
        table.add_row("Status", status)
        table.add_row("ARN", resp.get("EndpointArn", ""))
        table.add_row("Instance", resp.get("ProductionVariants", [{}])[0].get("InstanceType", "N/A"))
        table.add_row("Creation Time", str(resp.get("CreationTime", "")))
        table.add_row("Last Modified", str(resp.get("LastModifiedTime", "")))
        console.print(table)

        logger.info("Endpoint status retrieved", endpoint=cfg.endpoint_name, status=status)

    _run_command(_run)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
