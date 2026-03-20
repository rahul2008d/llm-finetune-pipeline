"""Integration tests for the full data preparation pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.pipeline import (
    DataPipeline,
    DataPipelineConfig,
    PIIAbortError,
    PipelineReport,
    StepResult,
)
from src.data.schemas import RawSample


# ── fixtures ────────────────────────────────────────────────────

_N_SAMPLES = 100


@pytest.fixture()
def fixture_dataset(tmp_path: Path) -> Path:
    """Write a 100-sample JSONL fixture file and return its path."""
    out = tmp_path / "fixture.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for i in range(_N_SAMPLES):
            record = {
                "instruction": f"Explain concept number {i} in detail.",
                "output": f"Concept {i} is about topic {i}. "
                f"It involves understanding factor {i} and applying it "
                f"to scenario {i}.",
            }
            fh.write(json.dumps(record) + "\n")
    return out


@pytest.fixture()
def base_config(fixture_dataset: Path, tmp_path: Path) -> DataPipelineConfig:
    """Minimal pipeline config pointing at the fixture dataset."""
    return DataPipelineConfig(
        source=str(fixture_dataset),
        source_format="jsonl",
        model_name="gpt2",
        template_name="alpaca",
        max_seq_length=512,
        output_s3_uri=str(tmp_path / "s3_mock"),
        pii_scan=False,  # skip PII for speed in default config
        dedup=False,  # skip dedup for speed
        seed=42,
        n_workers=1,
        cache_dir=str(tmp_path / "cache"),
    )


# ── full pipeline ───────────────────────────────────────────────


@pytest.mark.integration
class TestFullPipeline:
    """Run the entire pipeline on a 100-sample fixture."""

    def test_run_returns_report(
        self, base_config: DataPipelineConfig
    ) -> None:
        pipeline = DataPipeline()
        report = pipeline.run(base_config)
        assert isinstance(report, PipelineReport)

    def test_report_has_all_steps(
        self, base_config: DataPipelineConfig
    ) -> None:
        report = DataPipeline().run(base_config)
        step_nums = [s.step for s in report.steps]
        assert 1 in step_nums  # ingest
        assert 2 in step_nums  # validate
        assert 6 in step_nums  # tokenize
        assert 8 in step_nums  # report

    def test_manifest_populated(
        self, base_config: DataPipelineConfig
    ) -> None:
        report = DataPipeline().run(base_config)
        assert report.manifest is not None
        assert report.manifest.num_samples > 0
        assert len(report.manifest.sha256_checksum) == 64

    def test_all_samples_survive(
        self, base_config: DataPipelineConfig
    ) -> None:
        """With dedup off and no PII, all 100 samples should survive."""
        report = DataPipeline().run(base_config)
        assert report.manifest is not None
        assert report.manifest.num_samples == _N_SAMPLES

    def test_tokenized_files_on_disk(
        self, base_config: DataPipelineConfig
    ) -> None:
        DataPipeline().run(base_config)
        cache = Path(base_config.cache_dir)
        tok_dir = cache / "tokenized"
        assert (tok_dir / "train.jsonl").exists()
        assert (tok_dir / "val.jsonl").exists()
        assert (tok_dir / "test.jsonl").exists()
        assert (tok_dir / "manifest.json").exists()

    def test_statistics_reasonable(
        self, base_config: DataPipelineConfig
    ) -> None:
        report = DataPipeline().run(base_config)
        stats = report.manifest.statistics
        assert stats.total_samples == _N_SAMPLES
        assert stats.avg_input_tokens > 0
        assert stats.avg_output_tokens > 0
        assert stats.max_input_tokens <= base_config.max_seq_length

    def test_report_json_serialisable(
        self, base_config: DataPipelineConfig
    ) -> None:
        report = DataPipeline().run(base_config)
        j = report.to_json()
        parsed = json.loads(j)
        assert "steps" in parsed
        assert "manifest" in parsed

    def test_report_summary_string(
        self, base_config: DataPipelineConfig
    ) -> None:
        report = DataPipeline().run(base_config)
        s = report.summary()
        assert "Pipeline Report" in s
        assert base_config.model_name in s

    def test_report_saved_to_cache(
        self, base_config: DataPipelineConfig
    ) -> None:
        DataPipeline().run(base_config)
        report_file = Path(base_config.cache_dir) / "report.json"
        assert report_file.exists()
        parsed = json.loads(report_file.read_text())
        assert parsed["manifest"]["num_samples"] == _N_SAMPLES

    def test_deterministic_runs(
        self, fixture_dataset: Path, tmp_path: Path
    ) -> None:
        """Two runs with same seed produce identical manifests."""
        cfg1 = DataPipelineConfig(
            source=str(fixture_dataset),
            source_format="jsonl",
            model_name="gpt2",
            template_name="alpaca",
            max_seq_length=512,
            output_s3_uri=str(tmp_path / "s3_1"),
            pii_scan=False,
            dedup=False,
            seed=42,
            n_workers=1,
            cache_dir=str(tmp_path / "cache1"),
        )
        cfg2 = DataPipelineConfig(
            source=str(fixture_dataset),
            source_format="jsonl",
            model_name="gpt2",
            template_name="alpaca",
            max_seq_length=512,
            output_s3_uri=str(tmp_path / "s3_2"),
            pii_scan=False,
            dedup=False,
            seed=42,
            n_workers=1,
            cache_dir=str(tmp_path / "cache2"),
        )
        r1 = DataPipeline().run(cfg1)
        r2 = DataPipeline().run(cfg2)
        assert r1.manifest.sha256_checksum == r2.manifest.sha256_checksum


# ── resumption ──────────────────────────────────────────────────


@pytest.mark.integration
class TestResumption:
    def test_resume_from_step_4(
        self, base_config: DataPipelineConfig
    ) -> None:
        """Run full, then resume from step 4. Final output should match."""
        pipeline = DataPipeline()
        full = pipeline.run(base_config)

        resumed_config = base_config.model_copy(
            update={"resume_from_step": 4}
        )
        resumed = pipeline.run(resumed_config)

        assert resumed.manifest is not None
        assert resumed.manifest.num_samples == full.manifest.num_samples

    def test_resume_skips_earlier_steps(
        self, base_config: DataPipelineConfig
    ) -> None:
        pipeline = DataPipeline()
        pipeline.run(base_config)

        resumed_config = base_config.model_copy(
            update={"resume_from_step": 4}
        )
        resumed = pipeline.run(resumed_config)

        step_nums = [s.step for s in resumed.steps]
        assert 1 not in step_nums
        assert 2 not in step_nums


# ── max_samples ─────────────────────────────────────────────────


@pytest.mark.integration
class TestMaxSamples:
    def test_max_samples_limits_output(
        self, base_config: DataPipelineConfig
    ) -> None:
        cfg = base_config.model_copy(update={"max_samples": 10})
        report = DataPipeline().run(cfg)
        assert report.manifest.num_samples == 10


# ── PII abort ───────────────────────────────────────────────────


@pytest.mark.integration
class TestPIIAbort:
    def test_abort_on_pii_raises(self, tmp_path: Path) -> None:
        """Dataset with PII + abort_on_pii=True should raise."""
        out = tmp_path / "pii.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            record = {
                "instruction": "Who is the contact?",
                "output": "Call John Smith at john.smith@example.com or 555-123-4567.",
            }
            fh.write(json.dumps(record) + "\n")

        cfg = DataPipelineConfig(
            source=str(out),
            source_format="jsonl",
            model_name="gpt2",
            template_name="alpaca",
            max_seq_length=512,
            output_s3_uri=str(tmp_path / "s3"),
            pii_scan=True,
            abort_on_pii=True,
            n_workers=1,
            cache_dir=str(tmp_path / "cache"),
        )
        with pytest.raises(PIIAbortError):
            DataPipeline().run(cfg)


# ── step details ────────────────────────────────────────────────


@pytest.mark.integration
class TestStepDetails:
    def test_each_step_has_duration(
        self, base_config: DataPipelineConfig
    ) -> None:
        report = DataPipeline().run(base_config)
        for s in report.steps:
            assert isinstance(s.duration_seconds, float)
            assert s.duration_seconds >= 0

    def test_step_counts_non_negative(
        self, base_config: DataPipelineConfig
    ) -> None:
        report = DataPipeline().run(base_config)
        for s in report.steps:
            assert s.input_count >= 0
            assert s.output_count >= 0

