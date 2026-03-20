.PHONY: install lint test format typecheck clean

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

test:
	pytest tests/unit/ -m "not slow" -v
	pytest tests/integration/ -v --timeout=120
	pytest tests/e2e/ -v --timeout=300

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage

# ---- CDK targets ----
cdk-install:
	cd infra/cdk && make install
cdk-synth:
	cd infra/cdk && make synth ENV=$(ENV)
cdk-deploy:
	cd infra/cdk && make deploy ENV=$(ENV)
cdk-test:
	cd infra/cdk && make test
cdk-quality:
	cd infra/cdk && make quality
