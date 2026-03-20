.PHONY: install synth diff deploy destroy test lint typecheck quality

ENV ?= dev

install:
	pip install -e ".[dev]"

synth:
	cdk synth --context env=$(ENV)

diff:
	cdk diff --context env=$(ENV)

deploy:
	cdk deploy --all --context env=$(ENV) --require-approval broadening

destroy:
	cdk destroy --all --context env=$(ENV)

test:
	pytest tests/ -v --cov=stacks --cov=constructs --cov-report=term-missing

lint:
	ruff check stacks/ constructs/ config/ tests/ app.py
	ruff format --check stacks/ constructs/ config/ tests/ app.py

typecheck:
	mypy stacks/ constructs/ config/ app.py

quality: lint typecheck test
