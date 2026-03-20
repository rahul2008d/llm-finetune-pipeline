#!/usr/bin/env python3
"""CDK application entry point for LLM fine-tuning infrastructure."""
from __future__ import annotations

import os

import aws_cdk as cdk

from config.constants import DEFAULT_TAGS
from config.environments import get_environment_config
from stacks.iam_stack import IamStack
from stacks.network_stack import NetworkStack
from stacks.sagemaker_stack import SageMakerStack
from stacks.storage_stack import StorageStack

app = cdk.App()

env_name = app.node.try_get_context("env") or os.getenv("CDK_ENV", "dev")
config = get_environment_config(env_name)

cdk_env = cdk.Environment(account=config.account, region=config.region)

# Stack instantiation order matters: dependencies flow left to right
network = NetworkStack(
    app,
    f"{config.env_name}-network",
    config=config,
    env=cdk_env,
)
storage = StorageStack(
    app,
    f"{config.env_name}-storage",
    config=config,
    vpc=network.vpc_construct,
    env=cdk_env,
)
iam = IamStack(
    app,
    f"{config.env_name}-iam",
    config=config,
    training_data_bucket=storage.training_data_bucket.bucket,
    model_artifacts_bucket=storage.model_artifacts_bucket.bucket,
    mlflow_bucket=storage.mlflow_bucket.bucket,
    kms_key=storage.kms_key,
    training_ecr_repo=storage.training_ecr_repo,
    serving_ecr_repo=storage.serving_ecr_repo,
    vpc=network.vpc_construct.vpc,
    env=cdk_env,
)
sagemaker = SageMakerStack(
    app,
    f"{config.env_name}-sagemaker",
    config=config,
    vpc_construct=network.vpc_construct,
    storage_stack=storage,
    iam_stack=iam,
    env=cdk_env,
)

# Apply default tags to ALL resources in ALL stacks
for key, value in DEFAULT_TAGS.items():
    cdk.Tags.of(app).add(key, value)
cdk.Tags.of(app).add("Environment", config.env_name)

app.synth()
