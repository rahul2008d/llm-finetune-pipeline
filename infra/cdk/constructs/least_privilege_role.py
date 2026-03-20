"""Permission boundary and role factory for SageMaker pipelines."""
from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import aws_iam as iam
from constructs import Construct

from config.constants import PROJECT_NAME


class PermissionBoundary(Construct):
    """Managed policy used as permission boundary for all project roles."""

    def __init__(self, scope: Construct, id: str, *, env_name: str) -> None:
        super().__init__(scope, id)
        self.policy = iam.ManagedPolicy(
            self, "Boundary",
            managed_policy_name=f"{PROJECT_NAME}-{env_name}-boundary",
            statements=[
                # Allow everything EXCEPT IAM mutations and org-level actions
                iam.PolicyStatement(
                    sid="AllowMostActions",
                    effect=iam.Effect.ALLOW,
                    actions=["*"],
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    sid="DenyIAMMutations",
                    effect=iam.Effect.DENY,
                    actions=[
                        "iam:CreateUser", "iam:DeleteUser",
                        "iam:CreateRole", "iam:DeleteRole",
                        "iam:AttachRolePolicy", "iam:DetachRolePolicy",
                        "iam:PutRolePolicy", "iam:DeleteRolePolicy",
                        "iam:CreatePolicy", "iam:DeletePolicy",
                        "iam:UpdateAssumeRolePolicy",
                        "organizations:*",
                        "account:*",
                    ],
                    resources=["*"],
                ),
            ],
        )
