"""Auto-scaling configuration for SageMaker endpoints."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

try:
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

_RESOURCE_ID_TEMPLATE = "endpoint/{endpoint}/variant/{variant}"
_SERVICE_NAMESPACE = "sagemaker"
_SCALABLE_DIMENSION = "sagemaker:variant:DesiredInstanceCount"


class EndpointAutoScaler:
    """Configure auto-scaling policies for SageMaker endpoints."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with Application Auto Scaling client.

        Args:
            region: AWS region.
        """
        if boto3 is None:
            raise ImportError("boto3 is required for EndpointAutoScaler")
        self.region = region
        self._client: Any = boto3.client(
            "application-autoscaling", region_name=region
        )

    def configure_autoscaling(
        self,
        endpoint_name: str,
        variant_name: str = "AllTraffic",
        min_instances: int = 1,
        max_instances: int = 4,
        target_invocations_per_instance: int = 50,
        scale_in_cooldown: int = 300,
        scale_out_cooldown: int = 60,
    ) -> None:
        """Configure target tracking scaling policy.

        1. Register scalable target with Application Auto Scaling.
        2. Create target tracking policy on InvocationsPerInstance metric.

        Args:
            endpoint_name: SageMaker endpoint name.
            variant_name: Production variant name.
            min_instances: Minimum instance count.
            max_instances: Maximum instance count.
            target_invocations_per_instance: Target invocations per instance.
            scale_in_cooldown: Seconds to wait before scaling in.
            scale_out_cooldown: Seconds to wait before scaling out.
        """
        resource_id = _RESOURCE_ID_TEMPLATE.format(
            endpoint=endpoint_name, variant=variant_name
        )

        # 1. Register scalable target
        self._client.register_scalable_target(
            ServiceNamespace=_SERVICE_NAMESPACE,
            ResourceId=resource_id,
            ScalableDimension=_SCALABLE_DIMENSION,
            MinCapacity=min_instances,
            MaxCapacity=max_instances,
        )
        logger.info(
            "Registered scalable target",
            endpoint=endpoint_name,
            min=min_instances,
            max=max_instances,
        )

        # 2. Create target tracking policy
        policy_name = f"{endpoint_name}-{variant_name}-scaling-policy"
        self._client.put_scaling_policy(
            PolicyName=policy_name,
            ServiceNamespace=_SERVICE_NAMESPACE,
            ResourceId=resource_id,
            ScalableDimension=_SCALABLE_DIMENSION,
            PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration={
                "TargetValue": float(target_invocations_per_instance),
                "PredefinedMetricSpecification": {
                    "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
                },
                "ScaleInCooldown": scale_in_cooldown,
                "ScaleOutCooldown": scale_out_cooldown,
            },
        )
        logger.info(
            "Created scaling policy",
            policy=policy_name,
            target=target_invocations_per_instance,
        )

    def configure_scheduled_scaling(
        self,
        endpoint_name: str,
        variant_name: str = "AllTraffic",
        schedules: list[dict[str, Any]] | None = None,
    ) -> None:
        """Configure scheduled scaling actions.

        Each schedule dict: {name, schedule_expression (cron), min_capacity, max_capacity}.

        Args:
            endpoint_name: SageMaker endpoint name.
            variant_name: Production variant name.
            schedules: List of schedule definitions.
        """
        if not schedules:
            return

        resource_id = _RESOURCE_ID_TEMPLATE.format(
            endpoint=endpoint_name, variant=variant_name
        )

        for schedule in schedules:
            self._client.put_scheduled_action(
                ServiceNamespace=_SERVICE_NAMESPACE,
                ScheduledActionName=schedule["name"],
                ResourceId=resource_id,
                ScalableDimension=_SCALABLE_DIMENSION,
                Schedule=schedule["schedule_expression"],
                ScalableTargetAction={
                    "MinCapacity": schedule.get("min_capacity", 1),
                    "MaxCapacity": schedule.get("max_capacity", 4),
                },
            )
            logger.info(
                "Created scheduled scaling action",
                name=schedule["name"],
                schedule=schedule["schedule_expression"],
            )

    def get_scaling_status(self, endpoint_name: str) -> dict[str, Any]:
        """Get current scaling status.

        Args:
            endpoint_name: SageMaker endpoint name.

        Returns:
            Dict with scalable targets, policies, and scheduled actions.
        """
        resource_id_prefix = f"endpoint/{endpoint_name}/variant/"

        # Get scalable targets
        targets_resp = self._client.describe_scalable_targets(
            ServiceNamespace=_SERVICE_NAMESPACE,
        )
        targets = [
            t
            for t in targets_resp.get("ScalableTargets", [])
            if t.get("ResourceId", "").startswith(resource_id_prefix)
        ]

        # Get scaling policies
        policies_resp = self._client.describe_scaling_policies(
            ServiceNamespace=_SERVICE_NAMESPACE,
        )
        policies = [
            p
            for p in policies_resp.get("ScalingPolicies", [])
            if p.get("ResourceId", "").startswith(resource_id_prefix)
        ]

        # Get scaling activities
        activities_resp = self._client.describe_scaling_activities(
            ServiceNamespace=_SERVICE_NAMESPACE,
        )
        activities = [
            a
            for a in activities_resp.get("ScalingActivities", [])
            if a.get("ResourceId", "").startswith(resource_id_prefix)
        ][:10]  # Last 10

        return {
            "scalable_targets": targets,
            "scaling_policies": policies,
            "recent_activities": activities,
        }

    def remove_autoscaling(
        self, endpoint_name: str, variant_name: str = "AllTraffic"
    ) -> None:
        """Remove all scaling policies and deregister the scalable target.

        Args:
            endpoint_name: SageMaker endpoint name.
            variant_name: Production variant name.
        """
        resource_id = _RESOURCE_ID_TEMPLATE.format(
            endpoint=endpoint_name, variant=variant_name
        )

        # Remove scaling policies first
        try:
            policies_resp = self._client.describe_scaling_policies(
                ServiceNamespace=_SERVICE_NAMESPACE,
                ResourceId=resource_id,
                ScalableDimension=_SCALABLE_DIMENSION,
            )
            for policy in policies_resp.get("ScalingPolicies", []):
                self._client.delete_scaling_policy(
                    PolicyName=policy["PolicyName"],
                    ServiceNamespace=_SERVICE_NAMESPACE,
                    ResourceId=resource_id,
                    ScalableDimension=_SCALABLE_DIMENSION,
                )
                logger.info("Deleted scaling policy", policy=policy["PolicyName"])
        except Exception:
            logger.debug("No scaling policies to remove")

        # Deregister scalable target
        try:
            self._client.deregister_scalable_target(
                ServiceNamespace=_SERVICE_NAMESPACE,
                ResourceId=resource_id,
                ScalableDimension=_SCALABLE_DIMENSION,
            )
            logger.info(
                "Deregistered scalable target",
                endpoint=endpoint_name,
                variant=variant_name,
            )
        except Exception:
            logger.debug("No scalable target to deregister")


__all__: list[str] = ["EndpointAutoScaler"]
