# Disaster Recovery Plan

## Overview

This document describes the disaster recovery (DR) strategy for the LLM Fine-Tuning Pipeline, including RPO/RTO targets, backup procedures, failover steps, and drill instructions.

---

## RPO/RTO Targets

| Asset | RPO (Recovery Point Objective) | RTO (Recovery Time Objective) |
|-------|-------------------------------|-------------------------------|
| Trained model artifacts | 1 hour | 4 hours |
| Training datasets | 24 hours | 8 hours |
| Training configurations | Real-time (Git) | Immediate (Git clone) |
| MLflow experiment data | 1 hour | 4 hours |
| Endpoint configurations | 1 hour | 2 hours |
| Serving infrastructure | N/A | 4 hours |

---

## What Is Backed Up and Where

### Model Artifacts
- **Primary:** S3 bucket `llm-finetune-models/models/` with versioning enabled
- **Cross-region:** S3 Cross-Region Replication (CRR) to `us-west-2` backup bucket
- **Retention:** All model versions retained for 90 days; production models retained indefinitely
- **Backup frequency:** Automatic on every training completion

### Training Datasets
- **Primary:** S3 bucket `llm-finetune-data/` with versioning enabled
- **Cross-region:** S3 CRR to backup region
- **Retention:** Current version + 3 previous versions
- **Backup frequency:** On every data pipeline run

### Training Configurations
- **Primary:** Git repository (GitHub)
- **Backup:** GitHub repository mirroring (if configured)
- **Retention:** Full Git history
- **Backup frequency:** Real-time (every commit)

### MLflow Experiment Data
- **Primary:** MLflow Tracking Server backed by S3 + RDS
- **Backup:** RDS automated snapshots (daily) + S3 CRR for artifacts
- **Retention:** 30-day snapshot retention
- **Backup frequency:** Daily RDS snapshots, real-time S3 replication

### Endpoint Configurations
- **Primary:** Exported to S3 `llm-finetune-models/endpoint-configs/`
- **Backup:** S3 CRR to backup region
- **Retention:** Last 10 configurations per endpoint
- **Backup frequency:** On every deployment

---

## Failover Procedure

### Pre-Requisites
- Cross-region S3 replication is active and healthy
- IAM roles exist in the target region with equivalent permissions
- Docker images are available in ECR in the target region (or use cross-region replication)

### Step-by-Step Failover

#### Step 1: Assess the Outage
```bash
# Check source region SageMaker status
aws sagemaker describe-endpoint \
  --endpoint-name llm-ft-prod-v1 \
  --region us-east-1

# Check AWS Health Dashboard for regional issues
aws health describe-events --region us-east-1
```

#### Step 2: Verify Target Region Readiness
```bash
# Verify model artifacts are replicated
aws s3 ls s3://llm-finetune-models-us-west-2/models/ --region us-west-2

# Verify ECR images
aws ecr describe-images \
  --repository-name llm-finetune-serve \
  --region us-west-2
```

#### Step 3: Execute Failover
```python
from src.ops.disaster_recovery import DisasterRecoveryManager

dr = DisasterRecoveryManager(region="us-east-1")
result = dr.full_region_failover(
    source_region="us-east-1",
    target_region="us-west-2",
    endpoint_name="llm-ft-prod-v1",
)
print(f"Failover result: {result}")
assert result["smoke_test_passed"], "Smoke test failed after failover!"
```

#### Step 4: Update DNS / Route 53
```bash
# Update Route 53 health check and failover routing
aws route53 change-resource-record-sets \
  --hosted-zone-id <zone-id> \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "llm-api.example.com",
        "Type": "CNAME",
        "TTL": 60,
        "ResourceRecords": [{"Value": "<new-endpoint-url>"}]
      }
    }]
  }'
```

#### Step 5: Verify and Monitor
```bash
# Run full smoke test against new endpoint
python -c "
from src.serving.endpoint_tester import EndpointTester
tester = EndpointTester(region='us-west-2')
result = tester.smoke_test('llm-ft-prod-v1-us-west-2')
print(result)
"

# Set up monitoring in new region
python -c "
from src.monitoring.endpoint_monitor import EndpointMonitor
monitor = EndpointMonitor(region='us-west-2')
monitor.setup_monitoring(
    endpoint_name='llm-ft-prod-v1-us-west-2',
    alert_sns_topic_arn='arn:aws:sns:us-west-2:123:alerts',
)
"
```

#### Step 6: Notify Stakeholders
- Post in `#ml-platform-incidents` Slack channel
- Update status page
- Send email to stakeholders with new endpoint details

---

## DR Drill Instructions

### Schedule
- **Frequency:** Quarterly (every 3 months)
- **Duration:** 2–4 hours
- **Participants:** MLOps engineer (lead), Platform engineer, ML engineering lead (observer)

### Drill Procedure

1. **Pre-drill (1 day before)**
   - Notify team of upcoming drill
   - Verify cross-region replication is current
   - Ensure target region has capacity (check service quotas)

2. **During drill**
   - Create a temporary endpoint in source region for testing (do NOT use production)
   - Execute failover procedure to target region
   - Verify smoke tests pass
   - Measure actual RTO (target: < 4 hours)
   - Document any issues encountered

3. **Post-drill**
   - Clean up temporary resources in both regions
   - Write drill report with findings
   - File tickets for any improvements needed
   - Update this document if procedures changed

### Drill Checklist

- [ ] S3 cross-region replication verified (objects < RPO age)
- [ ] Endpoint config export works
- [ ] Model deployment in target region succeeds
- [ ] Smoke tests pass in target region
- [ ] Monitoring and alarms configured in target region
- [ ] DNS failover tested (or simulated)
- [ ] Actual RTO measured and documented
- [ ] All temporary resources cleaned up

---

## Recovery Verification Checklist

After any recovery (drill or real incident), verify the following:

- [ ] Endpoint is `InService` in target region
- [ ] Smoke test passes with expected latency (< 5s p99)
- [ ] No 5xx errors in first 15 minutes
- [ ] CloudWatch alarms are configured and active
- [ ] CloudWatch dashboard shows metrics flowing
- [ ] Autoscaling policies are applied
- [ ] Guardrails (if Bedrock) are configured
- [ ] Monitoring reports generate correctly
- [ ] Alert notifications arrive in Slack/PagerDuty
- [ ] Logs are flowing to CloudWatch Logs
- [ ] Cost tracking is updated for new region resources
