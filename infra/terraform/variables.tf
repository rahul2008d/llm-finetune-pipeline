variable "aws_region" {
  description = "AWS region for resource deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name for resource tagging"
  type        = string
  default     = "llm-finetune-pipeline"
}

variable "s3_bucket_prefix" {
  description = "Prefix for S3 bucket names"
  type        = string
  default     = "llm-finetune"
}

variable "sagemaker_instance_type" {
  description = "SageMaker training instance type"
  type        = string
  default     = "ml.g5.2xlarge"
}
