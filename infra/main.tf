# Optional: host the MLflow tracking server on EC2 with artifacts in S3.
# Starting point - review, then `terraform init && terraform plan`.
# Restrict my_ip to your own IP (x.x.x.x/32). Destroy when done: `terraform destroy`.

terraform {
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.0" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
}

provider "aws" { region = var.region }

variable "region"   { default = "eu-north-1" } # Stockholm
variable "my_ip"    { description = "Your IP in CIDR, e.g. 1.2.3.4/32" }
variable "key_name" { description = "Existing EC2 key pair name for SSH" }

resource "random_id" "suffix" { byte_length = 4 }

resource "aws_s3_bucket" "artifacts" {
  bucket        = "sensor-doc-rag-mlflow-${random_id.suffix.hex}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_iam_role" "mlflow" {
  name = "sensor-doc-rag-mlflow-${random_id.suffix.hex}"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "s3" {
  role = aws_iam_role.mlflow.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"]
      Resource = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
    }]
  })
}

resource "aws_iam_instance_profile" "mlflow" {
  role = aws_iam_role.mlflow.name
}

resource "aws_security_group" "mlflow" {
  name = "sensor-doc-rag-mlflow-${random_id.suffix.hex}"
  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

resource "aws_instance" "mlflow" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = "t3.small"
  key_name               = var.key_name
  iam_instance_profile   = aws_iam_instance_profile.mlflow.name
  vpc_security_group_ids = [aws_security_group.mlflow.id]
  user_data = <<-EOT
    #!/bin/bash
    dnf install -y python3-pip
    pip3 install "mlflow>=3.8" boto3
    nohup mlflow server --host 0.0.0.0 --port 5000 \
      --backend-store-uri sqlite:////home/ec2-user/mlflow.db \
      --artifacts-destination s3://${aws_s3_bucket.artifacts.bucket} > /var/log/mlflow.log 2>&1 &
  EOT
  tags = { Name = "sensor-doc-rag-mlflow" }
}

output "mlflow_url" { value = "http://${aws_instance.mlflow.public_ip}:5000" }
output "bucket"     { value = aws_s3_bucket.artifacts.bucket }
