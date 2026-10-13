"""A CDK app for the strands-decider SageMaker endpoint (README.md).

One stack: the execution role, the SageMaker model, the endpoint config and the endpoint.
The image is either built from deploy/sagemaker/Dockerfile (needs Docker), or an existing
ECR image passed as `-c image=URI`, such as one from build_and_push.sh.
"""

from __future__ import annotations

import os

import aws_cdk as cdk
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_sagemaker as sagemaker
from constructs import Construct

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ASSET_EXCLUDE = [
    "*",
    "!pyproject.toml",
    "!README.md",
    "!LICENSE",
    "!src/",
    "!deploy/sagemaker/Dockerfile",
    "!deploy/sagemaker/prefetch.py",
    "**/__pycache__",
    "**/*.egg-info",
]


class DeciderEndpointStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        ctx = self.node.try_get_context
        name = ctx("name")
        device = ctx("device")
        image_uri = ctx("image")

        role = iam.Role(
            self, "ExecutionRole", assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com")
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                resources=[f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*"],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(actions=["cloudwatch:PutMetricData"], resources=["*"])
        )

        if image_uri:
            # <account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>
            repo_name = image_uri.split("/", 1)[1].split(":")[0].split("@")[0]
            ecr.Repository.from_repository_name(self, "Repo", repo_name).grant_pull(role)
        else:
            asset = ecr_assets.DockerImageAsset(
                self,
                "Image",
                directory=REPO_ROOT,
                file="deploy/sagemaker/Dockerfile",
                # CDK copies the directory into cdk.out; keep only what the image needs, as
                # Dockerfile.dockerignore does for a plain docker build.
                exclude=ASSET_EXCLUDE,
                ignore_mode=cdk.IgnoreMode.DOCKER,
                build_args={"DEVICE": device},
                platform=ecr_assets.Platform.LINUX_AMD64,
            )
            asset.repository.grant_pull(role)
            image_uri = asset.image_uri

        model = sagemaker.CfnModel(
            self,
            "Model",
            execution_role_arn=role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=image_uri,
                environment={"DECIDER_DEVICE": device, "DECIDER_ARGS": ctx("serve_args") or ""},
            ),
        )
        # The role's policy must exist before SageMaker checks that it can pull the image.
        model.node.add_dependency(role)

        config = sagemaker.CfnEndpointConfig(
            self,
            "EndpointConfig",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    variant_name="AllTraffic",
                    model_name=model.attr_model_name,
                    instance_type=ctx("instance_type"),
                    initial_instance_count=int(ctx("instance_count")),
                    initial_variant_weight=1.0,
                    # The image holds the weights; pulling it and loading the model takes minutes.
                    container_startup_health_check_timeout_in_seconds=900,
                )
            ],
        )

        endpoint = sagemaker.CfnEndpoint(
            self,
            "Endpoint",
            endpoint_name=name,
            endpoint_config_name=config.attr_endpoint_config_name,
        )

        cdk.CfnOutput(self, "EndpointName", value=endpoint.attr_endpoint_name)
        cdk.CfnOutput(
            self,
            "InvokeUrl",
            value=f"https://runtime.sagemaker.{self.region}.amazonaws.com/endpoints/{name}/invocations",
        )


app = cdk.App()
DeciderEndpointStack(
    app,
    app.node.try_get_context("stack_name"),
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region=os.environ.get("CDK_DEFAULT_REGION")
    ),
)
app.synth()
