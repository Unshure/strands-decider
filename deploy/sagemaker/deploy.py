"""Create, call and delete a SageMaker real-time endpoint for strands-decider (README.md).

    python deploy/sagemaker/deploy.py create --image IMAGE_URI --role ROLE_ARN
    python deploy/sagemaker/deploy.py invoke request.json      # or the body on stdin
    python deploy/sagemaker/deploy.py delete

Needs boto3 (`pip install boto3`) and credentials that can use SageMaker; `create` also
passes ROLE_ARN to SageMaker, which needs iam:PassRole. The model, the endpoint config and
the endpoint all take the name in --name.
"""

from __future__ import annotations

import argparse
import sys

import boto3

DEFAULT_NAME = "strands-decider-2b-qwen35-v1"
DEFAULT_INSTANCE = "ml.g6.xlarge"


def create(sm, args: argparse.Namespace) -> None:
    sm.create_model(
        ModelName=args.name,
        ExecutionRoleArn=args.role,
        PrimaryContainer={"Image": args.image, "Environment": {"DECIDER_ARGS": args.serve_args}},
    )
    sm.create_endpoint_config(
        EndpointConfigName=args.name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": args.name,
                "InstanceType": args.instance_type,
                "InitialInstanceCount": args.instance_count,
                # Pulling the image (the weights are in it) and loading the model onto the GPU
                # takes some minutes; the default limit is shorter on some instance types.
                "ContainerStartupHealthCheckTimeoutInSeconds": 900,
            }
        ],
    )
    sm.create_endpoint(EndpointName=args.name, EndpointConfigName=args.name)
    print(f"creating endpoint {args.name} on {args.instance_count} x {args.instance_type}")
    if args.no_wait:
        return
    sm.get_waiter("endpoint_in_service").wait(
        EndpointName=args.name, WaiterConfig={"Delay": 30, "MaxAttempts": 60}
    )
    print(f"endpoint {args.name} is InService")


def invoke(runtime, args: argparse.Namespace) -> None:
    response = runtime.invoke_endpoint(
        EndpointName=args.name, ContentType="application/json", Body=args.body.read()
    )
    print(response["Body"].read().decode())


def delete(sm, args: argparse.Namespace) -> None:
    sm.delete_endpoint(EndpointName=args.name)
    sm.delete_endpoint_config(EndpointConfigName=args.name)
    sm.delete_model(ModelName=args.name)
    print(f"deleted endpoint, endpoint config and model {args.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--region", help="AWS region; defaults to the configured one")
    parser.add_argument("--name", default=DEFAULT_NAME, help=f"default {DEFAULT_NAME}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create", help="create the model, endpoint config and endpoint")
    p.add_argument("--image", required=True, help="ECR image URI from build_and_push.sh")
    p.add_argument("--role", required=True, help="SageMaker execution role ARN")
    p.add_argument("--instance-type", default=DEFAULT_INSTANCE, help=f"default {DEFAULT_INSTANCE}")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument(
        "--serve-args",
        default="",
        help='extra `strands-decider serve` options, e.g. "--max-batch 8"',
    )
    p.add_argument("--no-wait", action="store_true", help="return before the endpoint is InService")

    p = sub.add_parser("invoke", help="send a /v1/systemone request body; print the response")
    p.add_argument("body", nargs="?", type=argparse.FileType("rb"), default=sys.stdin.buffer)

    sub.add_parser("delete", help="delete the endpoint, endpoint config and model")

    args = parser.parse_args()
    session = boto3.Session(region_name=args.region)
    if args.command == "invoke":
        invoke(session.client("sagemaker-runtime"), args)
    else:
        {"create": create, "delete": delete}[args.command](session.client("sagemaker"), args)


if __name__ == "__main__":
    main()
