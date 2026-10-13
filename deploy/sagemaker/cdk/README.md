# The SageMaker endpoint as a CDK app

`app.py` deploys the endpoint from [../README.md](../README.md) as one CloudFormation stack,
`StrandsDeciderEndpoint`: a SageMaker execution role, the model, the endpoint config and the
endpoint. `cdk destroy` removes all of them. A change to the image or the instance type
updates the endpoint in place: SageMaker moves traffic to the new config when it is ready,
and CloudFormation rolls back if it fails.

## What you need

- Node.js (for `npx aws-cdk`) and Python 3.10 or later.
- AWS credentials, and the account and region bootstrapped for CDK once:
  `npx aws-cdk bootstrap aws://<account>/<region>`.
- Docker, unless you deploy an image that is already in ECR (`-c image=...`).

```bash
cd deploy/sagemaker/cdk
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Deploy

Build the image from `../Dockerfile` and deploy it on the defaults in `cdk.json` (one
`ml.g6.xlarge`, CUDA):

```bash
npx aws-cdk deploy
```

Or deploy an image already in ECR, such as one from `../build_and_push.sh`, on a CPU instance:

```bash
npx aws-cdk deploy \
  -c image=<account>.dkr.ecr.<region>.amazonaws.com/strands-decider-sagemaker:<tag> \
  -c device=cpu -c instance_type=ml.m5.2xlarge
```

The stack outputs the endpoint name and its invoke URL. Call it as in
[../README.md, 3. Ask it](../README.md#3-ask-it), with
`python ../deploy.py --name strands-decider invoke request.json`.

| Context (`-c key=value`) | Default | Meaning |
| --- | --- | --- |
| `name` | `strands-decider` | The endpoint name. It must not already exist, for example from `deploy.py`. |
| `image` | unset: build `../Dockerfile` | An ECR image URI to deploy instead |
| `device` | `cuda` | `cuda` or `cpu`: the build's `DEVICE` and the container's `DECIDER_DEVICE` |
| `instance_type` | `ml.g6.xlarge` | The instance type; CPU images need 16 GB or more (`ml.m5.2xlarge`) |
| `instance_count` | `1` | Instances behind the endpoint |
| `serve_args` | empty | More `strands-decider serve` options, such as `"--max-batch 8"` |
| `stack_name` | `StrandsDeciderEndpoint` | The CloudFormation stack |

If the endpoint cannot start, for example with `InsufficientInstanceCapacity`
([../README.md, 2. Endpoint](../README.md#2-endpoint)), CloudFormation rolls the stack back
and deletes what it created.

## Delete

```bash
npx aws-cdk destroy
```

This deletes the endpoint, its config, the model and the role. An image that `cdk deploy`
built stays in the CDK assets repository in ECR.
