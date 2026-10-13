# Serving on Amazon SageMaker

These files put a strands-decider checkpoint on a SageMaker real-time endpoint. The default
is `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` on one `ml.g6.xlarge` (one NVIDIA L4).
The endpoint takes the same request body as `POST /v1/systemone`
([docs/inference.md, Serve](../../docs/inference.md#serve)) and returns the same response.

| File | Contents |
| --- | --- |
| `Dockerfile` | The CUDA image: the package with the `cuda` extra, and the checkpoint and its base model inside it |
| `Dockerfile.dockerignore` | What the build copies from the repository |
| `prefetch.py` | Downloads the checkpoint and its base model at build time |
| `build_and_push.sh` | Builds the image and pushes it to ECR |
| `deploy.py` | Creates, calls and deletes the endpoint (boto3) |
| `cdk/` | The same endpoint as a CDK stack ([cdk/README.md](cdk/README.md)) |

The container runs `strands-decider serve` on port 8080. Besides `/health` and
`/v1/systemone`, the server answers the two paths that SageMaker calls: `GET /ping`, which
is `/health`, and `POST /invocations`, which is `/v1/systemone`.

## What you need

- Docker with BuildKit (`docker buildx`), the AWS CLI v2, and `pip install boto3`.
- Credentials that can push to ECR, and use SageMaker and `iam:PassRole` on the execution role.
- A SageMaker execution role. The image is in ECR, so the role needs
  `AmazonSageMakerFullAccess` or at least ECR pull and CloudWatch Logs.
- Quota for one `ml.g6.xlarge for endpoint usage` in the region (Service Quotas, SageMaker),
  or for the type you pass to `--instance-type`.

## 1. Image

```bash
deploy/sagemaker/build_and_push.sh
```

The script creates the ECR repository `strands-decider-sagemaker` if it is not there,
builds `deploy/sagemaker/Dockerfile` for `linux/amd64` from the repository root, and pushes
it. The last line it prints is the image URI. Set `REGION`, `REPO` or `TAG` to change
where the image goes. To serve another checkpoint, change `CHECKPOINT` in the Dockerfile to
its Hub id. To pin a release, change `CHECKPOINT_REVISION` to its date tag (`YYMMDD`,
[docs/inference.md, Model artifact](../../docs/inference.md#model-artifact)).

The weights are in the image: the checkpoint at `/opt/models/<repo id>`, and the base model
(`Qwen/Qwen3.5-2B-Base`, about 4.5 GB) in the Hugging Face cache at the revision the
checkpoint records. The container sets `HF_HUB_OFFLINE=1`, so the endpoint does not need
internet access and works in a VPC without a NAT gateway.

For a CPU instance, build with `DEVICE=cpu`. That image leaves out flash-linear-attention,
whose Triton kernels need a GPU, and the container runs with `--device cpu`. On CPU the
engine casts the torso to float32 (about 9 GB), so pick an instance with 16 GB or more, such
as `ml.m5.2xlarge` (32 GB):

```bash
DEVICE=cpu TAG=cpu deploy/sagemaker/build_and_push.sh
python deploy/sagemaker/deploy.py create --instance-type ml.m5.2xlarge --image <URI> --role <ARN>
```

On an Apple-silicon Mac the build runs under emulation and is slow. A Linux x86 host, or
AWS CodeBuild, builds it faster.

## 2. Endpoint

```bash
python deploy/sagemaker/deploy.py create \
  --image <image URI from step 1> \
  --role arn:aws:iam::<account>:role/<SageMaker execution role>
```

This creates a model, an endpoint config and an endpoint, all named
`strands-decider-2b-qwen35-v1` (`--name` changes this). Then it waits until the endpoint is
`InService`. `--instance-type` and `--instance-count` set what their names say.
`--serve-args` passes more options to `strands-decider serve`, for example
`--serve-args "--max-batch 8 --strict-window"`. Container logs are in CloudWatch Logs under
`/aws/sagemaker/Endpoints/<name>`.

A quota does not reserve capacity. If the endpoint fails with `InsufficientInstanceCapacity`,
SageMaker found no free instance of that type in the region; no instance ran and nothing
was billed. Delete it (step 4) and try another instance type or region, or try again later.
An endpoint that fails this way can stay in `Creating` for 30 minutes or more first.

## 3. Ask it

`deploy.py invoke` sends a `/v1/systemone` request body, from a file or from stdin, and
prints the response:

```bash
echo '{
  "state": "Help! My payouts have been failing for 3 days!",
  "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}
}' | python deploy/sagemaker/deploy.py invoke
```

From your own code, send the `/v1/systemone` body with `ContentType="application/json"`:

```python
import json, boto3

runtime = boto3.client("sagemaker-runtime")
response = runtime.invoke_endpoint(
    EndpointName="strands-decider-2b-qwen35-v1",
    ContentType="application/json",
    Body=json.dumps({
        "state": "Help! My payouts have been failing for 3 days!",
        "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
    }),
)
print(json.loads(response["Body"].read())["answers"])
```

## 4. Delete it

An endpoint costs money for every hour it runs, whether or not it gets requests. When you
no longer need it:

```bash
python deploy/sagemaker/deploy.py delete
```

This deletes the endpoint, the endpoint config and the model. The image stays in ECR.

## Notes

- The endpoint is private: callers need IAM permission `sagemaker:InvokeEndpoint`. The
  container itself has no authentication, as with `strands-decider serve`.
- The container runs one uvicorn worker, and one instance runs one model. The server's
  behaviour under concurrent requests is not verified
  ([docs/inference.md, Serve](../../docs/inference.md#serve)). For more throughput, add
  instances (`--instance-count`, or auto scaling).
- The image does not include `causal-conv1d`, which compiles against the CUDA toolkit
  ([docs/inference.md](../../docs/inference.md#environments-for-serving)). Transformers uses
  its PyTorch path for that layer instead, which is slower.
