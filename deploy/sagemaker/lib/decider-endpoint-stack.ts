import * as path from "path";
import { CfnOutput, CustomResource, Duration, IgnoreMode, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as ecrAssets from "aws-cdk-lib/aws-ecr-assets";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as s3assets from "aws-cdk-lib/aws-s3-assets";
import * as sagemaker from "aws-cdk-lib/aws-sagemaker";
import * as cr from "aws-cdk-lib/custom-resources";
import { Construct } from "constructs";
import { ContainerImageBuild } from "deploy-time-build";

/**
 * The stack and endpoint name for a model's Hugging Face Hub id: its name, lowercased, with
 * every character a SageMaker endpoint name does not allow replaced by a hyphen.
 * StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 -> strands-decider-2b-qwen3-5-v1-2610.
 */
export function endpointNameFor(model: string): string {
  const name = (model.split("/").pop() ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (!/^[a-z0-9](-*[a-z0-9]){0,62}$/.test(name)) {
    throw new Error(`${model}: cannot make a SageMaker endpoint name (1 to 63 letters, digits, hyphens) from it`);
  }
  return name;
}
const GPU_AMI_VERSION = "al2023-ami-sagemaker-inference-gpu-4-1";

const REPO_ROOT = path.join(__dirname, "..", "..", "..");

// The build uploads this directory to S3; keep only what the image needs, as
// Dockerfile.dockerignore does for a plain docker build.
const ASSET_EXCLUDE = [
  "*",
  "!pyproject.toml",
  "!README.md",
  "!LICENSE",
  "!src/",
  "!deploy/sagemaker/Dockerfile",
  "**/__pycache__",
  "**/*.egg-info",
];

/**
 * The image's device for a SageMaker instance type: `cuda` on NVIDIA GPU families
 * (ml.g4dn, ml.g5, ml.g6, ml.p4d, ...), `cpu` on x86 CPU families. The image is x86 and has
 * no Neuron support, so Graviton (ml.c7g, ml.m6gd, ...) and Inferentia or Trainium types fail.
 */
export function deviceFor(instanceType: string): "cuda" | "cpu" {
  const family = instanceType.replace(/^ml\./, "").split(".")[0];
  if (/^(inf|trn)\d/.test(family)) {
    throw new Error(`${instanceType}: Inferentia and Trainium instances are not supported`);
  }
  if (/^[a-z]+\d+[a-z]*g[a-z]*$/.test(family) && !/^g\d/.test(family)) {
    throw new Error(`${instanceType}: Graviton (arm64) instances are not supported; the image is x86`);
  }
  return /^(g|p)\d/.test(family) ? "cuda" : "cpu";
}

/**
 * Whether an instance type has local NVMe storage. SageMaker keeps the model data there, and
 * rejects VolumeSizeInGB for these types; the others get an EBS volume of the size set below.
 * GPU families from ml.g4dn on, and families with a `d` (ml.m5d, ml.c6id, ...), have it.
 */
export function hasInstanceStorage(instanceType: string): boolean {
  const family = instanceType.replace(/^ml\./, "").split(".")[0];
  return /^(g4dn|g5|g6|g6e|g7|g7e|p4d|p4de|p5|p5e|p5en|p6)/.test(family) || /^[a-z]+\d+[a-z]*d[a-z]*$/.test(family);
}

// Room for the largest model and its base model (about 52 GB for the 26B-A4B), with margin.
const MODEL_VOLUME_GB = 128;

// The custom resource that runs the model download (ModelFetch) and waits for it.
const START_FETCH = `
const { CodeBuildClient, StartBuildCommand } = require("@aws-sdk/client-codebuild");
exports.handler = async (event) => {
  if (event.RequestType === "Delete") return { PhysicalResourceId: event.PhysicalResourceId };
  const p = event.ResourceProperties;
  const { build } = await new CodeBuildClient({}).send(new StartBuildCommand({
    projectName: p.ProjectName,
    environmentVariablesOverride: [
      { name: "MODEL", value: p.Model, type: "PLAINTEXT" },
      { name: "MODEL_REVISION", value: p.ModelRevision, type: "PLAINTEXT" },
    ],
  }));
  return { PhysicalResourceId: p.Model + "@" + p.ModelRevision, Data: { BuildId: build.id } };
};`;
const FETCH_DONE = `
const { CodeBuildClient, BatchGetBuildsCommand } = require("@aws-sdk/client-codebuild");
exports.handler = async (event) => {
  if (event.RequestType === "Delete") return { IsComplete: true };
  const { builds } = await new CodeBuildClient({}).send(new BatchGetBuildsCommand({ ids: [event.Data.BuildId] }));
  const build = builds[0];
  if (build.buildStatus === "IN_PROGRESS") return { IsComplete: false };
  if (build.buildStatus !== "SUCCEEDED") {
    throw new Error("model download " + build.id + " " + build.buildStatus + ": " + (build.logs && build.logs.deepLink));
  }
  const vars = Object.fromEntries((build.exportedEnvironmentVariables || []).map((v) => [v.name, v.value]));
  return { IsComplete: true, Data: { ModelS3Uri: vars.MODEL_S3_URI } };
};`;

/**
 * The strands-decider SageMaker endpoint (README.md): a private bucket with the model, the image,
 * the execution role, the SageMaker model, the endpoint config and the endpoint. Settings are CDK
 * context values (cdk.json, or -c key=value).
 */
export interface DeciderEndpointStackProps extends StackProps {
  /** The endpoint name, from endpointNameFor(model). */
  readonly endpointName: string;
}

export class DeciderEndpointStack extends Stack {
  constructor(scope: Construct, id: string, props: DeciderEndpointStackProps) {
    super(scope, id, props);
    const endpointName = props.endpointName;
    const ctx = (key: string) => this.node.tryGetContext(key);
    const modelId: string = ctx("model");
    const instanceType: string = ctx("instance_type");
    const device = deviceFor(instanceType);

    // The model, copied from the Hugging Face Hub into a private bucket during `cdk deploy`.
    // SageMaker copies it from there to /opt/ml/model on each instance (ModelDataSource below).
    const bucket = new s3.Bucket(this, "Models", {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });
    const fetchScript = new s3assets.Asset(this, "FetchScript", {
      path: path.join(__dirname, ".."),
      exclude: ["*", "!fetch_model.py"],
    });
    // fetch_model.py writes the model and its base model to s3://<bucket>/<endpoint>/<commit>/.
    // A commit already there is not downloaded again.
    const fetch = new codebuild.Project(this, "ModelFetch", {
      source: codebuild.Source.s3({ bucket: fetchScript.bucket, path: fetchScript.s3ObjectKey }),
      environment: {
        buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
        computeType: codebuild.ComputeType.LARGE, // 128 GB of disk for the largest models
      },
      timeout: Duration.hours(2),
      environmentVariables: {
        BUCKET: { value: bucket.bucketName },
        PREFIX: { value: endpointName },
      },
      buildSpec: codebuild.BuildSpec.fromObject({
        version: "0.2",
        env: { "exported-variables": ["MODEL_S3_URI"] },
        phases: {
          install: {
            commands: [
              'pip install --quiet "huggingface_hub>=1.5"',
              "aws configure set default.s3.max_concurrent_requests 64",
            ],
          },
          build: {
            commands: [
              'commit=$(python fetch_model.py --commit "$MODEL" "$MODEL_REVISION")',
              'export MODEL_S3_URI="s3://$BUCKET/$PREFIX/$commit/"',
              'if aws s3 ls "s3://$BUCKET/$PREFIX/$commit.complete" > /dev/null; then echo "$MODEL@$commit is already in $MODEL_S3_URI"; else ' +
                'python fetch_model.py "$MODEL" "$commit" "$CODEBUILD_SRC_DIR/model" && ' +
                'aws s3 sync "$CODEBUILD_SRC_DIR/model" "$MODEL_S3_URI" --only-show-errors && ' +
                'echo "$MODEL@$commit" | aws s3 cp - "s3://$BUCKET/$PREFIX/$commit.complete"; fi',
            ],
          },
        },
      }),
    });
    bucket.grantReadWrite(fetch);
    fetchScript.grantRead(fetch);

    const handler = (name: string, code: string) =>
      new lambda.Function(this, name, {
        runtime: lambda.Runtime.NODEJS_22_X,
        handler: "index.handler",
        code: lambda.Code.fromInline(code),
        timeout: Duration.minutes(1),
      });
    const startFetch = handler("StartModelFetch", START_FETCH);
    const fetchDone = handler("ModelFetchDone", FETCH_DONE);
    startFetch.addToRolePolicy(new iam.PolicyStatement({ actions: ["codebuild:StartBuild"], resources: [fetch.projectArn] }));
    fetchDone.addToRolePolicy(new iam.PolicyStatement({ actions: ["codebuild:BatchGetBuilds"], resources: [fetch.projectArn] }));
    const fetchProvider = new cr.Provider(this, "ModelFetchProvider", {
      onEventHandler: startFetch,
      isCompleteHandler: fetchDone,
      queryInterval: Duration.minutes(1),
      totalTimeout: Duration.hours(2),
    });
    const modelData = new CustomResource(this, "ModelData", {
      serviceToken: fetchProvider.serviceToken,
      properties: { ProjectName: fetch.projectName, Model: modelId, ModelRevision: ctx("model_revision") },
    });

    const role = new iam.Role(this, "ExecutionRole", {
      assumedBy: new iam.ServicePrincipal("sagemaker.amazonaws.com"),
    });
    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
        resources: [`arn:aws:logs:${this.region}:${this.account}:log-group:/aws/sagemaker/*`],
      }),
    );
    role.addToPolicy(new iam.PolicyStatement({ actions: ["cloudwatch:PutMetricData"], resources: ["*"] }));
    bucket.grantRead(role);

    // Builds the image in AWS CodeBuild during `cdk deploy` (deploy-time-build) and publishes it
    // to an ECR repository that belongs to the stack. The image holds no model weights.
    const image = new ContainerImageBuild(this, "Image", {
      directory: REPO_ROOT,
      file: "deploy/sagemaker/Dockerfile",
      exclude: ASSET_EXCLUDE,
      ignoreMode: IgnoreMode.DOCKER,
      buildArgs: { DEVICE: device },
      platform: ecrAssets.Platform.LINUX_AMD64,
    });
    image.repository.grantPull(role);

    const model = new sagemaker.CfnModel(this, "Model", {
      executionRoleArn: role.roleArn,
      primaryContainer: {
        image: image.imageUri,
        modelDataSource: {
          s3DataSource: {
            s3Uri: modelData.getAttString("ModelS3Uri"),
            s3DataType: "S3Prefix",
            compressionType: "None",
          },
        },
        environment: {
          // fetch_model.py's layout: /opt/ml/model/<model name>/ and /opt/ml/model/base/.
          DECIDER_CHECKPOINT: `/opt/ml/model/${modelId.split("/").pop()}`,
          DECIDER_DEVICE: device,
          DECIDER_ARGS: ctx("serve_args") ?? "",
        },
      },
    });
    // The role's policy must exist before SageMaker checks that it can pull the image.
    model.node.addDependency(role);

    const config = new sagemaker.CfnEndpointConfig(this, "EndpointConfig", {
      productionVariants: [
        {
          variantName: "AllTraffic",
          modelName: model.attrModelName,
          instanceType,
          initialInstanceCount: Number(ctx("instance_count")),
          initialVariantWeight: 1,
          // GPU hosts run Amazon Linux 2023 (NVIDIA driver 580, which runs the image's CUDA 12.6).
          // SageMaker has no AL2023 option for CPU hosts, so they keep its default AMI.
          inferenceAmiVersion: device === "cuda" ? GPU_AMI_VERSION : undefined,
          volumeSizeInGb: hasInstanceStorage(instanceType) ? undefined : MODEL_VOLUME_GB,
          // Copying up to about 52 GB from S3, then loading it (about 100 GB of float32 weights
          // on CPU for the largest model), takes minutes.
          modelDataDownloadTimeoutInSeconds: 3600,
          containerStartupHealthCheckTimeoutInSeconds: 1800,
        },
      ],
    });

    const endpoint = new sagemaker.CfnEndpoint(this, "Endpoint", {
      endpointName,
      endpointConfigName: config.attrEndpointConfigName,
    });

    new CfnOutput(this, "EndpointName", { value: endpoint.attrEndpointName });
    new CfnOutput(this, "Device", { value: device });
    new CfnOutput(this, "ModelDataS3Uri", { value: modelData.getAttString("ModelS3Uri") });
    new CfnOutput(this, "InvokeUrl", {
      value: `https://runtime.sagemaker.${this.region}.amazonaws.com/endpoints/${endpointName}/invocations`,
    });
  }
}
