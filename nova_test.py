import os
import boto3

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MODEL_ID = os.environ["amazon.nova-2-lite-v1:0"]

client = boto3.client("bedrock-runtime", region_name=REGION)

resp = client.converse(
    modelId=MODEL_ID,
    messages=[
        {"role": "user", "content": [{"text": "Say hello in one short sentence and mention EdCatalyst AI."}]}
    ],
    inferenceConfig={"maxTokens": 80, "temperature": 0.2},
)

print(resp["output"]["message"]["content"][0]["text"])
