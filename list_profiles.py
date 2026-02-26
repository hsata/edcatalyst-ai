import boto3, json

client = boto3.client("bedrock", region_name="us-east-1")

try:
    resp = client.list_inference_profiles()
    print(json.dumps(resp, indent=2, default=str))
except Exception as e:
    print("Error:", e)
