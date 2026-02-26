import boto3, json

client = boto3.client("bedrock", region_name="us-east-1")
resp = client.list_inference_profiles()

print("==== RAW RESPONSE KEYS ====")
print(list(resp.keys()))

print("\n==== PROFILES ====")

# Find the correct list key dynamically
profiles = None
for key in resp:
    if isinstance(resp[key], list):
        profiles = resp[key]
        break

if not profiles:
    print("No profile list found.")
    exit()

for p in profiles:
    print("\n----------------------------")
    print("ID:", p.get("inferenceProfileId"))
    print("ARN:", p.get("inferenceProfileArn"))
    print("TYPE:", p.get("type"))
    print("STATUS:", p.get("status"))
