import json
import boto3

# Initialize the SSM client; ensure the region matches your EC2 instance's region.
ssm_client = boto3.client('ssm', region_name='us-east-1')

# env-variable: EC2_INSTANCE_ID
EC2_INSTANCE_IDS = ['EC2_INSTANCE_ID']

def lambda_handler(event, context):
    print("Received event:", json.dumps(event))
    
    # Log details about the S3 change event.
    for record in event.get('Records', []):
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        print(f"Change detected in bucket: {bucket}, object: {key}")
    
    # Update the sync command to target the Docker volume's data directory.
    # Docker volumes are stored under /var/lib/docker/volumes/<volume_name>/_data.
    sync_command = "sudo aws s3 sync s3://resume-storage-bucket-1/resume/ /var/lib/docker/volumes/resume-volume/_data"
    
    # (Optional) To ensure your container picks up the changes, you could restart it.
    # For example, you could append: " && docker restart resume-app"
    # sync_command += " && docker restart resume-app"
    
    try:
        response = ssm_client.send_command(
            InstanceIds=EC2_INSTANCE_IDS,
            DocumentName="AWS-RunShellScript",
            Parameters={
                "commands": [sync_command]
            },
            TimeoutSeconds=60,
        )
        command_id = response['Command']['CommandId']
        print("SSM Command sent successfully. Command ID:", command_id)
    except Exception as e:
        print("Error sending SSM command:", str(e))
        raise e
    
    return {
        'statusCode': 200,
        'body': json.dumps('SSM sync command executed successfully')
    }
