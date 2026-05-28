"""
Auto-launch EC2 GPU instance and start training when quota is approved.
Run: python3 scripts/launch_aws.py
"""
import boto3, json, time, os, subprocess

AWS_REGION = 'us-east-1'
INSTANCE_TYPE = 'g4dn.xlarge'
AMI_ID = 'ami-015fb7b9618ada2ff'  # Deep Learning AMI
KEY_NAME = 'math-ai-key'
SG_ID = 'sg-04a10b585df3f9463'

ec2 = boto3.client('ec2', region_name=AWS_REGION)

def check_quota():
    sq = boto3.client('service-quotas', region_name=AWS_REGION)
    quota = sq.get_service_quota(ServiceCode='ec2', QuotaCode='L-DB2E81BA')
    return quota['Quota']['Value']

def launch_instance():
    resp = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        SecurityGroupIds=[SG_ID],
        MinCount=1,
        MaxCount=1,
        BlockDeviceMappings=[{
            'DeviceName': '/dev/sda1',
            'Ebs': {'VolumeSize': 80, 'VolumeType': 'gp3'}
        }],
        TagSpecifications=[{
            'ResourceType': 'instance',
            'Tags': [{'Key': 'Name', 'Value': 'math-ai-training'}]
        }],
    )
    return resp['Instances'][0]['InstanceId']

def wait_for_instance(instance_id):
    print(f"Waiting for {instance_id} to be running...")
    waiter = ec2.get_waiter('instance_running')
    waiter.wait(InstanceIds=[instance_id])
    info = ec2.describe_instances(InstanceIds=[instance_id])
    return info['Reservations'][0]['Instances'][0].get('PublicIpAddress')

def upload_and_train(ip):
    print(f"Instance IP: {ip}")
    print("Waiting 60s for instance to fully initialize...")
    time.sleep(60)
    
    # Upload training script
    key_path = '/tmp/math-ai-key.pem'
    subprocess.run([
        'scp', '-i', key_path, '-o', 'StrictHostKeyChecking=no',
        '/workspaces/Ashua/colab_train.py', f'ubuntu@{ip}:/home/ubuntu/train.py'
    ], check=True)
    
    # Start training
    subprocess.run([
        'ssh', '-i', key_path, '-o', 'StrictHostKeyChecking=no',
        f'ubuntu@{ip}',
        'nohup python3 /home/ubuntu/train.py > /home/ubuntu/train.log 2>&1 &'
    ], check=True)
    
    print(f"\nTraining started on {ip}")
    print(f"SSH: ssh -i {key_path} ubuntu@{ip}")
    print(f"Logs: ssh -i {key_path} ubuntu@{ip} 'tail -f /home/ubuntu/train.log'")

if __name__ == '__main__':
    print("Checking GPU quota...")
    quota = check_quota()
    print(f"Current quota: {quota}")
    
    if float(quota) < 4:
        print("Quota not yet approved. Run this script again later.")
        exit(1)
    
    print("Quota approved! Launching instance...")
    instance_id = launch_instance()
    print(f"Instance: {instance_id}")
    
    ip = wait_for_instance(instance_id)
    print(f"Running at: {ip}")
    
    upload_and_train(ip)
