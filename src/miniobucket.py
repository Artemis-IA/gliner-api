# from gliner import GLiNER
# import json

# model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
# config = model.config.to_dict()
# print(json.dumps(config, indent=4))


from minio import S3Error
from minio import Minio

# Initialize Minio client
minio_client = Minio(
    "play.min.io",
    access_key="minio",
    secret_key="minio123",
    secure=True
)

try:
    bucket_name = "datasets"
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
except S3Error as e:
    print(f"MinIO Error: {e}")
