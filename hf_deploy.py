import os
from huggingface_hub import HfApi, login

import sys

# Read token from command line args
token = sys.argv[1]
login(token=token)

api = HfApi()
user = api.whoami()["name"]
repo_id = f"{user}/OpenEnv-Simulator"

print(f"Creating Space {repo_id}...")
try:
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
    )
    print("Space created successfully.")
except Exception as e:
    print(f"Space may already exist: {e}")

print("Uploading files to Space...")
# Upload the entire current directory to the space
api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=[
        "*.pyc",
        "__pycache__/*",
        ".git/*",
        ".env",
        ".venv/*",
        "tmp/*",
        ".gemini/*",
        "*.log",
        "*_out.txt",
        "hf_deploy.py"
    ]
)
print(f"Deployment complete! Check your space at: https://huggingface.co/spaces/{repo_id}")
