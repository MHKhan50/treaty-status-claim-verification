"""Create (if needed) and update the static Hugging Face Space from ``space/``.

Usage: python scripts/upload_space.py [owner/space-name]
Needs a Hugging Face login with write access (``hf auth login``).
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo_id = sys.argv[1] if len(sys.argv) > 1 else "MHkhan50/nuclear-treaty-claim-check"
folder = Path(__file__).resolve().parents[1] / "space"
api = HfApi()
api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(folder),
                  allow_patterns=["index.html", "README.md"], commit_message="Update app")
print(f"https://huggingface.co/spaces/{repo_id}")
