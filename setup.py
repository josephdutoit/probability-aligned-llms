"""
Run this script ONLY on the login node (not a compute node).
It downloads model weights and datasets needed for training.
"""

import socket
import os
from huggingface_hub import snapshot_download

def check_internet(host="huggingface.co", port=443, timeout=3):
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except Exception:
        return False

repo_id = "Qwen/Qwen3-4B"

# Specify your desired save path here (adjust as needed)
save_path = f"/home/jcdutoit/Projects/probability-aligned-llms/{repo_id.replace('/', os.sep)}"  # Example: local directory

if not check_internet():
    print("WARNING: No internet connectivity. Downloads will fail. Run this on the login node with internet access.")
else:
    # Download the full model repository (weights + tokenizer) to the specified path
    print(f"Downloading model and tokenizer to: {save_path}")
    snapshot_download(repo_id=repo_id, local_dir=save_path)
    
    # Verify by loading from the local path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(save_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(save_path, local_files_only=True)
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    print("Download and verification complete.")
