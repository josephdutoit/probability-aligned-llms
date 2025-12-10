"""
Run this script ONLY on the login node (not a compute node).
It downloads model weights and datasets needed for training.
"""

import os
import socket

def check_internet(host="huggingface.co", port=443, timeout=3):
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except Exception:
        return False

if not check_internet():
    print("WARNING: No internet connectivity. Downloads will fail. Run this on the login node with internet access.")
else:
    # Specify local cache directory (e.g., ./models)
    cache_dir = "./models"
    os.makedirs(cache_dir, exist_ok=True)
    
    print("Downloading model and tokenizer to:", cache_dir)
    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', cache_dir=cache_dir, trust_remote_code=True)
    
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct', cache_dir=cache_dir, trust_remote_code=True)
    
    print("Download complete. Files in:", cache_dir)

