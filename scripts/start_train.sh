export REF_SERVER_PORT=${REF_SERVER_PORT:-56712}

module load cuda/12.4.1-pw6cogp

source .venv/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE="offline"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /tmp/triton_cache
export TRITON_CACHE_DIR="/tmp/triton_cache"

# Start the reference server in the background, passing the port
# bash scripts/start_ref_server.sh $REF_SERVER_PORT > logs/ref_server.log 2>&1 &

# Wait for the server to start
# sleep 60

# Start the GRPO process (using 3 GPUs), passing the port
deepspeed --include localhost:2 grpo_lab.py --ref-server-port $REF_SERVER_PORT > logs/grpo.log 2>&1