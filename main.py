import argparse
import deepspeed
import numpy as np
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

from config import Config
from train import get_batch, GRPO_step, gen_worker

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--config", type=str, required=False)
    args, unknown = parser.parse_known_args()
    
    if args.config:
        cfg = Config.from_json(args.config)
    else:
        cfg = Config()

    cfg.set_ref_server(args.port)

    deepspeed.init_distributed()

    if dist.get_rank() == 0:
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn')
        Q = mp.Queue()
        p = mp.Process(target=gen_worker, args=(Q, cfg))
        p.start()

    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, 
            torch_dtype=torch.bfloat16, _attn_implementation="sdpa")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    engine, optimizer, _, _ = deepspeed.initialize(
        config=cfg.ds_config,
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer
    )

    if dist.get_rank() == 0:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "grpo-math"),
            name=os.environ.get("WANDB_RUN_NAME", f"{os.path.basename(__file__)}"),
            config={
                "model_path": cfg.model_path,
                "beta": cfg.beta,
                "all_steps": cfg.all_steps,
                "Q_batch_size": cfg.Q_batch_size,
                "num_pre_Q": cfg.num_pre_Q,
                "train_batch_size": cfg.train_batch_size,
                "gen_update_steps": cfg.gen_update_steps,
                "clip_param": cfg.clip_param,
                "compute_gen_logps": cfg.compute_gen_logps,
            },
        )

    progress = range(1, cfg.all_steps+1)
    if dist.get_rank() == 0: progress = tqdm(progress)
    for step in progress:
        batch = get_batch(cfg)
        while batch is None:
            print('waiting for batch...'); time.sleep(1)
            batch = get_batch(cfg)

        fallback_tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        loss = GRPO_step(
            batch, 
            pad_token_id=fallback_tokenizer.pad_token_id, 
            cfg=cfg,
            engine=engine
        )
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            current_loss = loss.item()
            current_reward = np.mean(batch.get('unnormalized_rewards', [0.0]))
            current_format_acc = batch.get('format_accuracy', None)
            progress.set_description(f"Loss: {current_loss:.6f}, Avg Reward: {current_reward:.3f}")
            log_payload = {
                "train/loss": current_loss,
                "train/avg_reward_raw": current_reward,
                "train/step": step,
            }
            if current_format_acc is not None:
                log_payload["train/format_accuracy"] = current_format_acc
            wandb.log(log_payload)

        if step % cfg.gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('[TRAINING PROC] sending latest state_dict ...')
                state_dict = engine.module.state_dict()
                Q.put(state_dict)
                print('[TRAINING PROC] send state_dict ok!')
            dist.barrier()

    # Finalize Weights & Biases
    if dist.get_rank() == 0:
        wandb.finish()


if __name__ == "__main__":
    main()