import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
import gymnasium as gym
import torch
import os
from policy import PolicyModule
from environment import BettingEnv
from policy import PolicyModule

os.environ["HF_HUB_OFFLINE"] = "1"  

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATA_DIR = "/home/jcdutoit/Projects/probability-aligned-llms/data"

ENV_CONFIG = {
    "data_dir": DATA_DIR,
    "max_action_length": 512,
    "max_observation_length": 256,
}

def main():
    ray.init(runtime_env={
        "working_dir": ".",
        "excludes": ["models", "data"],
    })
    
    env = BettingEnv(config=ENV_CONFIG)

    rl_module_spec = RLModuleSpec(
        module_class=PolicyModule,
        action_space=env.action_space,
        observation_space=env.observation_space,
        model_config={
            "model_id": MODEL_ID,
        }
    )

    config = (
        PPOConfig()
        .environment(env=BettingEnv, env_config=ENV_CONFIG)
        .rl_module(rl_module_spec=rl_module_spec)
        .training(
            train_batch_size=1,
            minibatch_size=1,
            num_epochs=1,
            lr=1e-5,
        )
    )

    algo = config.build()

    num_iterations = 1
    for i in range(num_iterations):
        result = algo.train()
        print(f"Iteration {i + 1}: reward_mean = {result}")

    algo.save("ppo_betting_model")
    algo.stop()

if __name__ == "__main__":
    main()