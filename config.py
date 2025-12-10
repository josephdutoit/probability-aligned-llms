from dataclasses import dataclass

@dataclass
class Config:
    gen_device: int = 1
    data_dir: str = "./data"
    model_path: str = "Qwen/Qwen2.5-3B"
    beta: float = 0.04
    all_steps: int = 500
    Q_batch_size: int = 1
    num_pre_Q: int = 3
    train_batch_size: int = 1
    gen_update_steps: int = 6
    save_steps: int = 50
    compute_gen_logps: bool = True
    clip_param: float = 0.2
    ref_server: str | None = None
    ds_config: dict = None

    def __post_init__(self):
        self.ds_config = {
            "train_micro_batch_size_per_gpu": self.train_batch_size,
            "gradient_accumulation_steps": 4,
            "bf16": {"enabled": True},
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": 2,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
                "stage3_gather_16bit_weights_on_model_save": True,
            }
        }

    def set_ref_server(self, port: int):
        self.ref_server = f"https://localhost:{port}"