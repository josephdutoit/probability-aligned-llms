import torch
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_DIR = "/home/jcdutoit/Projects/probability-aligned-llms/models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"

class PolicyModule(TorchRLModule):
    def setup(
        self,      
    ):
        super().setup()

        model_id = self.model_config.get("model_id", "Qwen/Qwen2.5-3B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)

        self.messages = [{
            "role": "system", 
            "content": "You are an assistant tasked with giving probability estimates for user queries."
        }]

        self.gen_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )


    def _forward(
            self,
            q1: str,
            q2: str,
    ):
        return {
            Columns.ACTIONS: (self._answer_query(q1), self._answer_query(q2))
        }

    def _answer_query(
            self,
            query: str,
    ) -> str:
        msg = self.messages + [{"role": "user", "content": query}]

        input_text = self.tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=True, 
        )

        outputs = self.gen_pipe(
            input_text,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            return_full_text=False,
        )

        generated_text = outputs[0]["generated_text"] if outputs else ""
        return generated_text