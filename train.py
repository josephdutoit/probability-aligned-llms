from vllm.inputs import TokensPrompt
import json, os, random, requests, time
import torch

from rewards import compute_rewards
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

def get_batch(cfg):
    try:
        r = requests.get(f"{cfg.ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    if len(dd) == 5: data['gen_logps'] = bytes_to_tensor(dd[4])
    return data

def get_per_token_logps(logits, input_ids):
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

def GRPO_step(batch, pad_token_id, cfg, engine):
    """Compute GRPO loss for a batch.
    
    Args:
        batch: dict with keys:
            'plen': (int) length of prompt (non-generated part)
            'inputs': (torch.LongTensor) input token IDs, shape (B, L)
            'rewards': (torch.FloatTensor) rewards for each sequence, shape (B,)
            'refs': (torch.FloatTensor) reference log probabilities, shape (B, L-prompt_length)
            'gen_logps' (optional): (torch.FloatTensor) generated log probabilities, shape
                (B, L-prompt_length), needed if compute_gen_logps is True. In this lab, it will always be provided.
        pad_token_id: (int) token ID used for padding, needed to create completion mask
    Returns:
        loss: (torch.FloatTensor) scalar GRPO loss for the batch
    Note:
        - Assumes `engine` and `beta`, `clip_param`, `compute_gen_log
        - Uses `get_per_token_logps` to compute log probabilities
        - Implements GRPO loss with optional PPO-style clipping
        - Averages loss over completion tokens and batch
        - Uses `pad_token_id` to create mask for completion tokens
        - `engine` is the DeepSpeed engine with the model, it will be neccesary to move tensors to engine.device
        - ~ 20 lines of code (excluding comments)
    Example Input:
            batch inputs shape: torch.Size([1, 196])
            batch rewards shape: torch.Size([1])
            batch refs shape: torch.Size([1, 22])
            batch gen_logps shape: torch.Size([1, 22])
    """
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1) # (B, 1)
    logits = engine(inputs).logits
    B, L, V = logits.shape
    logits = logits[:, :-1, :]  # (B, L-1, V)
    input_ids = inputs[:, 1:]  # (B, L-1)
    refs_per_token_logps = batch['refs']
    gen_logps = batch.get('gen_logps', None)
    assert gen_logps is not None
    
    gen_logps = gen_logps.to(engine.device)
    new_logps = get_per_token_logps(logits, input_ids)
    new_logps = new_logps.to(engine.device)
    # 2. Slice to keep only completion tokens (after prompt)
    new_logps = new_logps[:, prompt_length - 1:]
    sliced_input_ids = input_ids[:, prompt_length - 1:]

    # 3. Move reference log probabilities to the same device as per_token_logps
    refs_per_token_logps.to(engine.device)
    
    # 4. Compute per-token KL divergence approximation for regularization
    kl = torch.exp(refs_per_token_logps - new_logps) - (refs_per_token_logps - new_logps) - 1

    # 5. Create mask for completion tokens (not padding)
    mask = sliced_input_ids != pad_token_id

    # 6. Compute importance sampling ratio
    ratio = torch.exp(new_logps - gen_logps)

    # 7. Clip ratio for PPO-style loss
    clipped_ratio = torch.clamp(ratio, 1 - cfg.clip_param, 1 + cfg.clip_param)
    
    # 8. Compute per-token GRPO loss
    loss = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    
    # 9. Average loss over completion tokens and batch
    loss = loss * mask
    kl = kl * mask
    loss = (loss - cfg.beta * kl).sum(dim=-1) / mask.sum()
    
    # 10. Return final loss 
    return -loss
    

def gen_worker(Q, cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{cfg.gen_device}'
    torch.cuda.set_device(0)
    print(f"Generation worker process uses GPU {cfg.gen_device}")
    from vllm import LLM, SamplingParams
    vllm_gen = LLM(model=cfg.model_path, gpu_memory_utilization=0.3, max_model_len=4096, dtype="float16")
    vllm_tokenizer = vllm_gen.get_tokenizer()
    ref_server_ver = 'tensor'  # don't worry, it will auto switch based on the first upload

    sampling_params = SamplingParams(n=cfg.num_pre_Q, temperature=0.9, max_tokens=700)
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

    from dataset import load_qs
    Qs = load_qs(cfg.data_dir)
    
    system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a probability question, and based on your knowledge, you provide the probability (e.g., 0.75 or 75%). Ensure your answer is clear and concise."""
    
    def make_chat_prompt(question):
        return vllm_tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    
    def gen_answers(prompts):
        tip_text = []
        for x in prompts:
            tip_text.append(make_chat_prompt(x[0]))
            tip_text.append(make_chat_prompt(x[1]))
        voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)
        answers = [];  ans_token_ids = []
        for v in voutputs:
            for z in v.outputs: 
                answers.append(z.text)
                ans_token_ids.append(z.token_ids)
        return answers, ans_token_ids

    def gen_samples(inputs):
        print(f"Generating answers for the following inputs: {inputs}")
        answers, ans_token_ids = gen_answers(inputs)
        print('Generated answers:', answers)
        rewards = []
        for i in range(len(inputs)):
            # for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
            #     rewards.append(reward_correct(inp, a) + reward_format(inp, a))
            # TODO: Implement rewards for these -- they should look at every 2 answers together
            for a in range(0, 2 * cfg.num_pre_Q, 2):
                reward = compute_rewards(
                    answers[i + a], 
                    answers[i + a + 1],
                )
                rewards.append(reward)
                rewards.append(reward)

        prompts_text = [vllm_tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in inputs]
        return prompts_text, torch.tensor(rewards, dtype=torch.float32), answers, ans_token_ids

    def try_update_model():
        try:
            new_state_dict = Q.get_nowait()
            print('[VLLM PROC] recving new model ...')
            llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(new_state_dict.items())
            print('[VLLM PROC] model updated')
            del new_state_dict
        except:
            #print('[VLLM PROC] no new model')
            return
        
    from torch.nn.utils.rnn import pad_sequence
    for it in range(999999999):
        if it % 3 == 0: try_update_model()
        inputs = random.sample(Qs,  cfg.Q_batch_size)
        inputs = [inp for tup in inputs for inp in tup]
        tic = time.time()
        prompt_inputs, rewards, answers, ans_token_ids = gen_samples(inputs)
        print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards, )
        if it % 5 == 0: print('answers:', answers[0])

        for i, pp in enumerate(prompt_inputs):
            prompt_ids = vllm_tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
            plen = prompt_ids.shape[1]
            curr_answers = answers[i*cfg.num_pre_Q:(i+1)*cfg.num_pre_Q]
            curr_ans_ids = ans_token_ids[i*cfg.num_pre_Q:(i+1)*cfg.num_pre_Q]
            curr_rewards = rewards[i*cfg.num_pre_Q:(i+1)*cfg.num_pre_Q]
            if curr_rewards.max() - curr_rewards.min() < 1e-4: continue

            if ref_server_ver == 'tensor':
                # Preserve unnormalized rewards for logging/analytics
                curr_rewards_unnorm = curr_rewards.clone()
                # Compute format-only accuracy for each generated answer in this group
                # 1 if matches required <think>...</think><answer>...</answer> format, else 0
                # fmt_flags = []
                # for a in curr_answers:
                #     fmt_flags.append(1 if reward_format(None, a) > 0 else 0)
                # Normalize rewards for training stability
                curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
                for ii in range(0, cfg.num_pre_Q, cfg.train_batch_size):
                    sub_rewards = curr_rewards[ii:ii+cfg.train_batch_size]
                    sub_rewards_unnorm = curr_rewards_unnorm[ii:ii+cfg.train_batch_size]
                    sub_ans_ids = curr_ans_ids[ii:ii+cfg.train_batch_size]
                    # sub_fmt_flags = fmt_flags[ii:ii+train_batch_size]
                    tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                    output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=vllm_tokenizer.pad_token_id) 
                    Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)
                    merged_ids = torch.cat([Qrep, output_ids], dim=1)
                    # Include unnormalized rewards and format accuracy in JSON header for training-side logging
                    data = [json.dumps({
                                "plen": plen,
                                "unnormalized_rewards": sub_rewards_unnorm.tolist(),
                                # "format_accuracy": float(np.mean(sub_fmt_flags)) if len(sub_fmt_flags) > 0 else None
                            }).encode(),
                            tensor_to_bytes(merged_ids),
                            tensor_to_bytes(sub_rewards)]       

                    if cfg.compute_gen_logps:
                        zz = vllm_gen.generate(
                            TokensPrompt(prompt_token_ids=merged_ids[0].tolist()),
                            sampling_params=gen_logps_sp,
                            use_tqdm=False
                        )
                        zz = [xx.prompt_logprobs[plen:] for xx in zz]
                        gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                        data.append(tensor_to_bytes(gen_logps))

                    xdata = make_bytes_list(data)
                    r = requests.post(f"{cfg.ref_server}/upload", data=xdata)
                    if r.content == b'string': ref_server_ver = 'string'
            elif ref_server_ver == 'string':
                xdata = make_bytes_list([json.dumps({"Q": pp[0], "As": curr_answers}).encode(), 
                                        tensor_to_bytes(curr_rewards)])
                r = requests.post(f"{cfg.ref_server}/upload", data=xdata)
                if r.content == b'tensor': ref_server_ver = 'tensor'
