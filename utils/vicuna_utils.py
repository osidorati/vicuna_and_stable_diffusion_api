import gc

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

from utils.compression import compress_module
from utils.monkey_patch_non_inplace import replace_llama_attn_with_non_inplace_operations


def load_model(model_name, device, num_gpus, load_8bit=False, debug=False):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if load_8bit:
            if num_gpus != "auto" and int(num_gpus) != 1:
                print("8-bit weights are not supported on multiple GPUs. Revert to use one GPU.")
            kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        else:
            if num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                num_gpus = int(num_gpus)
                if num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: "14GiB" for i in range(num_gpus)},
                    })
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        low_cpu_mem_usage=True, **kwargs)

    # calling model.cuda() mess up weights if loading 8-bit weights
    if device == "cuda" and num_gpus == 1 and not load_8bit:
        model.to("cuda")
    elif device == "mps":
        model.to("mps")

    if (device == "mps" or device == "cpu") and load_8bit:
        compress_module(model, device)

    return model, tokenizer


@torch.inference_mode()
def generate_stream(tokenizer, model, params, device,
                    context_len=2048, stream_interval=2):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 512))
    stop_str = params.get("stop", None)

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device)
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            pos = output.rfind(stop_str, l_prompt)
            if pos != -1:
                output = output[:pos]
                stopped = True
            yield output

        if stopped:
            break
    torch.cuda.empty_cache()
    gc.collect()
    del past_key_values