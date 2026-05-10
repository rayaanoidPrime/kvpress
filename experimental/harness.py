# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from collections import Counter

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_PATHS = {
    "TinyLlama": r"C:\Users\sengu\kvpress\.models\TinyLlama-1.1B-Chat-v1.0",
    "Qwen": r"C:\Users\sengu\kvpress\.models\Qwen2.5-0.5B-Instruct",
}

MODEL_ARCH = {
    "TinyLlama": {"n_layers": 22, "n_kv_heads": 4, "head_dim": 64},
    "Qwen": {"n_layers": 24, "n_kv_heads": 8, "head_dim": 64},
}


def load_model(model_name: str):
    path = MODEL_PATHS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map="cpu"
    ).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def tokenize(tokenizer, text: str, max_length: int = 1024) -> Tensor:
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_length)
    return torch.tensor(tokens, dtype=torch.long, device="cpu").unsqueeze(0)


def measure_perplexity(
    model, tokenizer, context: str, press=None, max_length: int = 512
) -> dict:
    prefix_len = max_length // 2
    cont_len = max_length // 2

    prefix_ids = tokenize(tokenizer, context, max_length=prefix_len)
    prefix_ids = prefix_ids[:, :prefix_len]
    n_prefix = prefix_ids.shape[1]

    cont_ids = tokenize(tokenizer, context, max_length=cont_len)
    cont_ids = cont_ids[:, :cont_len]
    n_cont = cont_ids.shape[1]

    t0 = time.perf_counter()

    with torch.no_grad():
        cache = DynamicCache()
        if press is not None:
            with press(model):
                outputs = model(
                    prefix_ids,
                    past_key_values=cache,
                    cache_position=torch.arange(n_prefix, device="cpu"),
                    use_cache=True,
                )
        else:
            outputs = model(
                prefix_ids,
                past_key_values=cache,
                cache_position=torch.arange(n_prefix, device="cpu"),
                use_cache=True,
            )

        cont_cache_pos = torch.arange(n_prefix, n_prefix + n_cont, device="cpu")
        outputs = model(
            cont_ids,
            past_key_values=cache,
            cache_position=cont_cache_pos,
            use_cache=True,
            labels=cont_ids,
        )
        loss = outputs.loss.item()

    dt = (time.perf_counter() - t0) * 1000.0

    avg_kv = 0
    n_layers = 0
    for layer_cache in cache.layers:
        if hasattr(layer_cache, "keys") and isinstance(layer_cache.keys, torch.Tensor):
            avg_kv += layer_cache.keys.shape[2]
            n_layers += 1
    avg_kv = avg_kv / max(n_layers, 1) if n_layers > 0 else 0

    import math

    ppl = math.exp(loss)

    return {
        "perplexity": ppl,
        "loss": loss,
        "n_prefix_tokens": n_prefix,
        "n_continuation_tokens": n_cont,
        "inference_ms": dt,
        "avg_kv_entries": avg_kv,
    }


def needle_in_haystack(
    model,
    tokenizer,
    haystack_text: str,
    needle_fact: str,
    question: str,
    expected_answer: str,
    needle_position: float = 0.5,
    max_context_tokens: int = 512,
    max_new_tokens: int = 32,
    press=None,
) -> dict:
    words = haystack_text.split()
    insert_idx = int(len(words) * needle_position)
    words_with_needle = words[:insert_idx] + needle_fact.split() + words[insert_idx:]
    full_text = " ".join(words_with_needle)

    context_ids = tokenize(tokenizer, full_text, max_length=max_context_tokens)
    context_ids = context_ids[:, :max_context_tokens]
    n_ctx = context_ids.shape[1]

    question_raw = tokenizer.encode(question, add_special_tokens=False)
    question_ids = torch.tensor(question_raw, dtype=torch.long, device="cpu").unsqueeze(0)

    t0 = time.perf_counter()

    with torch.no_grad():
        cache = DynamicCache()
        if press is not None:
            with press(model):
                model(
                    context_ids,
                    past_key_values=cache,
                    cache_position=torch.arange(n_ctx, device="cpu"),
                    use_cache=True,
                )
        else:
            model(
                context_ids,
                past_key_values=cache,
                cache_position=torch.arange(n_ctx, device="cpu"),
                use_cache=True,
            )

        gen_output = model.generate(
            question_ids,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    dt = (time.perf_counter() - t0) * 1000.0

    generated_part = gen_output[0, question_ids.shape[1]:]
    answer = tokenizer.decode(generated_part, skip_special_tokens=True)

    exact_match = int(expected_answer.lower() in answer.lower())

    def _unigram_f1(pred: str, ref: str) -> float:
        pred_toks = pred.lower().split()
        ref_toks = ref.lower().split()
        if not pred_toks or not ref_toks:
            return 0.0
        common = Counter(pred_toks) & Counter(ref_toks)
        num = sum(common.values())
        prec = num / max(len(pred_toks), 1)
        rec = num / max(len(ref_toks), 1)
        if prec + rec == 0:
            return 0.0
        return 2.0 * prec * rec / (prec + rec)

    f1 = _unigram_f1(answer, expected_answer)

    return {
        "answer": answer,
        "exact_match": exact_match,
        "f1": f1,
        "needle_position": needle_position,
        "context_tokens": n_ctx,
        "inference_ms": dt,
    }
