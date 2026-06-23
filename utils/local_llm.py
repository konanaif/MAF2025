import os
from functools import lru_cache


DEFAULT_LOCAL_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
API_PREFIXES = ("gpt", "claude", "HCX", "gemini")

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache-maf-demo")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-maf-demo")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
if os.environ.get("MAF_LOCAL_LLM_LOCAL_FILES_ONLY", "0").lower() in {
    "1",
    "true",
    "yes",
}:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def default_local_model_name():
    return os.environ.get("MAF_LOCAL_TEXT_MODEL", DEFAULT_LOCAL_LLM)


def local_files_only():
    use_local_only = os.environ.get("MAF_LOCAL_LLM_LOCAL_FILES_ONLY", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    if use_local_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return use_local_only


def should_use_local_llm(model_name=None):
    force_local = os.environ.get("MAF_USE_LOCAL_LLM", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if force_local:
        return True
    if model_name and not model_name.startswith(API_PREFIXES):
        return True
    return not os.environ.get("OPENAI_API_KEY")


def resolve_local_model_name(model_name=None):
    if model_name and not model_name.startswith(API_PREFIXES):
        return model_name
    return default_local_model_name()


@lru_cache(maxsize=2)
def _load_model(model_name, use_local_files_only):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        local_files_only=use_local_files_only,
        trust_remote_code=True,
    )
    model_kwargs = {
        "local_files_only": use_local_files_only,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs.update({"device_map": "auto", "torch_dtype": torch.float16})
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def generate_text(prompt, model_name=None, max_new_tokens=128):
    import torch

    model_name = resolve_local_model_name(model_name)
    tokenizer, model = _load_model(model_name, local_files_only())
    device = getattr(model, "device", None) or next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
