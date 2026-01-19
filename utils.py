"""
Utilities for loading lightweight LLMs and computing attention head scores.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

# Allow short aliases in config.yaml.
MODEL_ALIASES: Dict[str, str] = {
    "tiny-gpt2": "sshleifer/tiny-gpt2",
}


def get_config_models() -> List[str]:
    """
    Read model names from config.yaml:

    llms:
      - tiny-gpt2
      - distilgpt2
      - google/gemma-3-4b-it
    """
    if not CONFIG_PATH.exists():
        return ["tiny-gpt2", "distilgpt2"]

    models: List[str] = []
    in_llms = False
    for raw in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("llms:"):
            in_llms = True
            continue
        if not in_llms:
            continue
        if line.startswith("-"):
            item = line[1:].strip()
            if item:
                models.append(item)
        else:
            # stop if we hit another key
            break

    return models or ["tiny-gpt2", "distilgpt2"]


def resolve_model_name(model_key: str) -> str:
    """
    Map a config-provided key (e.g. tiny-gpt2) to an HF repo id.
    """
    return MODEL_ALIASES.get(model_key, model_key)


def get_model_device(model) -> torch.device:
    """
    Safely get the device of a model, avoiding meta device issues.
    
    Args:
        model: PyTorch model
        
    Returns:
        torch.device: The device the model is on (defaults to CPU if detection fails)
    """
    try:
        # Try to get device from parameters (most reliable)
        device = next(model.parameters()).device
        # Ensure it's not meta device
        if device.type == "meta":
            return torch.device("cpu")
        return device
    except Exception:
        # Fallback to CPU if we can't determine device
        return torch.device("cpu")


@lru_cache(maxsize=2)
def load_llm(model_key: str):
    """
    Load and cache a model/tokenizer pair.

    Args:
        model_key: Key from AVAILABLE_MODELS.

    Returns:
        (tokenizer, model) tuple.
    """
    # Validate model exists in config list (prevent random remote fetches).
    allowed = set(get_config_models())
    if model_key not in allowed:
        raise ValueError(f"Unknown model key: {model_key}. Allowed: {sorted(allowed)}")

    model_name = resolve_model_name(model_key)
    model, tokenizer = load_base_model(model_name)
    return tokenizer, model


def load_base_model(base_model_name: str):
    """
    User-style loader: returns (model, tokenizer).

    Important: Some models + `device_map="auto"` can return `outputs.attentions`
    as tuples of Nones. Since this app *requires* attentions, we auto-fallback
    to a simple CPU load if we detect that issue.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    def _load(device_map):
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            output_attentions=True,
        )
        model.eval()
        return model

    # Try accelerate-style loading first (user requested).
    model = _load(device_map="auto")

    # Check if model is on meta device (no actual data)
    def _is_meta_device(model):
        """Check if any parameter is on meta device."""
        try:
            # Check parameters first (most reliable)
            # If any parameter is on meta, the model is on meta
            for param in model.parameters():
                if param.device.type == "meta":
                    return True
            # If no parameters are on meta, model is not on meta
            # We avoid checking model.device directly as it may trigger errors
            return False
        except Exception:
            # If we can't check parameters, assume it might be problematic
            # and let the fallback handle it
            return False

    # Smoke test: verify attentions are real tensors, not None, and not on meta device.
    try:
        if _is_meta_device(model):
            raise RuntimeError("Model is on meta device")
        with torch.inference_mode():
            enc = tokenizer("hello", return_tensors="pt")
            # Get actual device from model (not meta)
            # Safely get device from parameters
            device = next(model.parameters()).device
            if device.type == "meta":
                raise RuntimeError("Model device is meta")
            out = model(**enc.to(device), output_attentions=True)
        if not out.attentions or any(a is None for a in out.attentions):
            raise RuntimeError("attentions are None")
    except (RuntimeError, Exception) as e:
        # Fallback to plain CPU load to guarantee attentions exist.
        # This handles meta device, None attentions, and other issues
        print(f"Warning: Falling back to CPU load due to: {e}")
        # Reload without device_map to avoid meta device issues
        # When device_map=None, model loads to CPU by default
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,  # Explicitly None to avoid meta device
            output_attentions=True,
        )
        model.eval()
        # Explicitly ensure model is on CPU (not meta)
        model = model.to(torch.device("cpu"))

    return model, tokenizer


def _find_token_positions(sequence_ids: List[int], pattern_ids: List[int]) -> List[int]:
    """
    Find token positions for a pattern (sub-sequence) within a sequence.
    Returns starting indices for each match.
    """
    if not pattern_ids or len(pattern_ids) > len(sequence_ids):
        return []
    matches: List[int] = []
    for i in range(len(sequence_ids) - len(pattern_ids) + 1):
        if sequence_ids[i : i + len(pattern_ids)] == pattern_ids:
            matches.append(i)
    return matches


def _token_mask(input_ids: List[int], phrases: Sequence[str], tokenizer) -> torch.Tensor:
    """
    Build a boolean mask of positions that match any phrase.
    """
    mask = torch.zeros(len(input_ids), dtype=torch.bool)
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
        pattern_ids = tokenizer.encode(phrase, add_special_tokens=False)
        for start in _find_token_positions(input_ids, pattern_ids):
            mask[start : start + len(pattern_ids)] = True
    return mask


def compute_head_scores(
    tokenizer,
    model,
    text: str,
    positives: Sequence[str],
    negatives: Sequence[str],
    mode: str = "final_token",
) -> Tuple[List[List[float]], List[str]]:
    """
    Compute per-head scores = mean(attn to positives) - mean(attn to negatives)
    for the final token, across all layers/heads.

    Returns:
        scores: List[layers][heads]
        tokens: decoded tokens of the input (for potential UI tooltips)
    """
    if not text.strip():
        raise ValueError("Input text is empty.")

    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", None)
    input_ids_list: List[int] = input_ids[0].tolist()

    pos_mask = _token_mask(input_ids_list, positives, tokenizer)
    neg_mask = _token_mask(input_ids_list, negatives, tokenizer)

    with torch.inference_mode():
        device = get_model_device(model)
        outputs = model(
            **encoded.to(device),
            output_attentions=True,
        )

    attentions: Sequence[torch.Tensor] = outputs.attentions
    # Each attention: (batch, heads, seq, seq)
    last_index = attention_mask.sum().item() - 1 if attention_mask is not None else input_ids.size(1) - 1

    scores: List[List[float]] = []
    for layer_attn in attentions:
        head_scores: List[float] = []
        # Move to CPU for lightweight processing.
        layer_attn_cpu = layer_attn[0].detach().cpu()  # (heads, seq, seq)

        if mode == "final_token":
            # Query = final token
            query_slice = layer_attn_cpu[:, last_index, :]  # (heads, seq)
        elif mode == "mean_query":
            # Query = mean over all tokens
            query_slice = layer_attn_cpu.mean(dim=1)  # (heads, seq)
        else:
            raise ValueError(f"Unknown score mode: {mode}")

        pos_scores = query_slice[:, pos_mask].mean(dim=1) if pos_mask.any() else torch.zeros(query_slice.size(0))
        neg_scores = query_slice[:, neg_mask].mean(dim=1) if neg_mask.any() else torch.zeros(query_slice.size(0))
        head_scores_tensor = pos_scores - neg_scores
        head_scores.extend(head_scores_tensor.tolist())
        scores.append(head_scores)

    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids_list)
    return scores, decoded_tokens