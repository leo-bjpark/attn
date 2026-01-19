"""
Utilities for computing token-to-token attention head scores.
"""

from typing import Dict, List, Tuple

import torch
from utils import get_model_device


def token_to_token_head_scores(
    tokenizer,
    model,
    text: str,
    src_index: int,
    dst_index: int,
) -> Tuple[List[List[float]], List[str]]:
    """
    Compute per-head attention scores from a source token to a destination token
    across all layers.

    Args:
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face causal LM with `output_attentions=True` support.
        text: Input text that will be tokenized.
        src_index: Index of the source token (key position).
        dst_index: Index of the destination token (query position).

    Returns:
        scores: List[layers][heads] with scalar attention values.
        tokens: Decoded tokens of the input sequence.
    """
    if not text.strip():
        raise ValueError("Input text is empty.")

    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    seq_len = input_ids.size(1)

    if not (0 <= src_index < seq_len) or not (0 <= dst_index < seq_len):
        raise ValueError(
            f"src_index ({src_index}) and dst_index ({dst_index}) must be within "
            f"the token range [0, {seq_len - 1}]."
        )

    with torch.inference_mode():
        device = get_model_device(model)
        outputs = model(
            **encoded.to(device),
            output_attentions=True,
        )

    attentions = outputs.attentions  # sequence of (batch, heads, seq, seq)

    scores: List[List[float]] = []
    for layer_attn in attentions:
        # shape: (heads, seq, seq)
        layer_attn_cpu = layer_attn[0].detach().cpu()
        # Attention from dst (query row) to src (key column).
        head_scores_tensor = layer_attn_cpu[:, dst_index, src_index]  # (heads,)
        scores.append(head_scores_tensor.tolist())

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    return scores, tokens


def compute_all_token_pairs_head_scores(
    tokenizer,
    model,
    text: str,
) -> Tuple[Dict[str, List[List[float]]], List[str]]:
    """
    Compute per-head attention scores for all token pairs (src, dst) across all layers.

    Args:
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face causal LM with `output_attentions=True` support.
        text: Input text that will be tokenized.

    Returns:
        all_scores: Dict with keys "{src_index}_{dst_index}" and values List[layers][heads]
        tokens: Decoded tokens of the input sequence.
    """
    if not text.strip():
        raise ValueError("Input text is empty.")

    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    seq_len = input_ids.size(1)

    with torch.inference_mode():
        device = get_model_device(model)
        outputs = model(
            **encoded.to(device),
            output_attentions=True,
        )

    attentions = outputs.attentions  # sequence of (batch, heads, seq, seq)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # Debug: check attention shape
    if attentions and len(attentions) > 0:
        first_layer = attentions[0]
        print(f"Debug: First layer attention shape: {first_layer.shape}")
        if len(first_layer.shape) == 4:
            print(f"Debug: First layer[0] shape: {first_layer[0].shape}")
            # Check if heads are actually different
            layer_0 = first_layer[0].detach().cpu()
            if layer_0.shape[0] > 1:  # if more than 1 head
                sample_val_0 = layer_0[0, 0, 0].item()
                sample_val_1 = layer_0[1, 0, 0].item()
                print(f"Debug: Head 0 vs Head 1 at [0,0]: {sample_val_0} vs {sample_val_1}")

    all_scores: Dict[str, List[List[float]]] = {}
    for dst_idx in range(seq_len):
        for src_idx in range(seq_len):
            key = f"{src_idx}_{dst_idx}"
            scores: List[List[float]] = []
            for layer_idx, layer_attn in enumerate(attentions):
                # shape: (batch, heads, seq, seq)
                # layer_attn[0] gives (heads, seq, seq)
                layer_attn_cpu = layer_attn[0].detach().cpu()
                
                # Verify shape
                if dst_idx == 0 and src_idx == 0 and layer_idx == 0:
                    print(f"Debug: Layer 0 attention shape: {layer_attn_cpu.shape}")
                    print(f"Debug: Accessing [:, {dst_idx}, {src_idx}]")
                
                # Attention from dst (query row) to src (key column).
                # attention[head, query_idx, key_idx] = attention[head, dst_idx, src_idx]
                # layer_attn_cpu shape: (heads, seq, seq)
                # We want attention[head, dst_idx, src_idx] for each head
                head_scores_tensor = layer_attn_cpu[:, dst_idx, src_idx]  # (heads,)
                
                # Debug: check if heads are different
                if dst_idx == 0 and src_idx == 0 and layer_idx == 0:
                    print(f"Debug: Layer {layer_idx}, position [{dst_idx}, {src_idx}]")
                    print(f"Debug: Head scores shape: {head_scores_tensor.shape}")
                    print(f"Debug: Head scores values: {head_scores_tensor.tolist()}")
                    # Also check raw attention values for first few heads
                    if layer_attn_cpu.shape[0] >= 2:
                        print(f"Debug: Raw attention[0, {dst_idx}, {src_idx}] = {layer_attn_cpu[0, dst_idx, src_idx].item()}")
                        print(f"Debug: Raw attention[1, {dst_idx}, {src_idx}] = {layer_attn_cpu[1, dst_idx, src_idx].item()}")
                
                scores.append(head_scores_tensor.tolist())
            all_scores[key] = scores
    
    # Debug: print a few sample values to verify they're different
    if len(all_scores) > 0:
        sample_keys = list(all_scores.keys())[:3]
        for sk in sample_keys:
            if all_scores[sk] and all_scores[sk][0]:
                print(f"Debug: {sk} layer0 heads = {all_scores[sk][0]}")

    return all_scores, tokens