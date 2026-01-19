from flask import Flask, jsonify, render_template, request

from attn_scores import (
    compute_all_token_pairs_head_scores,
    token_to_token_head_scores,
)
from utils import compute_head_scores, get_config_models, load_llm

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", models=get_config_models())


@app.get("/api/models")
def api_models():
    return jsonify({"models": get_config_models()})


@app.post("/api/load_model")
def api_load_model():
    data = request.get_json(force=True)
    model_key = data.get("model", "tiny-gpt2")
    try:
        load_llm(model_key)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok", "model": model_key})


@app.post("/api/tokenize")
def api_tokenize():
    """Tokenize text and return tokens for display."""
    data = request.get_json(force=True)
    model_key = data.get("model", "tiny-gpt2")
    text = data.get("text", "")
    try:
        tokenizer, _ = load_llm(model_key)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text, add_special_tokens=False))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    return jsonify({"tokens": tokens})


@app.post("/api/head_scores")
def api_head_scores():
    data = request.get_json(force=True)
    model_key = data.get("model", "tiny-gpt2")
    text = data.get("text", "")
    positives = data.get("positives", [])
    negatives = data.get("negatives", [])
    score_mode = data.get("score_mode", "final_token")
    view_mode = data.get("view_mode", "concept")  # "concept" | "pair"
    src_index = data.get("src_index")
    dst_index = data.get("dst_index")

    try:
        tokenizer, model = load_llm(model_key)
        if view_mode == "pair":
            if src_index is None or dst_index is None:
                raise ValueError("src_index and dst_index are required for pair view.")
            scores, tokens = token_to_token_head_scores(
                tokenizer,
                model,
                text,
                int(src_index),
                int(dst_index),
            )
        else:
            scores, tokens = compute_head_scores(
                tokenizer,
                model,
                text,
                positives,
                negatives,
                mode=score_mode,
            )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "model": model_key,
            "scores": scores,
            "tokens": tokens,
            "positives": positives,
            "negatives": negatives,
            "score_mode": score_mode,
            "view_mode": view_mode,
        }
    )


@app.post("/api/compute_all_pairs")
def api_compute_all_pairs():
    """Compute attention scores for all token pairs."""
    data = request.get_json(force=True)
    model_key = data.get("model", "tiny-gpt2")
    text = data.get("text", "")

    try:
        tokenizer, model = load_llm(model_key)
        all_scores, tokens = compute_all_token_pairs_head_scores(
            tokenizer,
            model,
            text,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "model": model_key,
            "all_scores": all_scores,
            "tokens": tokens,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)