"""Build the fixed prompt set of the vLLM benchmarks from files that ship inside the model directory (so every host can rebuild it byte for byte).

    python make_workload.py --model MODEL_DIR --out workload.json [--lengths 16 128 512 1000 2000]

Prompts are token-id lists cut from the tokenised corpus (the model's README and LICENSE, in that order), so a prompt has exactly the requested token count and the
client needs no tokenizer. The output records the hashes of the corpus, the tokenizer files and the prompts. Needs ``transformers`` (the vLLM environment has it).
"""
import argparse
import hashlib
import json
from pathlib import Path

CORPUS_FILES = ("README.md", "LICENSE")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[16, 128, 512, 1000, 2000])
    a = p.parse_args()
    from transformers import AutoTokenizer
    model = Path(a.model)
    tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    text = "\n\n".join((model / f).read_text() for f in CORPUS_FILES)
    ids = tok.encode(text, add_special_tokens=False)
    need = max(a.lengths)
    if len(ids) < need:
        raise SystemExit(f"the corpus has {len(ids)} tokens, {need} requested")
    prompts = {f"p{n}": ids[:n] for n in a.lengths}
    out = {"schema": "tbccl-vllm-workload/1", "model_dir_name": model.name, "vocab_size": len(tok), "corpus_files": list(CORPUS_FILES),
           "corpus_sha256": sha256(text.encode()), "tokenizer_sha256": {f: sha256((model / f).read_bytes()) for f in TOKENIZER_FILES if (model / f).exists()},
           "prompts": prompts, "prompt_sha256": {k: sha256(json.dumps(v).encode()) for k, v in prompts.items()},
           "prompt_text_prefix": {k: tok.decode(v[:12]) for k, v in prompts.items()}}
    Path(a.out).write_text(json.dumps(out))
    print(a.out, {k: len(v) for k, v in prompts.items()}, "corpus tokens", len(ids))


if __name__ == "__main__":
    main()
