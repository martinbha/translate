"""Local X -> English translation with NLLB-200.

Loaded lazily and kept resident on the GPU for the worker's lifetime.
"""
from typing import Callable, Optional

from app.config import settings
from worker.lang import ENGLISH_NLLB

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is not None:
        return
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(settings.nllb_model)
    _model = AutoModelForSeq2SeqLM.from_pretrained(settings.nllb_model)
    if settings.whisper_device == "cuda" and torch.cuda.is_available():
        _model = _model.to("cuda").half()
    _model.eval()


def translate_segments(
    texts: list[str],
    src_nllb: str,
    batch_size: int = 16,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> list[str]:
    """Translate a list of segment texts from `src_nllb` into English."""
    _load()
    import torch

    device = next(_model.parameters()).device
    _tokenizer.src_lang = src_nllb
    bos = _tokenizer.convert_tokens_to_ids(ENGLISH_NLLB)

    out: list[str] = []
    total = max(1, len(texts))
    for i in range(0, len(texts), batch_size):
        batch = [t if t.strip() else " " for t in texts[i : i + batch_size]]
        enc = _tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            gen = _model.generate(
                **enc, forced_bos_token_id=bos, max_length=512, num_beams=2
            )
        out.extend(_tokenizer.batch_decode(gen, skip_special_tokens=True))
        if progress_cb:
            progress_cb(min(1.0, (i + batch_size) / total))
    return out
