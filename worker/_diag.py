"""One-off diagnostic for the pyannote 4.x diarization API.

Run on the server:
    docker compose exec worker python3 -m worker._diag /srv/data/full-6.m4a

Shows exactly what `pipeline(audio)` returns so we can call it correctly
instead of relying on whisperx's wrapper.
"""
import os
import sys
import types

import numpy as np
import torch
import whisperx
from pyannote.audio import Pipeline

path = sys.argv[1] if len(sys.argv) > 1 else None
if path:
    audio = whisperx.load_audio(path)[: 16000 * 20]  # first 20s is plenty
else:
    audio = np.zeros(16000 * 10, dtype=np.float32)
inp = {"waveform": torch.from_numpy(audio[None, :]), "sample_rate": 16000}

model_name = os.environ.get(
    "DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
)
print(f"Loading {model_name} ...")
pipe = Pipeline.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
pipe.to(torch.device("cuda"))

print("\n=== pipeline(inp) ===")
out = pipe(inp)
print("type:", type(out))
is_gen = isinstance(out, types.GeneratorType)
print("is generator:", is_gen)

result = out
if is_gen:
    items = list(out)
    print(f"generator yielded {len(items)} items")
    for it in items[-4:]:
        attrs = [a for a in dir(it) if not a.startswith("_")]
        print(f"  item type={type(it)}  attrs={attrs[:20]}")
    result = items[-1] if items else None

print("\n=== final result ===")
print("type:", type(result))
print("attrs:", [a for a in dir(result) if not a.startswith("_")][:30])
for attr in ("speaker_diarization", "speaker_embeddings", "labels", "itertracks"):
    print(f"  has {attr}: {hasattr(result, attr)}")

# Also show whether there's a non-generator entry point.
print("\n=== has .apply? ===", hasattr(pipe, "apply"))
