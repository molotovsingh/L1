"""
model_probe.py  —  enumerate & ping every OpenAI model your org can access
---------------------------------------------------------------------------
• Works on free‑trial, Pay‑as‑you‑go, Plus, Team, and Enterprise orgs
• Handles chat, embeddings, images (DALL·E), audio (TTS / STT)
• Gracefully skips models that aren't enabled (shows the exact error)

Run:
  $ python model_probe.py
---------------------------------------------------------------------------
"""

import os
import json
import time
from collections import defaultdict

import openai
from openai import OpenAI         # v1+ Python SDK (pip install --upgrade openai)

# ── 0.  Configuration ────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in Replit Secrets first!")

client = OpenAI(api_key=OPENAI_API_KEY, organization=os.getenv("OPENAI_ORG_ID"))

# A minimal prompt for each modality
PROMPTS = {
    "chat":      [{"role": "user", "content": "Say hello in one sentence."}],
    "audio_in":  [{"role": "user", "content": "Transcribe the following audio."}],
    "embeddings": "OpenAI embeddings rock.",
    "image_gen": "A tiny robot reading a book, watercolor style",
    "tts":       "This is a test of OpenAI TTS."
}

# ── 1.  Discover models ──────────────────────────────────────────────────
print("🔍 Fetching model list …")
models = client.models.list().data
model_names = sorted([m.id for m in models])

# Heuristic grouping by prefix
tasks = defaultdict(list)
for name in model_names:
    if any(name.startswith(p) for p in ("gpt-", "o", "chatgpt-")):
        tasks["chat"].append(name)
    elif name.startswith("text-embedding"):
        tasks["embeddings"].append(name)
    elif name.startswith(("dall-", "gpt-image")):
        tasks["image_gen"].append(name)
    elif name.endswith("-tts") or name.startswith("tts-"):
        tasks["tts"].append(name)
    elif name.endswith("-transcribe") or name == "whisper-1":
        tasks["audio_in"].append(name)
    else:
        tasks["other"].append(name)

# ── 2.  Probe each model ────────────────────────────────────────────────
def try_call(label, func, **kwargs):
    try:
        start = time.time()
        _ = func(**kwargs)
        dur = time.time() - start
        return ("✅", dur, None)
    except Exception as e:
        return ("❌", 0, f"{type(e).__name__}: {e}")

results = []

print("\n🚀 Probing models (this will cost a few cents at most):\n")
for task, names in tasks.items():
    for name in names:
        if task == "chat":
            status, dur, err = try_call(
                task,
                client.chat.completions.create,
                model=name,
                messages=PROMPTS["chat"],
                max_tokens=20,
                stream=False,
            )
        elif task == "embeddings":
            status, dur, err = try_call(
                task,
                client.embeddings.create,
                model=name,
                input=PROMPTS["embeddings"],
            )
        elif task == "image_gen":
            status, dur, err = try_call(
                task,
                client.images.generate,
                model=name,
                prompt=PROMPTS["image_gen"],
                n=1,
                size="256x256",
            )
        elif task == "tts":
            status, dur, err = try_call(
                task,
                client.audio.speech.create,
                model=name,
                voice="alloy",
                input=PROMPTS["tts"],
                response_format="json",
            )
        elif task == "audio_in":
            # Just hit the endpoint with empty audio to check permissions
            status, dur, err = try_call(
                task,
                client.audio.transcriptions.create,
                model=name,
                file=b"",  # invalid, but permission check happens first
                prompt="",
            )
        else:
            continue
        results.append(
            dict(model=name, task=task, status=status, ms=int(dur * 1000), error=err)
        )
        print(f"{status}  {name:<35}  {task:<11}  {('%.0f ms' % (dur*1000)) if dur else ''}")
        if err and "insufficient_quota" in err:
            print("    ↳ looks like this model requires org verification or higher tier.")

# ── 3.  Save JSON report (useful for diffing later) ──────────────────────
with open("model_probe_report.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n📄  Full report saved to model_probe_report.json")
print("💡  Tip: rerun after org verification or spend‑tier upgrade to see changes.\n")