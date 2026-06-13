"""
Download BLIP models via Clash proxy.
Run this script ONCE to cache models locally (~800MB).
After downloading, the main app uses cached models without proxy.
"""
import os
import sys
import io

# Fix Windows GBK encoding for emoji/unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# --- Proxy setup for HuggingFace downloads ---
PROXY_URL = "http://127.0.0.1:7890"
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL
os.environ['http_proxy'] = PROXY_URL
os.environ['https_proxy'] = PROXY_URL
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,.local'
# Use official HF endpoint (not mirror) since we go through proxy
os.environ.pop('HF_ENDPOINT', None)
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
# -------------------------------------------

print("=" * 60)
print("BLIP Model Downloader (via Clash proxy)")
print(f"Proxy: {PROXY_URL}")
print("=" * 60)

# Test proxy connectivity first
import requests
try:
    r = requests.get("https://huggingface.co", timeout=10)
    print(f"[Proxy] HuggingFace reachable: HTTP {r.status_code}")
except Exception as e:
    print(f"[Proxy] ERROR: Cannot reach HuggingFace: {e}")
    print("Make sure Clash is running at 127.0.0.1:7890")
    sys.exit(1)

from transformers import BlipProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration

models_to_download = [
    ("BLIP VQA", "Salesforce/blip-vqa-base"),
    ("BLIP Captioning", "Salesforce/blip-image-captioning-base"),
]

for name, model_id in models_to_download:
    print(f"\n{'─' * 60}")
    print(f"[Download] {name} ({model_id})")
    print(f"{'─' * 60}")

    try:
        print(f"  Loading processor...")
        processor = BlipProcessor.from_pretrained(model_id)
        print(f"  ✓ Processor ready")

        print(f"  Loading model (this may take a few minutes)...")
        model = BlipForQuestionAnswering.from_pretrained(model_id) if "vqa" in model_id else BlipForConditionalGeneration.from_pretrained(model_id)
        print(f"  ✓ Model ready")

        # Test inference
        print(f"  Testing inference...")
        from PIL import Image
        import numpy as np
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        inputs = processor(test_img, return_tensors="pt")
        if "vqa" in model_id:
            inputs = processor(test_img, "What is in this image?", return_tensors="pt")
        outputs = model.generate(**inputs, max_length=20)
        print(f"  ✓ Inference OK")

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        sys.exit(1)

print(f"\n{'=' * 60}")
print("✅ All BLIP models downloaded successfully!")
print(f"   Cache location: {os.path.expanduser('~/.cache/huggingface/hub/')}")
print("=" * 60)
