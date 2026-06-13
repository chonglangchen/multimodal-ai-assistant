"""
Download Japanese dictionary (unidic) for Kokoro Japanese TTS support.
Must be run once before using Japanese voices.

Usage: python download_unidic.py

This script applies the same proxy bypass as main.py, so it should work
even in restricted network environments.
"""
import os
import urllib.request

# ============================================================
# Same proxy bypass as main.py
# ============================================================
urllib.request.getproxies = lambda: {}

for _pv in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
            'ALL_PROXY', 'all_proxy']:
    os.environ.pop(_pv, None)
os.environ['no_proxy'] = '*'
os.environ['NO_PROXY'] = '*'
# ============================================================

print("[UniDic] Downloading Japanese dictionary (~300MB)...")
print("[UniDic] If this fails, download manually from:")
print("         https://cotonoha-dic.s3-ap-northeast-1.amazonaws.com/unidic-3.1.0.zip")
print("         and extract to: python -c \"import unidic; print(unidic.DICDIR)\"")

from unidic import download
download.download_version('3.1.0+2021-08-31')
print("[UniDic] Dictionary downloaded successfully!")
print("[UniDic] Japanese TTS is now ready.")
