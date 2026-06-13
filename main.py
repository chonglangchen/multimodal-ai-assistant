import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================
# Network: bypass broken proxy for direct connections
# hf-mirror.com + api.deepseek.com work directly (no proxy needed)
# ============================================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import urllib.request as _ur
_ur.getproxies = lambda: {}

for _pv in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
            'ALL_PROXY', 'all_proxy']:
    os.environ.pop(_pv, None)
os.environ['no_proxy'] = '*'
os.environ['NO_PROXY'] = '*'
# ============================================================

import threading
import atexit
from app.backend.services.flask_app import app as flask_app
from app.frontend.gradio_app import launch_gradio

UPLOADS_DIR = os.path.join("app", "uploads")

def cleanup_uploads_folder():
    """Clean the uploads folder by deleting all files within it."""
    if os.path.exists(UPLOADS_DIR):
        for filename in os.listdir(UPLOADS_DIR):
            file_path = os.path.join(UPLOADS_DIR, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not delete {file_path}: {e}")

# Clean up uploads on startup
cleanup_uploads_folder()

atexit.register(cleanup_uploads_folder)

def run_flask():
    """Run the Flask backend server."""
    flask_app.run(debug=False, port=5000, use_reloader=False)

def run_gradio():
    """Launch the Gradio frontend interface."""
    launch_gradio()

if __name__ == "__main__":
    # Create uploads directory if it doesn't exist
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    # Start Flask backend in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # Start Gradio frontend (main thread)
    run_gradio() 