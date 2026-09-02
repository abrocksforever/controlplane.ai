"""
frontend/run.py - Launch Script for ControlPlane.ai Demonstration UI
Run with: python frontend/run.py
"""

import os
import sys
import webbrowser
import threading
import time

# Ensure parent directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from frontend.server import start_server


def open_browser(url: str):
    time.sleep(1.2)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    
    print("\n" + "=" * 70)
    print("  [*] CONTROLPLANE.AI -- STEP-BY-STEP DEMONSTRATION UI")
    print(f"  [*] Opening web interface at {url}")
    print("=" * 70 + "\n")

    threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    start_server(host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
