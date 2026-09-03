"""Moodify local launcher.

Modes:
  python run_local.py local   -> http://127.0.0.1:8501
  python run_local.py ngrok   -> starts local Streamlit + ngrok (token required)

No Cloudflare tunnel is started automatically.
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


def install_deps() -> None:
    print("[1/3] Checking/installing project dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)


def start_streamlit() -> subprocess.Popen:
    print("[2/3] Starting Streamlit locally on port 8501...")
    log = open(ROOT / "streamlit.log", "w", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.address=127.0.0.1", "--server.port=8501",
         "--server.headless=true", "--browser.gatherUsageStats=false"],
        stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
    )


def wait_for_streamlit() -> None:
    for _ in range(60):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    print("LOCAL STREAMLIT: http://127.0.0.1:8501")
                    print("Also available at: http://localhost:8501")
                    return
        except Exception:
            time.sleep(1)
    log = ROOT / "streamlit.log"
    tail = log.read_text(encoding="utf-8", errors="ignore")[-6000:] if log.exists() else "No log found."
    raise RuntimeError("Streamlit failed its local health check.\n\n" + tail)


def start_ngrok() -> None:
    token = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "NGROK_AUTHTOKEN is not set. For ngrok mode, configure your ngrok auth token first."
        )
    print("[3/3] Starting ngrok using NGROK_AUTHTOKEN...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ngrok"], check=True)
    import ngrok  # type: ignore
    forwarder = ngrok.forward("http://127.0.0.1:8501", authtoken_from_env=True)
    print("NGROK PUBLIC URL:", forwarder.url())
    print("Keep this terminal open while using the URL.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        forwarder.close()


def main() -> None:
    mode = (sys.argv[1].lower() if len(sys.argv) > 1 else "local")
    if mode not in {"local", "ngrok"}:
        print("Usage: python run_local.py [local|ngrok]")
        raise SystemExit(2)
    install_deps()
    proc = start_streamlit()
    try:
        wait_for_streamlit()
        if mode == "ngrok":
            start_ngrok()
        else:
            print("Mode: LOCAL ONLY — no Cloudflare/ngrok tunnel is used.")
            print("Press Ctrl+C to stop Moodify.")
            proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
