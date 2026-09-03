# Moodify — one-command Google Colab launcher
# This script starts the real Streamlit app, verifies the local server, exposes
# it through a Cloudflare Quick Tunnel (no account/token), and falls back to
# ngrok only when an NGROK_AUTHTOKEN is supplied.

from pathlib import Path
import os, re, subprocess, time, urllib.request

ROOT = Path.cwd()
while not (ROOT / "app.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if not (ROOT / "app.py").exists():
    raise FileNotFoundError("Could not find app.py. Run this from the extracted Moodify project folder.")
os.chdir(ROOT)

print("[1/5] Installing project dependencies...")
subprocess.run(["python", "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

print("[2/5] Starting Streamlit on port 8501...")
log = open("streamlit.log", "w")
proc = subprocess.Popen(
    ["python", "-m", "streamlit", "run", "app.py",
     "--server.address=0.0.0.0", "--server.port=8501",
     "--server.headless=true", "--browser.gatherUsageStats=false"],
    stdout=log, stderr=subprocess.STDOUT, cwd=ROOT
)

print("[3/5] Waiting for the local health check...")
healthy = False
for _ in range(60):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=2) as r:
            healthy = (r.status == 200)
        if healthy:
            break
    except Exception:
        time.sleep(1)
if not healthy:
    print(Path("streamlit.log").read_text(errors="ignore")[-6000:])
    proc.terminate()
    raise RuntimeError("Streamlit did not pass its local health check. See streamlit.log above.")
print("LOCAL STREAMLIT: http://127.0.0.1:8501  ✓")

print("[4/5] Trying Cloudflare Quick Tunnel (preferred for Colab demos)...")
cloudflared = ROOT / "cloudflared"
if not cloudflared.exists():
    import urllib.request as ur
    ur.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        cloudflared
    )
    cloudflared.chmod(0o755)

tunnel = subprocess.Popen(
    [str(cloudflared), "tunnel", "--url", "http://127.0.0.1:8501", "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)
public_url = None
start = time.time()
while time.time() - start < 45:
    line = tunnel.stdout.readline()
    if not line:
        time.sleep(.2)
        continue
    print("[tunnel]", line.rstrip())
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if m:
        public_url = m.group(0)
        break

if public_url:
    print("\n====================================================")
    print("MOODIFY PUBLIC STREAMLIT URL")
    print(public_url)
    print("====================================================")
else:
    print("Cloudflare Quick Tunnel did not return a URL.")
    token = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if token:
        print("[5/5] Falling back to ngrok using NGROK_AUTHTOKEN...")
        subprocess.run(["python", "-m", "pip", "install", "-q", "ngrok"], check=True)
        import ngrok
        forwarder = ngrok.forward("localhost:8501", authtoken_from_env=True)
        print("NGROK PUBLIC URL:", forwarder.url())
    else:
        print("[5/5] No ngrok token supplied; local Streamlit is still running.")
        print("Set NGROK_AUTHTOKEN only if you specifically want the ngrok fallback.")

print("\nKeep this Colab runtime running while using the public URL.")
print("To diagnose startup problems, inspect streamlit.log.")
