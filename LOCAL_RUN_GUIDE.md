# Moodify — Local Streamlit Run Guide

## Recommended: local only
From the project root:

```bash
python run_local.py local
```

Open:
- http://127.0.0.1:8501
- http://localhost:8501

This mode does **not** start Cloudflare or ngrok and does not require any authentication token.

### Windows
Double-click `run_local.bat`.

## Optional: ngrok public link
Only use this when you actually need to access the local app from another device/network.

Set your ngrok auth token as an environment variable named `NGROK_AUTHTOKEN`, then:

```bash
python run_local.py ngrok
```

Windows shortcut: `run_ngrok.bat`.

The ngrok token is read from the environment and is never stored in the project files.

## Troubleshooting
If Streamlit fails, inspect `streamlit.log` in the project root.

If port 8501 is already occupied, stop the existing Streamlit process or change the port in `run_local.py`.
