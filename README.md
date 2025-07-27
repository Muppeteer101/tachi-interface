# Tachi Interface (Gradio)

This repository contains a simple Gradio application that mimics a multi‑modal Tachi interface.  It provides four independent tabs:

- **Voice** — upload an audio file and receive a summary of its length.
- **Image** — enter a text prompt and receive a placeholder (random noise) image.
- **Video** — enter a prompt and receive a short randomly generated video.
- **File Upload** — upload one or more files and receive a summary of how many were uploaded.

The code is intentionally lightweight and self‑contained.  You can extend each tab’s function to call your own AI models if desired.

## Files

- `app.py` – the Gradio application.
- `requirements.txt` – Python dependencies for the app.

## Running Locally

Install the dependencies and launch the app:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

The UI will be available at [http://localhost:8888](http://localhost:8888).

## Deploying on Runpod

To deploy this application on Runpod using the `RTX A4000` GPU pod:

1. **Create a new pod** using the `runpod/vscode-server:0.0.0` (or any Python‑capable) container image.  Make sure the on‑demand GPU you select (such as RTX A4000 at ~$0.25/hr) fits within your budget.
2. **Set the volume mount path** to `/workspace` and upload this repository into that directory (for example, by cloning the GitHub repo or uploading the ZIP and unzipping it in the terminal).
3. **Edit the pod** and set the *Container Start Command* to:

   ```bash
   python3 /workspace/app.py
   ```

   This command installs no extra dependencies because everything is already declared in `requirements.txt`.  The application will automatically launch on port `8888` and bind to all interfaces.

4. **Expose HTTP Port 8888** so Runpod will forward requests to your app.  You can leave the TCP ports unchanged unless you need SSH access.
5. **Save** the pod configuration.  Runpod will restart the container and automatically run your Gradio interface.  Within a minute or two you should see a public URL under *HTTP Services*.  Open that URL in your browser—you should see the Tachi interface right away.

If you need to make changes after deployment, connect to the pod using the *Web Terminal* or *VS Code* connection and edit `app.py` accordingly.  After saving your changes you can restart the pod to pick them up.

