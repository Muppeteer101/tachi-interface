import os
import gradio as gr
from PIL import Image
import numpy as np
import soundfile as sf
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ------------------------------------------------------------------------
# Hugging Face authentication and model loading
# ------------------------------------------------------------------------

# Fetch your Hugging Face token from an environment variable
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    # Log in so gated models can be downloaded
    login(token=hf_token)

# Name of the Mistral model to load; adjust if you use a different checkpoint
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Load tokenizer and model (this will download weights on first run)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=hf_token,
)

# ------------------------------------------------------------------------
# Handler functions for each tab
# ------------------------------------------------------------------------

def transcribe(audio_file):
    """Return a simple summary of the uploaded audio."""
    if not audio_file:
        return "No audio uploaded."
    data, samplerate = sf.read(audio_file)
    duration = len(data) / float(samplerate)
    return f"Received audio of {duration:.2f} seconds."

def generate_image(prompt):
    """Generate a random placeholder image."""
    width, height = 512, 512
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)

def handle_text(prompt):
    """Generate a response using the Mistral model."""
    if not prompt:
        return "No input provided."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def handle_files(uploaded_files):
    """Return a message summarizing uploaded files."""
    if not uploaded_files:
        return "No files uploaded."
    return f"Received {len(uploaded_files)} file(s)."

# ------------------------------------------------------------------------
# Build Gradio interface with four tabs
# ------------------------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("# Tachi Interface Demo")

    with gr.Tab(label="Voice"):
        audio_input = gr.Audio(type="filepath", label="Upload Audio")
        voice_output = gr.Textbox(label="Result")
        audio_input.change(fn=transcribe, inputs=audio_input, outputs=voice_output)

    with gr.Tab(label="Image"):
        text_input = gr.Textbox(label="Prompt", placeholder="Describe the image you want to generate")
        image_output = gr.Image(label="Generated Image")
        text_input.submit(fn=generate_image, inputs=text_input, outputs=image_output)

    with gr.Tab(label="Text"):
        chat_input = gr.Textbox(label="Enter your text prompt", lines=3)
        chat_output = gr.Textbox(label="Response")
        chat_input.submit(fn=handle_text, inputs=chat_input, outputs=chat_output)

    with gr.Tab(label="File Upload"):
        file_input = gr.File(label="Upload Files", file_count="multiple")
        file_status = gr.Textbox(label="Status")
        file_input.change(fn=handle_files, inputs=file_input, outputs=file_status)

if __name__ == "__main__":
    # Expose the app on all interfaces so RunPod can proxy it
    demo.launch(server_name="0.0.0.0", server_port=8888, show_error=True)
