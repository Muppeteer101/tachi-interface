import gradio as gr
from PIL import Image
import numpy as np
import soundfile as sf

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
    """Echo back whatever the user types."""
    if not prompt:
        return "No input provided."
    return f"You said: {prompt}"

def handle_files(uploaded_files):
    """Return a message summarizing uploaded files."""
    if not uploaded_files:
        return "No files uploaded."
    return f"Received {len(uploaded_files)} file(s)."

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
    demo.launch(server_name="0.0.0.0", server_port=8888, show_error=True)
