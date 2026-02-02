import gradio as gr
from main import (
    transcribe_audio,
    translate,
    generate_image,
    generate_output,
    StableDiffusionEngine
)

# Load Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found in environment variables!")

# Initialize Stable Diffusion
sd_engine = StableDiffusionEngine(hf_token=HF_TOKEN)

def generate_image_wrapper(prompt, steps, cfg, width, height):
    return sd_engine.generate_image(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        width=width,
        height=height,
    )

with gr.Blocks(title="🎧 Multimodal AI Assistant") as demo:
    gr.Markdown("# 🚀 Multimodal AI Assistant")
    gr.Markdown("Voice ➜ Text | Translate | Image Generation | Chatbot")

    with gr.Tab("🎙️ Voice to Text"):
        audio_input = gr.Audio(type="filepath", label="Upload Voice Note")
        text_output = gr.Textbox(label="Transcribed Text",lines=3)
        audio_input.change(transcribe_audio, audio_input, text_output)

    with gr.Tab("🌍 Translation"):
        input_text = gr.Textbox(label="English Text")
        lang = gr.Radio(["French", "German"], value="French", label="Translate To")
        translate_btn = gr.Button("Translate")
        translated_text = gr.Textbox(label="Translated Text")
        translate_btn.click(translate, [input_text, lang], translated_text)

   # ================= IMAGE GENERATOR TAB =================
    with gr.Tab("🖼️ Image Generator"):
        img_prompt = gr.Textbox(
            label="Prompt",
            placeholder="Describe the image you want...",
            lines=4,
        )

        steps = gr.Slider(5, 20, value=10, step=1, label="Inference Steps")
        cfg = gr.Slider(5.0, 9.0, value=6.0, step=0.5, label="CFG Scale")
        width = gr.Dropdown([512, 768, 1024], value=768, label="Width")
        height = gr.Dropdown([512, 768, 1024], value=512, label="Height")

        img_generate_btn = gr.Button("🚀 Generate Image", variant="primary")
        img_output = gr.Image(label="Generated Image", type="pil")

        examples = [
            "A futuristic African city at sunset, ultra realistic",
            "Microscopic view of malaria parasites",
            "A humanoid AI researcher in a high-tech lab",
        ]
        gr.Examples(examples=examples, inputs=img_prompt)

        img_generate_btn.click(
            generate_image_wrapper,
            inputs=[img_prompt, steps, cfg, width, height],
            outputs=img_output,
        )

    # ================= CHATBOT TAB =================
    with gr.Tab("🤖 AI Chatbot"):
        chat_prompt = gr.Textbox(
            label="Enter your prompt",
            placeholder="Example: What is an algorithm?",
            lines=3,
        )

        max_length = gr.Slider(50, 600, value=300, step=25, label="Max Length")
        temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
        seed = gr.Number(value=0, label="Random Seed (0 = random)")

        chat_btn = gr.Button("Generate ✨")
        chat_output = gr.Textbox(label="Generated Answer ✨", lines=5)

        chat_btn.click(
            generate_output,
            inputs=[chat_prompt, max_length, temperature, top_p, seed],
            outputs=chat_output,
        )

demo.launch(debug=True)
