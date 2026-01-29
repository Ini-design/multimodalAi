import gradio as gr
from main import (
    transcribe_audio,
    translate,
    generate_image,
    generate_output
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

    with gr.Tab("🖼️ Image Generator"):
        image_prompt = gr.Textbox(label="Image Prompt")
        generate_btn = gr.Button("Generate Image")
        image_output = gr.Image()
        generate_btn.click(generate_image, image_prompt, image_output)

    with gr.Tab("AI Chatbot (Gemini)") :
        prompt = gr.Textbox(
        label="Enter your prompt here...",
        placeholder="Example: what is algorithm...",
        lines=3
    )
   # with gr.Row():
        max_length = gr.Slider(50, 600, value=300, step=25, label="Max Length")
        temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
        seed = gr.Number(value=0, label="Random Seed (0 = random)")
        
        btn = gr.Button("Generate ✨")
        output = gr.Textbox(label="Generated Answer ✨", lines=5)
    
    btn.click(
        generate_output,
        inputs=[prompt, max_length, temperature, top_p, seed],
        outputs=output
    )
demo.launch(debug=True)
