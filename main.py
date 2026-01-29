import os
import torch
import requests
from transformers import pipeline, MarianMTModel, MarianTokenizer
from diffusers import StableDiffusionPipeline
import google.generativeai as genai

# =========================
# CONFIG
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Add in HuggingFace Secrets
GROQ_MODEL = "gpt-4o-mini"
# Load Gemini API key from HuggingFace Secrets
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found! Add it in HuggingFace → Settings → Secrets")

genai.configure(api_key=api_key)

# Load Gemini model
model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# SPEECH TO TEXT (WHISPER)
# =========================

# Decide device and dtype once
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32

speech_to_text = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    dtype=TORCH_DTYPE,
    device=DEVICE,                 # "cuda" or "cpu" — both work well
    model_kwargs={"attn_implementation": "sdpa"}   # optional: faster on modern torch
)

def transcribe_audio(audio):
    if audio is None:
        return ""

    try:
        # Optional but strongly recommended for files > ~30 seconds
        result = speech_to_text(
            audio,
            chunk_length_s=30,          # ← enables long audio support
            batch_size=8,               # adjust depending on GPU memory
            return_timestamps=True,     # very useful in most apps
            generate_kwargs={
                "language": "english",  # change or remove if multilingual needed
                "task": "transcribe"
            }
        )
        return result["text"].strip()

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            return "❌ GPU out of memory — try smaller batch_size or shorter audio"
        return f"❌ Runtime error: {str(e)}"
    except ValueError as e:
        return f"❌ Invalid input format: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"
# =========================
# TRANSLATION
# =========================
def load_translator(model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
    return tokenizer, model

fr_tokenizer, fr_model = load_translator("Helsinki-NLP/opus-mt-en-fr")
de_tokenizer, de_model = load_translator("Helsinki-NLP/opus-mt-en-de")

def translate(text, lang):
    if not text:
        return ""
    try:
        tokenizer, model = (fr_tokenizer, fr_model) if lang == "French" else (de_tokenizer, de_model)
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"❌ Translation error: {str(e)}"

# =========================
# IMAGE GENERATION
# =========================

# ─── Global setup ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

image_pipe = None
error_message = ""

try:
    print("Loading Stable Diffusion 1.5...")

    image_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        dtype=DTYPE,
        variant="fp16",                      # smaller download & faster load
        safety_checker=None,                 # most common choice → disables NSFW filter
        requires_safety_checker=False,       # removes annoying warning
    )

    # Critical memory optimizations (prevents 90% of OOM crashes on 16–24 GB GPUs)
    image_pipe.enable_attention_slicing()    # #1 most important fix
    image_pipe.enable_vae_slicing()          # helps with larger images / batches

    # Optional: stronger memory saving (slower but very safe on <12 GB VRAM)
    # image_pipe.enable_model_cpu_offload()

    if DEVICE == "cuda":
        image_pipe.to("cuda")

    print("Stable Diffusion loaded successfully ✓")

except Exception as e:
    error_message = f"Failed to load model: {type(e).__name__}: {str(e)}"
    print(f"❌ {error_message}")
    image_pipe = None

def generate_image(prompt: str, negative_prompt: str = "") -> tuple:
    """
    Returns: (PIL.Image or None, status_message: str)
    """
    if not prompt.strip():
        return None, "Please enter a prompt"

    if image_pipe is None:
        return None, error_message or "Model failed to load earlier"

    try:
        # Good defaults for speed + decent quality
        result = image_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "blurry, low quality, distorted, deformed, ugly, bad anatomy",
            num_inference_steps=28,          # 20–35 is the 2026 sweet spot
            guidance_scale=7.0,              # 7–9 usually best
            height=512,
            width=512,
            # You can expose these as parameters later if you want a more advanced UI
        )

        return result.images[0], "Image generated successfully"

    except torch.cuda.OutOfMemoryError:
        msg = "GPU out of memory – try shorter prompt or fewer steps"
        print(f"❌ {msg}")
        return None, msg

    except Exception as e:
        msg = f"Generation failed: {type(e).__name__}: {str(e)}"
        print(f"❌ {msg}")
        return None, msg
# =========================
# CHATBOT (Gemini)
# Output generation function
def generate_output(prompt, max_length, temperature, top_p, seed):
    if not prompt.strip():
        return "Please enter a prompt."

    # Optional random seed
    generation_config = {
        "max_output_tokens": int(max_length),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    if seed != 0:
        generation_config["seed"] = int(seed)
    response = model.generate_content(
        prompt,
        generation_config=generation_config
    )
    return response.text
