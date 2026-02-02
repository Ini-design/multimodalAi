import os
import torch
import requests
from transformers import pipeline, MarianMTModel, MarianTokenizer
from diffusers import StableDiffusionXLPipeline
import google.generativeai as genai

# =========================
# CONFIG
# =========================

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
class StableDiffusionEngine:
    """
    Stable Diffusion XL Engine (HF Spaces safe version)
    """

    def __init__(
        self,
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        hf_token: str | None = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        # ✅ Proper dtype assignment
        if self.device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32   # 🔥 REQUIRED FOR CPU
    
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,      # ✅ NOW ACTUALLY USED
            use_safetensors=True,
            token=hf_token,
        )
    
        self.pipe.to(self.device)


    def generate_image(
        self,
        prompt: str,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
    ):
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            ).images[0]
        return image
# =========================
# EXPORTABLE IMAGE FUNCTION
# =========================
sd_engine = StableDiffusionEngine()

def generate_image(prompt):
    return sd_engine.generate_image(prompt)

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
