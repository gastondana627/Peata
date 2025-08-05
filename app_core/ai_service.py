# app_core/ai_service.py

import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread

# --- 1. ONE-TIME INITIALIZATION FOR BOTH AI MODELS ---
def initialize_models():
    # Use flags in session state to check if models are already loaded
    if "vertex_ai_initialized" not in st.session_state:
        try:
            creds_info = st.secrets["vertex_ai"]
            vertexai.init(
                project=creds_info["project_id"],
                credentials=service_account.Credentials.from_service_account_info(creds_info)
            )
            # Load the Gemini model once and store it in session state
            st.session_state.vertex_model = GenerativeModel("gemini-1.5-flash")
            st.session_state.vertex_ai_initialized = True
        except Exception as e:
            st.session_state.vertex_ai_initialized = False
            st.error(f"Vertex AI Initialization Failed: {e}")

    # The @st.cache_resource decorator handles caching for the Gemma model
    # We just call the function to ensure it's loaded into the cache.
    load_gemma_model()

@st.cache_resource(show_spinner="Loading Gemma-2 fallback model...")
def load_gemma_model():
    """Loads and caches the Gemma model. Returns a tuple (model, tokenizer)."""
    try:
        hf_token = st.secrets.get("HF_TOKEN")
        if not hf_token:
            st.warning("Hugging Face token (HF_TOKEN) not found in secrets.")
            return None, None
        
        model_id = "google/gemma-2-2b-it"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading Gemma model: {e}")
        return None, None

# --- 2. THE MAIN STREAMING CHATBOT FUNCTION ---
def get_chatbot_response_stream(user_prompt):
    # This is the system prompt to guide the AI's personality
    system_prompt = """
    You are Peata, a friendly and helpful AI assistant for the Peata Animal Shelter.
    Your purpose is to provide information about animal adoption, shelter services, and how to care for pets.
    Keep your answers concise and helpful. If asked about topics outside of animal welfare,
    politely steer the conversation back to the shelter's mission.
    """
    full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nPeata:"

    use_online_model = st.session_state.get("is_online", True)
    vertex_is_ready = st.session_state.get("vertex_ai_initialized", False)

    # --- ATTEMPT TO USE VERTEX AI (GEMINI) ---
    if use_online_model and vertex_is_ready:
        st.session_state.ai_mode = "Online (Vertex AI)"
        try:
            # Get the model from session state instead of re-creating it
            gemini_model = st.session_state.vertex_model
            # Use the streaming parameter
            response_stream = gemini_model.generate_content(full_prompt, stream=True)
            for chunk in response_stream:
                yield chunk.text
            return # End the function after successful streaming
        except Exception as e:
            st.warning(f"Vertex AI streaming failed: {e}. Falling back to Gemma 2.")

    # --- FALLBACK TO GEMMA 2 ---
    st.session_state.ai_mode = "Offline (Gemma 2)"
    gemma_model, tokenizer = load_gemma_model()
    
    if gemma_model and tokenizer:
        try:
            # Setup for streaming with Transformers
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            chat = [{"role": "user", "content": full_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(gemma_model.device)
            
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=250)
            
            # Run generation in a separate thread for streaming
            thread = Thread(target=gemma_model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Yield tokens as they become available
            for new_text in streamer:
                yield new_text
        except Exception as e:
            st.error(f"Gemma inference error: {e}")
            yield "My apologies, I encountered an issue with the Gemma model."
    else:
        yield "I'm sorry, but both AI models are currently unavailable."