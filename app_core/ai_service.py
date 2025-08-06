# app_core/ai_service.py

import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# This function initializes the primary online model (Vertex AI).
def initialize_vertex_ai():
    if "vertex_ai_initialized" not in st.session_state:
        try:
            creds_info = st.secrets["vertex_ai"]
            vertexai.init(
                project=creds_info["project_id"],
                credentials=service_account.Credentials.from_service_account_info(creds_info)
            )
            st.session_state.vertex_model = GenerativeModel("gemini-2.0-flash")
            st.session_state.vertex_ai_initialized = True
        except Exception as e:
            st.session_state.vertex_ai_initialized = False
            st.error(f"Vertex AI Initialization Failed: {e}")
    return st.session_state.get("vertex_ai_initialized", False)

# This function loads the fallback offline model (Gemma) only when needed.
@st.cache_resource(show_spinner="Loading offline model for the first time...")
def load_gemma_model():
    try:
        hf_token = st.secrets.get("HF_TOKEN")
        if not hf_token:
            st.warning("Hugging Face token (HF_TOKEN) not found in secrets.")
            return None

        model_id = "google/gemma-2-2b-it"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, token=hf_token
        ).to("cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        
        st.session_state.gemma_model = {"model": model, "tokenizer": tokenizer}
        return st.session_state.gemma_model
    except Exception as e:
        from huggingface_hub.utils import HfHubHTTPError

        if isinstance(e, HfHubHTTPError) and e.response.status_code == 401:
            st.error("Gemma Model Loading Failed: Unauthorized. "
                     "This is likely due to a missing or invalid HF_TOKEN in your Streamlit secrets. "
                     "Please add your Hugging Face token to the secrets to enable the offline model.")
        else:
            st.error(f"An unexpected error occurred while loading the Gemma model: {e}")

        # Store failure state to inform the UI
        st.session_state.gemma_model = None
        return None

# This is the main function called by your app.
def get_chatbot_response(user_prompt):
    system_prompt = """
    You are Peata, a friendly and helpful AI assistant for the Peata Animal Shelter.
    Your purpose is to provide information about animal adoption, shelter services, and how to care for pets.
    Keep your answers concise and helpful. If asked about topics outside of animal welfare,
    politely steer the conversation back to the shelter's mission.
    """
    full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nPeata:"

    # Try the online model if it's selected and initialized
    if st.session_state.get("is_online", True) and initialize_vertex_ai():
        try:
            st.session_state.ai_mode = "Online (Vertex AI)"
            gemini_model = st.session_state.vertex_model
            response_stream = gemini_model.generate_content(full_prompt, stream=True)
            return (chunk.text for chunk in response_stream)
        except Exception as e:
            st.warning(f"Online AI failed ({e.__class__.__name__}). Falling back to offline model.")

    # --- FALLBACK LOGIC ---
    st.session_state.ai_mode = "Offline (Gemma 2)"
    
    # Lazy-load Gemma only if it's not already in memory
    if "gemma_model" not in st.session_state:
        load_gemma_model()

    if "gemma_model" in st.session_state and st.session_state.gemma_model:
        gemma = st.session_state.gemma_model
        model = gemma["model"]
        tokenizer = gemma["tokenizer"]
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=250)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        model_reply_start = response_text.find("Peata:")
        if model_reply_start != -1:
            return response_text[model_reply_start + len("Peata:"):]
        return response_text
    
    return "I'm sorry, but both AI models are currently unavailable."