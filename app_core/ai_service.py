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
    print("AI_SERVICE: Entered `load_gemma_model` function.")
    try:
        hf_token = st.secrets.get("HF_TOKEN")
        if not hf_token:
            print("AI_SERVICE: HF_TOKEN not found in secrets.")
            st.warning("Hugging Face token (HF_TOKEN) not found in secrets.")
            return None

        print("AI_SERVICE: HF_TOKEN found. Proceeding to load model.")
        model_id = "google/gemma-2-2b-it"

        print(f"AI_SERVICE: Attempting to download and load tokenizer for '{model_id}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        print("AI_SERVICE: Tokenizer loaded successfully.")

        print(f"AI_SERVICE: Attempting to download and load model for '{model_id}'...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, token=hf_token
        ).to("cpu")
        print("AI_SERVICE: Model loaded successfully and moved to CPU.")
        
        st.session_state.gemma_model = {"model": model, "tokenizer": tokenizer}
        print("AI_SERVICE: Gemma model and tokenizer stored in session state.")
        return st.session_state.gemma_model
    except Exception as e:
        print(f"AI_SERVICE: An exception occurred while loading Gemma: {e}")
        st.error(f"Error loading Gemma model: {e}")
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
    
    # Gemma is now loaded on-demand from the UI.
    # We just check if it's available in the session state.
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