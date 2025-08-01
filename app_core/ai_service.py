import streamlit as st
import requests
import torch
import vertexai
from vertexai.generative_models import GenerativeModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# This function now reads from a session state variable instead of a real network check.
def check_internet_connection():
    # We default to True (online) if the variable hasn't been set yet.
    return st.session_state.get("is_online", True)

# This function loads the Gemma model from the Hugging Face Hub.
# The @st.cache_resource decorator is CRUCIAL for performance, as it
# prevents the model from being re-downloaded or re-loaded on every user interaction.
# It also handles caching the model in memory for subsequent uses.
@st.cache_resource(show_spinner="Loading Gemma-2 model (this may take a while)...")
def load_gemma_model():
    try:
        # We are now loading the new, second-generation Gemma model.
        # It has the same parameter count but with improved performance and safety.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            quantization_config=bnb_config,
            device_map="auto" # This automatically uses your Mac's MPS for acceleration.
        )
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading Gemma model locally: {e}. Please ensure all dependencies are installed and the model can fit in memory.")
        return None, None

# This function interacts with Google's Vertex AI (Gemini).
# It uses the secrets you've stored in the .streamlit/secrets.toml file.
def get_vertex_ai_response(prompt):
    try:
        vertexai.init(
            project=st.secrets["VERTEX_AI_PROJECT_ID"],
            location=st.secrets["VERTEX_AI_LOCATION"]
        )
        model = GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Vertex AI error: {e}")
        return None

# A helper function to handle the Gemma inference process.
def get_gemma_response_from_model(prompt):
    model, tokenizer = load_gemma_model()
    if model and tokenizer:
        try:
            # Gemma's instruction-tuned model requires a specific chat format.
            formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer.encode(formatted_prompt, add_special_tokens=False, return_tensors="pt")

            # Move inputs to the correct device (MPS for your Mac)
            if torch.backends.mps.is_available():
                inputs = inputs.to("mps")
            else:
                inputs = inputs.to("cpu")

            with torch.no_grad():
                outputs = model.generate(input_ids=inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
            
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # The output includes the prompt, so we extract only the model's reply.
            return response_text.split('<start_of_turn>model\n')[-1].strip()

        except Exception as e:
            st.error(f"Gemma inference error: {e}")
            return "I'm sorry, I'm having trouble processing that request offline."
    return "Gemma model not loaded."

# This is the central orchestration function for the dual-AI chatbot.
# It decides whether to use Vertex AI or Gemma based on connectivity.
def get_chatbot_response(user_prompt):
    st.session_state.ai_mode = "Online (Vertex AI)" # Default mode
    if check_internet_connection():
        # First, try to get a response from Vertex AI.
        response = get_vertex_ai_response(user_prompt)
        if response:
            return response
        else:
            # If Vertex AI fails despite an internet connection, fall back to Gemma.
            st.session_state.ai_mode = "Fallback (Gemma 2)"
            st.warning("Vertex AI failed. Falling back to Gemma 2.")
            return get_gemma_response_from_model(user_prompt)
    else:
        # If there is no internet, use Gemma as the primary model.
        st.session_state.ai_mode = "Offline (Gemma 2)"
        st.info("No internet connection detected. Using Gemma 2 offline.")
        return get_gemma_response_from_model(user_prompt)
