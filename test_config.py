import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
import google.api_core.exceptions

st.set_page_config(layout="wide")
st.title("Configuration Diagnostic Test")

# --- Test 1: Hugging Face Token ---
st.header("1. Hugging Face Secret Test")
try:
    hf_token = st.secrets["HF_TOKEN"]
    if hf_token and hf_token.startswith("hf_"):
        st.success("✅ SUCCESS: Hugging Face token (HF_TOKEN) was found and looks valid.")
        st.write(f"Token starts with: `{hf_token[:7]}...`")
    else:
        st.error("❌ FAILURE: HF_TOKEN was found but appears to be empty or invalid.")
except KeyError:
    st.error("❌ FAILURE: The key 'HF_TOKEN' was NOT found in your secrets.toml file.")
    st.info("Check for typos in the key name and ensure it's not nested under [vertex_ai].")


st.divider()

# --- Test 2: Vertex AI Authentication ---
st.header("2. Vertex AI Credentials Test")
try:
    # Check if vertex_ai secrets exist
    if "vertex_ai" not in st.secrets:
        raise KeyError("The `[vertex_ai]` section is missing from your secrets.toml file.")

    # Initialize Vertex AI using credentials from secrets
    vertexai.init(project=st.secrets.vertex_ai.project_id, credentials=st.secrets.vertex_ai)
    
    # Try to reference the model (this step checks permissions)
    model = GenerativeModel("gemini-1.5-flash")
    st.success("✅ SUCCESS: Vertex AI authenticated and found the 'gemini-1.5-flash' model.")
    st.info("Your permissions appear to be correct.")

except KeyError as e:
    st.error(f"❌ FAILURE: A required key is missing from your secrets file. Error: {e}")
except google.api_core.exceptions.PermissionDenied as e:
    st.error("❌ FAILURE: Google Cloud returned a 'Permission Denied' (403) error.")
    st.info("This confirms the service account is missing the 'Vertex AI User' IAM role.")
    st.code(e, language=None)
except google.api_core.exceptions.NotFound as e:
    st.error("❌ FAILURE: Google Cloud returned a 'Not Found' (404) error.")
    st.info("This confirms the service account cannot find the model, which is almost always a permissions issue.")
    st.code(e, language=None)
except Exception as e:
    st.error(f"❌ FAILURE: An unexpected error occurred during Vertex AI initialization.")
    st.code(e, language=None)