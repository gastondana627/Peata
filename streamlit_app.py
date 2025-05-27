import sys
import streamlit as st

# VERIFY PYTHON PATH - KEEP THIS
print(f"Python Executable used by Streamlit: {sys.executable}")

st.set_page_config(page_title="Animal Shelter", layout="wide")  # MUST come right after importing streamlit

print(f"Python Executable used by Streamlit: {sys.executable}")

# YOUR HELPER FUNCTION HERE
def get_media_as_base64(path):
    if not os.path.exists(path):
        # print(f"Warning: Media file not found at {path}") # Optional: for debugging
        return None
    with open(path, "rb") as media_file:
        return base64.b64encode(media_file.read()).decode()

st.markdown(
    """
    <style>
    /* Main background color for the entire app (excluding sidebar, which is Streamlit's theme) */
    /* You may need to inspect your browser to find the correct data-testid or class for the main content wrapper. */
    /* Common selectors include: [data-testid="stAppViewContainer"], [data-testid="stApp"] > div > section.main */
    /* Or the auto-generated classes like 'st-emotion-cache-z5fcl4', 'st-emotion-cache-czk54k' etc. */
    /* Try with a broad selector first, then refine if it's too aggressive. */
    .stApp {
        background-color: #1A1A1A; /* A dark grey/almost black for the main background */
    }

    /* Target the main content wrapper explicitly if the .stApp isn't enough */
    /* IMPORTANT: You might need to change these class names based on your browser's inspector! */
    div.st-emotion-cache-z5fcl4,
    div.st-emotion-cache-czk54k,
    div[data-testid="stVerticalBlock"] > div.st-emotion-cache-z5fcl4 { /* Common inner container for content */
        background-color: #1A1A1A !important; /* Keep consistent dark background for main content blocks */
    }

    /* Styling for all text input fields, password fields, and text areas */
    input[type="text"],
    input[type="password"],
    textarea {
        background-color: #ffe0f0 !important; /* Light pink background */
        color: black !important; /* Text color inside input fields */
        border: 1px solid #ff99cc !important; /* Optional: pink border */
        border-radius: 5px !important; /* Optional: rounded corners */
        padding: 10px !important; /* Optional: padding inside the input */
    }
    /* Styling for input fields when they are focused (clicked/typed into) */
    input[type="text"]:focus,
    input[type="password"]:focus,
    textarea:focus {
        border-color: #ff007f !important; /* Darker pink border on focus */
        box-shadow: 0 0 0 0.2rem rgba(255, 105, 180, 0.25) !important; /* Optional: pink glow on focus */
    }

    /* Styling for the file uploader dropzone background */
    div[data-testid="stFileUploadDropzone"] {
        background-color: #ffe0f0 !important; /* Light pink for the dropzone */
        border: 1px solid #ff99cc !important; /* Match border style */
        border-radius: 5px !important; /* Match border-radius */
    }

    /* Styling for the 'Browse files' button itself within the file uploader */
    div[data-testid="stFileUploadDropzone"] button {
        background-color: #ff69b4 !important; /* Hot pink for the button */
        color: white !important;
        border: none !important;
    }

    /* --- Custom Text Classes (for pet names and details in adoption section) --- */
    .pet-name {
        color: #FF69B4 !important; /* Hot pink for pet names */
        font-weight: bold; /* Keep bold style */
    }
    .pet-detail {
        color: #FFC0CB !important; /* Light pink for breed, age etc. */
    }

    /* --- Global Text Colors (to ensure all text is visible against dark background) --- */
    /* This targets various text elements in Streamlit, apply carefully. */
    h1, h2, h3, h4, h5, h6,
    div[data-testid="stText"],
    div[data-testid="stMarkdown"] p,
    div[data-testid="stMarkdown"] li,
    div[data-testid="stMarkdown"] ul,
    div[data-testid="stMarkdown"] ol,
    div[data-testid="stExpanderTitle"], /* For expander titles if used */
    div[data-testid="stLinkButton"], /* For link buttons if used */
    div[data-testid="stAlert"] > div, /* For alert box text */
    div[data-testid="stToast"] > div, /* For toast message text */
    .st-emotion-cache-k3g096.e1f1d6z61 { /* General text, might need inspection for accuracy */
        color: #FFC0CB !important; /* Light pink for general text */
    }

    /* Specific adjustment for labels of input widgets */
    label.st-b6 { /* Label for text_input, selectbox, etc. */
        color: #FFC0CB !important; /* Ensure labels are light pink */
    }
    /* Specific adjustment for options in selectbox/radio buttons */
    .st-emotion-cache-1g8wz9e, /* for selected option */
    .st-emotion-cache-1g8wz9e > div, /* for dropdown options */
    .st-emotion-cache-1v4a6f7, /* for radio buttons */
    .st-emotion-cache-1v4a6f7 > div {
        color: #FFC0CB !important;
    }


    </style>
    """,
    unsafe_allow_html=True,
)


# Diagnostics
print("Contents of st.secrets:")
print(st.secrets)

# Standard library
import os
import json
import hashlib
import datetime
import importlib
import base64
# import sys # Already imported above

# Third-party libraries
import pandas as pd
import toml
import streamlit.components.v1 as components
import requests # Ensure requests is imported if used for forms

# Local imports
# IMPORTANT: Use the new pet_matcher.py
from pet_matcher import find_match, precompute_shelter_image_features, IMAGE_FOLDER_CATS, IMAGE_FOLDER_OTHER
# from datetime import datetime # Already imported above (standard library import)

# --- Check if cv2 (OpenCV) is importable --- # NEW
try:
    import cv2
    print("OpenCV (cv2) is successfully imported within app.py!")  # Success message
except ImportError as e:
    print(f"Error importing cv2 within app.py: {e}")  # Error message if import fails

from google.oauth2 import service_account
from google.cloud import aiplatform # Import aiplatform here

try:
    creds_dict = st.secrets["vertex_ai"]
    credentials_gcp = service_account.Credentials.from_service_account_info(creds_dict)
    print("✅ GCP credentials loaded successfully.")

    # Initialize Vertex AI here, once per session if possible, or use @st.cache_resource
    project_id = creds_dict['project_id']
    # Ensure 'location' is in your secrets.toml or define a default
    location = creds_dict.get('location', "us-central1") # Get location from secrets or default

    @st.cache_resource
    def initialize_vertex_ai_cached(_creds, proj_id, loc): # Added underscore to _creds
        aiplatform.init(project=proj_id, location=loc, credentials=_creds)
        print("Vertex AI Initialized (cached)")
        return True # Return a success indicator

    if initialize_vertex_ai_cached(credentials_gcp, project_id, location):
        st.sidebar.success("✅ GCP/Vertex AI initialized.") # Use sidebar for less intrusive messages
    else:
        st.sidebar.error("❌ GCP/Vertex AI failed to initialize.")

except Exception as e:
    print(f"❌ Failed to load GCP credentials or initialize Vertex AI: {e}")
    st.error(f"❌ Failed to load GCP credentials or initialize Vertex AI: {e}")

import urllib.parse  # Add this line here

# --- IMAGE FOLDERS ---
# These are now imported from pet_matcher, but you can define them here too if you prefer
# IMAGE_FOLDER_CATS = "img/cats"
# IMAGE_FOLDER_OTHER = "img/other"
ADOPTION_WEBSITE = "https://www.sa.gov/Directory/Departments/ACS/Adopt/Pet-Search"


# --- LOAD DATA (for lost_pets database.json) ---
def load_data():
    try:
        with open("database.json", "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        # Create an empty database.json if it doesn't exist
        initial_data = {"lost_pets": []}
        with open("database.json", "w") as f:
            json.dump(initial_data, f, indent=4)
        return initial_data
    except json.JSONDecodeError:
        print("Warning: database.json is empty or malformed. Resetting.")
        initial_data = {"lost_pets": []}
        with open("database.json", "w") as f:
            json.dump(initial_data, f, indent=4)
        return initial_data


def save_data(data):
    with open("database.json", "w") as f:
        json.dump(data, indent=4, fp=f)

# --- LOAD CREDENTIALS (reads from credentials.json) ---
def load_credentials():
    try:
        # Load the single credentials.json file
        with open("credentials.json", "r") as f:
            data = json.load(f)

        # Separate credentials and leaderboard from the loaded data
        # All keys in data are user accounts EXCEPT 'leaderboard'
        credentials = {k: v for k, v in data.items() if k != 'leaderboard'}
        leaderboard = data.get('leaderboard', {}) # Get leaderboard, or an empty dict if not present
        return credentials, leaderboard
    except FileNotFoundError:
        # If credentials.json doesn't exist, create an empty one and return empty dicts
        initial_creds = {"leaderboard": {}}
        with open("credentials.json", "w") as f:
            json.dump(initial_creds, f, indent=4)
        return {}, {}
    except json.JSONDecodeError:
        # Handle case where file exists but is empty or malformed JSON
        print("Warning: credentials.json found but is empty or malformed. Starting fresh.")
        initial_creds = {"leaderboard": {}}
        with open("credentials.json", "w") as f:
            json.dump(initial_creds, f, indent=4)
        return {}, {}

# --- SAVE CREDENTIALS (writes to credentials.json) ---
def save_credentials(credentials, leaderboard):
    # Combine credentials (user accounts) and leaderboard into a single dictionary
    # It's important to put the leaderboard back into the 'data' structure before saving
    data_to_save = {**credentials, 'leaderboard': leaderboard}
    with open("credentials.json", "w") as f:
        json.dump(data_to_save, indent=4, fp=f)

# --- HASH PASSWORD (SIMPLE) ---
def hash_password(password):
    # NEVER DO THIS IN PRODUCTION, USE BCRYPT OR SCRYPT
    return hashlib.sha256(password.encode()).hexdigest()

# --- LOAD SECRETS TOML ---
def load_secrets_toml():
    try:
        return toml.load("secrets.toml")
    except FileNotFoundError:
        print("Error: secrets.toml not found.")
        return None

# Sample data for animals
animals = [
    # ... your animal data ...
    {"name": "Hay Hay", "breed": "Chicken", "age": 1, "type": "Other", "image": "Hay Hay.jpg", "description": "A lovely chicken looking for a home."},
    # Cats
    {"name": "Ocean", "breed": "Domestic Shorthair", "age": 3, "image": "Ocean.jpg"},
    {"name": "Marie", "breed": "Domestic Shorthair", "age": "3", "image": "Marie.jpg"},
    {"name": "Tess", "breed": "Domestic Shorthair", "age": 2, "image": "Tess.jpg"},
    {"name": "Peeta", "breed": "Domestic Shorthair", "age": .2, "image": "Peeta.jpg"},
    {"name": "Theodore", "breed": "Domestic Shorthair", "age": 2, "image": "Theodore.jpg"},
    {"name": "Milo", "breed": "Domestic Shorthair", "age": "3", "image": "Milo.jpg"},
    {"name": "Rudy Valentino", "breed": "Domestic Shorthair", "age": 1, "image": "Rudy Valentino.jpg"},
    {"name": "Kitty", "breed": "Domestic Shorthair", "age": .55, "image": "Kitty.jpg"},
    {"name": "Sunscreen", "breed": "Domestic Shorthair", "age": .2, "image": "Sunscreen.jpg"},
    {"name": "Sundress", "breed": "Domestic Shorthair", "age": .2, "image": "Sundress.jpg"},
    {"name": "Sunshine", "breed": "Domestic Shorthair", "age": .2, "image": "Sunshine.jpg"},
    {"name": "Sundae", "breed": "Domestic Shorthair", "age": .2, "image": "Sundae.jpg"},
    {"name": "Luna", "breed": "Domestic Shorthair", "age": 3, "image": "Luna.jpg"},
    {"name": "Sherbert", "breed": "Domestic Shorthair", "age": .25, "image": "Sherbert.jpg"},
    {"name": "Kenny", "breed": "Domestic Shorthair", "age": 1, "image": "Kenny.jpg"},
    {"name": "Jewels", "breed": "Domestic Shorthair", "age": 1, "image": "Jewels.jpg"},
    {"name": "Sasha_36", "breed": "Domestic Shorthair", "age": 2, "image": "Sasha_36.jpg"},
    {"name": "Hank", "breed": "Domestic Shorthair", "age": "4", "image": "Hank.jpg"},
    {"name": "Sakura", "breed": "Domestic Shorthair", "age": 3, "image": "Sakura.jpg"},
    {"name": "Khaleesi", "breed": "Domestic Shorthair", "age": 8, "image": "Khaleesi.jpg"},
    {"name": "Esme", "breed": "Domestic Shorthair", "age": 16, "image": "Esme.jpg"},
    {"name": "GiGi", "breed": "Domestic Shorthair", "age": "13", "image": "GiGi_07.jpg"},
    {"name": "Tang", "breed": "Domestic Shorthair", "age": 2, "image": "Tang.jpg"},
    {"name": "Toothie", "breed": "Domestic Shorthair", "age": 1, "image": "Toothie.jpg"},
    {"name": "Daisy", "breed": "Domestic Shorthair", "age": 1, "image": "Daisy.jpg"},
    {"name": "Pooper", "breed": "Domestic Shorthair", "age": "2", "image": "Pooper.jpg"},
    {"name": "Rebecca", "breed": "Domestic Shorthair", "age": 1.75, "image": "Rebecca.jpg"},
    {"name": "Malibu", "breed": "Domestic Shorthair", "age": 3, "image": "Malibu.jpg"},
    {"name": "Orengga", "breed": "Domestic Shorthair", "age": 2, "image": "Orengga.jpg"},
    {"name": "Bulag", "breed": "Domestic Shorthair", "age": "2", "image": "Bulag.jpg"},
    {"name": "Clarisse", "breed": "Domestic Shorthair", "age": 2, "image": "Clarisse.jpg"},
    {"name": "Ash", "breed": "Domestic Shorthair", "age": 3, "image": "Ash.jpg"},
    {"name": "Sawyer", "breed": "Domestic Shorthair", "age": .75, "image": "Sawyer.jpg"},
    {"name": "Gemini", "breed": "Domestic Shorthair", "age": "6", "image": "Gemini.jpg"},
    {"name": "Sweet Sweet", "breed": "Domestic Shorthair", "age": .2, "image": "Sweet Sweet.jpg"},
    {"name": "Gummy Bear", "breed": "Domestic Shorthair", "age": .2, "image": "Gummy Bear.jpg"},
    {"name": "Coffee", "breed": "Domestic Shorthair", "age": .2, "image": "Coffee.jpg"},
    {"name": "Nala", "breed": "Domestic Shorthair", "age": .2, "image": "Nala.jpg"},
    {"name": "George", "breed": "Domestic Shorthair", "age": .75, "image": "George.jpg"},
    {"name": "Felix", "breed": "Domestic Shorthair", "age": 5, "image": "Felix.jpg"},
    {"name": "Kayla", "breed": "Domestic Shorthair", "age": 6, "image": "Kayla.jpg"},
    {"name": "Gullum", "breed": "Domestic Shorthair", "age": "1", "image": "Gullum.jpg"},
    {"name": "Chloe", "breed": "Domestic Shorthair", "age": .5, "image": "Chloe.jpg"},
    {"name": "Sasha_95", "breed": "Domestic Shorthair", "age": 1, "image": "Sasha_95.jpg"},
    {"name": "Itti-Bitty", "breed": "Domestic Shorthair", "age": 7, "image": "Itti-Bitty.jpg"},
    {"name": "Chad", "breed": "Domestic Shorthair", "age": .75, "image": "Chad.jpg"},
    {"name": "Stormy", "breed": "Domestic Shorthair", "age": 2, "image": "Stormy.jpg"},
    {"name": "Diamond", "breed": "Domestic Shorthair", "age": 2, "image": "Diamond.jpg"},
    {"name": "Cali", "breed": "Domestic Shorthair", "age": 3, "image": "Cali.jpg"},
    {"name": "Orange Pop", "breed": "Domestic Shorthair", "age": "3", "image": "Orange Pop.jpg"},
    {"name": "Spike", "breed": "Domestic Shorthair", "age": 4, "image": "Spike.jpg"},
    {"name": "Maria", "breed": "Domestic Shorthair", "age": 1, "image": "Maria.jpg"},
    {"name": "Maevyth", "breed": "Domestic Shorthair", "age": .12, "image": "Maevyth.jpg"},
    {"name": "Cheese", "breed": "Domestic Shorthair", "age": "1", "image": "Cheese.jpg"},
    {"name": "Binx", "breed": "Domestic Shorthair", "age": 3, "image": "Binx.jpg"},
    {"name": "Patches", "breed": "Domestic Shorthair", "age": 8, "image": "Patches.jpg"},
    {"name": "William", "breed": "Domestic Shorthair", "age": 2, "image": "William.jpg"},
    {"name": "Fran", "breed": "Domestic Shorthair", "age": .55, "image": "Fran.jpg"},
    {"name": "Celeste", "breed": "Domestic Shorthair", "age": 1, "image": "Celeste.jpg"},
    {"name": "Dotty", "breed": "Domestic Shorthair", "age": 1.5, "image": "Dotty.jpg"},
    {"name": "Bitty", "breed": "Domestic Shorthair", "age": 1.5, "image": "Bitty.jpg"},
    {"name": "Rascal", "breed": "Domestic Shorthair", "age": .75, "image": "Rascal.jpg"},
    {"name": "Kendall", "breed": "Domestic Shorthair", "age": 2, "image": "Kendall.jpg"},
    {"name": "Red", "breed": "Domestic Shorthair", "age": 1, "image": "Red.jpg"},
    {"name": "Meelo", "breed": "Domestic Shorthair", "age": 8, "image": "Meelo.jpg"},
    {"name": "Melina", "breed": "Domestic Shorthair", "age": "1", "image": "Melina.jpg"},
    {"name": "Zoey", "breed": "Domestic Shorthair", "age": .1, "image": "Zoey.jpg"},
    {"name": "Max", "breed": "Domestic Shorthair", "age": .1, "image": "Max.jpg"},
    {"name": "Tulip", "breed": "Domestic Shorthair", "age": .1, "image": "Tulip.jpg"},
    {"name": "Evan", "breed": "Domestic Shorthair", "age": .1, "image": "Evan.jpg"},
    {"name": "Oscar", "breed": "Domestic Shorthair", "age": .1, "image": "Oscar.jpg"},
    {"name": "Bruno", "breed": "Domestic Shorthair", "age": .1, "image": "Bruno.jpg"},
    {"name": "Kaliegh", "breed": "Domestic Shorthair", "age": .1, "image": "Kaliegh.jpg"},
    {"name": "Willow", "breed": "Domestic Shorthair", "age": .1, "image": "Willow.jpg"},
    {"name": "Bluffy", "breed": "Domestic Shorthair", "age": 10, "image": "Bluffy.jpg"},
    {"name": "Onyx", "breed": "Domestic Shorthair", "age": "Unknown", "image": "Onyx.jpg"},
    {"name": "GiGi_45", "breed": "Domestic Shorthair", "age": 3, "image": "GiGi_45.jpg"},
    {"name": "Pico De Gato", "breed": "Domestic Shorthair", "age": "7", "image": "Pico De Gato.jpg"},
    {"name": "Micah", "breed": "Domestic Shorthair", "age": 4, "image": "Micah.jpg"},
    {"name": "Sunny Blaze", "breed": "Domestic Shorthair", "age": 6, "image": "Sunny Blaze.jpg"},
    {"name": "Moon", "breed": "Domestic Shorthair", "age": 2, "image": "Moon.jpg"},
    {"name": "Tina", "breed": "Domestic Shorthair", "age": .55, "image": "Tina.jpg"},
    {"name": "Alfie", "breed": "Domestic Shorthair", "age": "Unknown", "image": "Alfie.jpg"},
    {"name": "Lupe", "breed": "Domestic Shorthair", "age": 2, "image": "Lupe.jpg"},
    {"name": "Spring", "breed": "Domestic Shorthair", "age": 2, "image": "Spring.jpg"},
    {"name": "Biniyam", "breed": "Domestic Shorthair", "age": 2, "image": "Biniyam.jpg"},
    {"name": "Jellybean", "breed": "Domestic Shorthair", "age": 1, "image": "Jellybean.jpg"},
    {"name": "Mikayla", "breed": "Domestic Shorthair", "age": 1, "image": "Mikayla.jpg"},
    {"name": "Persephone", "breed": "Domestic Shorthair", "age": 4, "image": "Persephone.jpg"},
    {"name": "Vienna", "breed": "Domestic Shorthair", "age": "3", "image": "Vienna.jpg"},
    {"name": "Gimli", "breed": "Domestic Shorthair", "age": 1, "image": "Gimli.jpg"},
    {"name": "Bluffy", "breed": "Domestic Shorthair", "age": 1, "image": "Bluffy.jpg"},
    {"name": "Frankie", "breed": "Domestic Shorthair", "age": 3, "image": "Frankie.jpg"},
    {"name": "Bow", "breed": "Domestic Shorthair", "age": "3", "image": "Bow.jpg"},
    {"name": "Gary", "breed": "Domestic Shorthair", "age": 6, "image": "Gary.jpg"},
    {"name": "Rachel", "breed": "Domestic Shorthair", "age": 2, "image": "Rachel.jpg"},
    {"name": "Molly", "breed": "Domestic Shorthair", "age": 2, "image": "Molly.jpg"},
    {"name": "Danny", "breed": "Domestic Shorthair", "age": "2", "image": "Danny.jpg"},
    {"name": "Angel", "breed": "Domestic Shorthair", "age": 2, "image": "Angel.jpg"},
    {"name": "Canyon", "breed": "Domestic Shorthair", "age": 4, "image": "Canyon.jpg"},
    {"name": "Mechi", "breed": "Domestic Shorthair", "age": 1, "image": "Mechi.jpg"},
    {"name": "Bailey", "breed": "Domestic Shorthair", "age": "1", "image": "Bailey.jpg"},
    {"name": "OHara", "breed": "Domestic Shorthair", "age": 11, "image": "OHara.jpg"},
    {"name": "Childa", "breed": "Domestic Shorthair", "age": 2, "image": "Childa.jpg"},
    {"name": "Nana", "breed": "Domestic Shorthair", "age": 3,"image": "Nana.jpg"},
    {"name": "Love", "breed": "Domestic Shorthair", "age": "1", "image": "Love.jpg"},
    {"name": "Ruby", "breed": "Domestic Shorthair", "age": 4, "image": "Ruby.jpg"},
    {"name": "Fluffy", "breed": "Domestic Shorthair", "age": 1, "image": "Fluffy.jpg"},
    {"name": "Emory", "breed": "Domestic Shorthair", "age": 2, "image": "Emory.jpg"},
    {"name": "Alex", "breed": "Domestic Shorthair", "age": "2", "image": "Alex.jpg"},
    {"name": "Nelly", "breed": "Domestic Shorthair", "age": 1, "image": "Nelly.jpg"},
    {"name": "Ash", "breed": "Domestic Shorthair", "age": 2, "image": "Ash_88.jpg"},
    {"name": "Eowyn", "breed": "Domestic Shorthair", "age": 1.5, "image": "Eowyn.jpg"},
    {"name": "Sunny", "breed": "Domestic Shorthair", "age": "2", "image": "Sunny.jpg"},
    {"name": "German", "breed": "Domestic Shorthair", "age": 2, "image": "German.jpg"},
    {"name": "Joy", "breed": "Domestic Shorthair", "age": 2, "image": "Joy.jpg"},
    {"name": "Rhea", "breed": "Domestic Shorthair", "age": 1.25, "image": "Rhea.jpg"},
    {"name": "Lovie", "breed": "Domestic Shorthair", "age": "1", "image": "Lovie.jpg"},
    {"name": "Tom", "breed": "Domestic Shorthair", "age": "3", "image": "Tom.jpg"},
    {"name": "May", "breed": "Domestic Shorthair", "age": .75, "image": "May.jpg"},
    {"name": "Tigger", "breed": "Domestic Shorthair", "age": "Unknown", "image": "Tiger.jpg"},
    {"name": "Simon", "breed": "Domestic Shorthair", "age": "Unknown", "image": "Simon.jpg"},
    {"name": "Knight", "breed": "Domestic Shorthair", "age": .30, "image": "Knight.jpg"},
    {"name": "Tommy", "breed": "Domestic Shorthair", "age": 3, "image": "Tommy.jpg"},
    {"name": "Rocko", "breed": "Domestic Shorthair", "age": 5, "image": "Rocko.jpg"},
    {"name": "Motzey", "breed": "Domestic Shorthair", "age": 1.2, "image": "Motzey.jpg"},
    {"name": "Eli", "breed": "Domestic Shorthair", "age": 6, "image": "Eli.jpg"},
    {"name": "Lola", "breed": "Domestic Shorthair", "age": 5, "image": "Lola.jpg"},
    {"name": "Lily", "breed": "Domestic Shorthair", "age": 2.1, "image": "Lily.jpg"},
    {"name": "Whitey", "breed": "Domestic Shorthair", "age": 2.1, "image": "Whitey.jpg"},
    {"name": "Stewart", "breed": "Domestic Shorthair", "age": 8, "image": "Stewart.jpg"},
    {"name": "Mowgli", "breed": "Domestic Shorthair", "age": 4, "image": "Mowgli.jpg"},
    {"name": "Peach", "breed": "Domestic Shorthair", "age": 1.2, "image": "Peach.jpg"},
    {"name": "Jada", "breed": "Domestic Shorthair", "age": 2.2, "image": "Jada.jpg"},
    {"name": "Piper", "breed": "Domestic Shorthair", "age": 8, "image": "Piper.jpg"},
    {"name": "Ollie", "breed": "Domestic Shorthair", "age": .75, "image": "Ollie.jpg"},
    {"name": "Sammy", "breed": "Domestic Shorthair", "age": 2.25, "image": "Sammy.jpg"},
    {"name": "Coco", "breed": "Domestic Shorthair", "age": 2.25, "image": "Coco.jpg"},
    {"name": "Colby", "breed": "Domestic Shorthair", "age": 8, "image": "Colby.jpg"},
    {"name": "Orange", "breed": "Domestic Shorthair", "age": 1.25, "image": "Orange.jpg"},
    {"name": "Miracle", "breed": "Domestic Shorthair", "age": .8, "image": "Miracle.jpg"},
    {"name": "Elliot", "breed": "Domestic Shorthair", "age": 2.75, "image": "Elliot.jpg"},
    {"name": "Victory Freedom", "breed": "Domestic Shorthair", "age": 3, "image": "Victory Freedom.jpg"},
    {"name": "Kylee", "breed": "Domestic Shorthair", "age": 2, "image": "Kylee.jpg"},
    {"name": "Chicken", "breed": "Domestic Shorthair", "age": 3, "image": "Chicken.jpg"},
    {"name": "Mouse_20", "breed": "Domestic Shorthair", "age": "7", "image": "Mouse_20.jpg"},
    {"name": "Sam", "breed": "Domestic Shorthair", "age": 8, "image": "Sam.jpg"}
]

# --- Precompute shelter animal features once at startup ---
# This dictionary will hold the pHashes and ORB features of all known animals.
@st.cache_resource
def get_shelter_features_db():
    print("Precomputing shelter animal features for the first time...")
    # Pass the actual image folders to the precomputation function
    # You can add more folders if you have them, e.g., [IMAGE_FOLDER_CATS, IMAGE_FOLDER_OTHER, "img/dogs"]
    return precompute_shelter_image_features([IMAGE_FOLDER_CATS, IMAGE_FOLDER_OTHER])

SHELTER_FEATURES_DB = get_shelter_features_db()


# --- Pagination ---
ANIMALS_PER_PAGE = 20
NUM_PAGES = (len(animals) + ANIMALS_PER_PAGE - 1) // ANIMALS_PER_PAGE

# Streamlit setup for custom CSS
st.markdown(
    """
<style>
.social-button {
    background-color: #4CAF50; /* Green */
    border: none;
    color: white;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 5px;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- User Authentication Setup (Session State Initialization) ---
if 'username' not in st.session_state:
    st.session_state['username'] = None
if "view" not in st.session_state:
    st.session_state.view = "login" # default view

def switch_view(new_view):
    st.session_state.view = new_view
    ##st.rerun()

## Def Login Function
def login():
    # --- TASK 3: CODE TO DISPLAY VIDEO/ANIMATION ---
    # Replace with your actual file name and path
    animation_path = "assets/5_second_promo.mp4"  # Or "assets/your_animation.gif"
    media_base64 = get_media_as_base64(animation_path)
    video_html = "" # Initialize to empty string

    if media_base64:
        if animation_path.lower().endswith((".mp4", ".webm")):
            desired_max_width = "600px"
            
            video_html = f"""
            <div style="display: flex; justify-content: center; margin-bottom: 25px; margin-top: 15px;">
                <video autoplay loop muted controls playsinline
                       style="width: 100%; max-width: {desired_max_width}; height: auto; border-radius: 10px;">
                    <source src="data:video/mp4;base64,{media_base64}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            """
        elif animation_path.lower().endswith(".gif"):
            video_html = f"""
            <div style="display: flex; justify-content: center; margin-bottom: 25px; margin-top: 15px;">
                <img src="data:image/gif;base64,{media_base64}" alt="Login Animation" width="250" style="border-radius: 10px;">
            </div>
            """
        # Add more conditions here for other types if needed (e.g., .ogg for video/ogg)

        if video_html: # Only display if video_html was populated
            st.markdown(video_html, unsafe_allow_html=True)
    else:
        # Optional: You can add a placeholder or warning if the media isn't found,
        # especially during development. For production, you might just want it to be blank.
        # st.warning(f"Login animation not found at {animation_path}. Please check the path.")
        pass
    # --- END OF TASK 3 CODE ---

    # Your existing login page styling for inputs (if any, otherwise remove this markdown block)
    st.markdown(
        """
        <style>
        /* Ensure your specific input styles for login are here or in the main CSS block */
        input[type="text"],
        input[type="password"] {
            background-color: #ffe0f0 !important;
            color: black !important;
            border: 1px solid #ff99cc !important;
            border-radius: 5px !important;
            padding: 10px !important;
        }
        input[type="text"]:focus,
        input[type="password"]:focus {
            border-color: #ff007f !important;
            box-shadow: 0 0 0 0.2rem rgba(255, 105, 180, 0.25) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Welcome to the Local Animal Shelter!")
    st.write("Please sign in or create an account.")
    username = st.text_input("Username:", key="login_username_input")
    password = st.text_input("Password:", type="password", key="login_password_input")

    if st.button("Sign In", key="login_button"):
        credentials, leaderboard = load_credentials() # Ensure load_credentials is defined
        if username in credentials and credentials[username]["password"] == hash_password(password): # Ensure hash_password is defined
            st.session_state['username'] = username
            st.session_state['leaderboard'] = leaderboard
            st.success("Logged in successfully!")
            st.session_state.view = "main_app"
            st.rerun()
        else:
            st.error("Invalid credentials.")
    st.button("Create Account", on_click=lambda: switch_view("signup"), key="go_to_signup_button") # Ensure switch_view is defined


def create_account():
    # --- TASK 3: CODE TO DISPLAY VIDEO/ANIMATION (Refined for Responsiveness) ---
    animation_path = "assets/5_second_promo.mp4"  # Your video file
    media_base64 = get_media_as_base64(animation_path)
    video_html = "" # Initialize to empty string

    if media_base64:
        if animation_path.lower().endswith((".mp4", ".webm")):
            # Use the same desired_max_width as your login page for consistency
            desired_max_width = "600px"  # Make this the same as in login() (e.g., "600px", "450px")

            # This f-string should now be clean of invalid comments and use responsive styling
            video_html = f"""
            <div style="display: flex; justify-content: center; margin-bottom: 25px; margin-top: 15px;">
                <video autoplay loop muted controls playsinline
                       style="width: 100%; max-width: {desired_max_width}; height: auto; border-radius: 10px;">
                    <source src="data:video/mp4;base64,{media_base64}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            """

        elif animation_path.lower().endswith(".gif"):
            # Current GIF part with fixed width.
            # To make GIF responsive, apply similar style changes as the video:
            # desired_max_width_gif = "400px"; # example for GIF
            # video_html = f"""
            # <div style="display: flex; justify-content: center; margin-bottom: 25px; margin-top: 15px;">
            #     <img src="data:image/gif;base64,{media_base64}" alt="Signup Animation"
            #          style="width: 100%; max-width: {desired_max_width_gif}; height: auto; border-radius: 10px;">
            # </div>
            # """
            # Keeping your original fixed-width GIF code for now:
            video_html = f"""
            <div style="display: flex; justify-content: center; margin-bottom: 25px; margin-top: 15px;">
                <img src="data:image/gif;base64,{media_base64}" alt="Signup Animation" width="250" style="border-radius: 10px;">
            </div>
            """
        # Add more conditions here for other types if needed

        if video_html: # Only display if video_html was populated
            st.markdown(video_html, unsafe_allow_html=True)
    else:
        # Optional warning for development
        # st.warning(f"Signup animation not found at {animation_path}. Please check the path.")
        pass

    # ... rest of your create_account function (st.title, inputs, buttons, etc.)
    # --- END OF TASK 3 CODE ---

    # Your existing signup page styling for inputs (if any, otherwise remove this markdown block)
    st.markdown(
        """
        <style>
        /* Ensure your specific input styles for signup are here or in the main CSS block */
        input[type="text"],
        input[type="password"],
        textarea {
            background-color: #ffe0f0 !important;
            color: black !important;
            border: 1px solid #ff99cc !important;
            border-radius: 5px !important;
            padding: 10px !important;
        }
        input[type="text"]:focus,
        input[type="password"]:focus,
        textarea:focus {
            border-color: #ff007f !important;
            box-shadow: 0 0 0 0.2rem rgba(255, 105, 180, 0.25) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Create Your Account")
    new_username = st.text_input("New Username:", key="signup_username_input")
    new_password = st.text_input("New Password:", type="password", key="signup_password_input")

    st.write("Tell us about your ideal pet:")
    has_children = st.checkbox("Do you have children?", key="signup_children")
    has_other_pets = st.checkbox("Do you have other pets?", key="signup_other_pets")
    preferred_size = st.selectbox("Preferred size:", ["Small", "Medium", "Large"], key="signup_size")
    activity_level = st.select_slider("Preferred activity level:", options=["Low", "Moderate", "High"], key="signup_activity")
    preferred_pet = st.selectbox("Do you prefer:", ["Cats", "Dogs", "Both"], key="signup_pet")

    if st.button("Create Account", key="signup_button"):
        credentials, leaderboard = load_credentials() # Ensure load_credentials is defined
        if new_username in credentials:
            st.error("Username already exists.")
        else:
            hashed_password = hash_password(new_password) # Ensure hash_password is defined
            credentials[new_username] = {
                "password": hashed_password,
                "has_children": has_children,
                "has_other_pets": has_other_pets,
                "preferred_size": preferred_size,
                "activity_level": activity_level,
                "preferred_pet": preferred_pet,
                "share_history": [],
                "lost_pet_history": [],
                "found_pet_history": [],
            }
            save_credentials(credentials, leaderboard) # Ensure save_credentials is defined
            st.success("Account created successfully! Please log in.")
            st.session_state.view = "login"
            ##st.rerun()
    st.button("Back to Login", on_click=lambda: switch_view("login"), key="back_to_login_button") # Ensure switch_view is defined


# --- Share Pet Function (Persistence Verified) ---
def share_pet(pet_name):
    st.info(f"You shared {pet_name}'s profile! +5 Points")
    username = st.session_state['username']
    if username:
        credentials, leaderboard = load_credentials()
        user_data = credentials.get(username, {})
        share_history = user_data.get('share_history', [])
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

        share_history.append({
            "pet_name": pet_name,
            "timestamp": timestamp_str,
            "points_earned": 5,
        })
        credentials[username]['share_history'] = share_history
        if username not in leaderboard:
            leaderboard[username] = 0
        leaderboard[username] += 5
        st.session_state['leaderboard'] = leaderboard # Update session state leaderboard
        save_credentials(credentials, leaderboard) # SAVE - Point persistence
        # Share receipt
        st.success(f"Share Receipt:")
        st.write(f"- Pet: {pet_name}")
        st.write(f"- Date/Time: {timestamp_str}")
        st.write(f"- Points Earned: 5")

# --- Report Lost Pet Function (Points Added) ---
def report_lost_pet_points(pet_name, pet_breed):
    st.info(f"You reported a lost pet: {pet_name}! +5 Points")
    username = st.session_state['username']
    if username:
        credentials, leaderboard = load_credentials()
        user_data = credentials.get(username, {})
        lost_pet_history = user_data.get('lost_pet_history', [])

        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

        lost_pet_history.append({
            "pet_name": pet_name,
            "pet_breed": pet_breed,
            "timestamp": timestamp_str,
            "points_earned": 5,
        })
        credentials[username]['lost_pet_history'] = lost_pet_history

        if username not in leaderboard:
            leaderboard[username] = 0
        leaderboard[username] += 5
        st.session_state['leaderboard'] = leaderboard # Update session state leaderboard

        save_credentials(credentials, leaderboard) # SAVE - Point persistence

        # Report receipt
        st.success(f"Lost Pet Reported Successfully!") # Updated success message
        st.write(f"Lost Pet Report Receipt:")
        st.write(f"- Pet Name: {pet_name}")
        st.write(f"- Pet Breed: {pet_breed}")
        st.write(f"- Date/Time: {timestamp_str}")
        st.write(f"- Points Earned: 5")
    return # Added return to stop further execution in button click

# --- Report Found Pet Function (Points Added) ---
def report_found_pet_points():
    st.header("Lost Pet Reunification Hub")
    st.markdown(
        """
        This hub helps reunite lost pets with their owners.
        **Report a found pet:** Upload a photo. AI matching will attempt to find the owner. Earn points for reporting!
        **Report a lost pet:** Fill the form below with pet info and photo. Earn points for reporting! We'll notify you of matches.
        """
    )


    st.info(f"You reported a found pet! +5 Points")
    username = st.session_state['username']
    if username:
        credentials, leaderboard = load_credentials()
        user_data = credentials.get(username, {})
        found_pet_history = user_data.get('found_pet_history', []) # Get found pet history

        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

        found_pet_history.append({ # Save to found pet history
            "timestamp": timestamp_str,
            "points_earned": 5,
        })
        credentials[username]['found_pet_history'] = found_pet_history # Update credentials

        if username not in leaderboard:
            leaderboard[username] = 0
        leaderboard[username] += 5
        st.session_state['leaderboard'] = leaderboard # Update session state leaderboard

        save_credentials(credentials, leaderboard) # SAVE - Point persistence

        # Report receipt
        st.success(f"Found Pet Reported Successfully!") # Updated success message
        st.write(f"Found Pet Report Receipt:")
        st.write(f"- Date/Time: {timestamp_str}")
        st.write(f"- Points Earned: 5")
    return # Added return to stop further execution in button click


# --- Display User History Function ---
def display_user_history():
    username = st.session_state['username']
    if username:
        credentials, leaderboard = load_credentials()
        user_data = credentials.get(username, {})
        share_history = user_data.get('share_history', [])
        lost_pet_history = user_data.get('lost_pet_history', []) # Retrieve lost pet history
        found_pet_history = user_data.get('found_pet_history', []) # Retrieve found pet history
        total_points = st.session_state['leaderboard'].get(username, 0)

        st.sidebar.header(f"Your History, {username}")
        st.sidebar.subheader(f"Total Points: {total_points}")

        if share_history:
            st.sidebar.write("---")
            st.sidebar.subheader("Share Log:")
            df_share = pd.DataFrame(share_history)
            st.sidebar.dataframe(df_share)
        if lost_pet_history: # Display lost pet history if available
            st.sidebar.write("---")
            st.sidebar.subheader("Lost Pet Reports:")
            df_lost_pet = pd.DataFrame(lost_pet_history)
            st.sidebar.dataframe(df_lost_pet)
        if found_pet_history: # Display found pet history if available
            st.sidebar.write("---")
            st.sidebar.subheader("Found Pet Reports:")
            df_found_pet = pd.DataFrame(found_pet_history)
            st.sidebar.dataframe(df_found_pet)
        else:
            st.sidebar.write("No shares or pet reports yet.")
    else:
        st.sidebar.info("Please log in to see your history.")


# --- Main Application Flow Control ---
if st.session_state.view == "login":
    login()
elif st.session_state.view == "signup":
    create_account()
elif st.session_state.view == "main_app" and st.session_state['username']:
    # --- Main App Content (after successful login) ---
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # START OF STICKY LOGO DISPLAY BLOCK (This is where you ADD the code)
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    logo_path = "assets/Peata_Logo.svg"  # <<<< CHANGE "your_logo.svg" TO YOUR ACTUAL FILENAME
    logo_base64 = get_media_as_base64(logo_path) # Use your existing function

    if logo_base64:
        mime_type = "image/svg+xml"  # Specific for SVG files

        sticky_logo_html = f"""
            <style>
                .sticky-logo-container {{
                    position: fixed;
                    top: 55px;       /* Adjust as needed */
                    left: 1px;      /* Adjust as needed */
                    z-index: 9999;
                }}
                .sticky-logo-container img {{
                    width: 80px;      /* Adjust logo size */
                    height: auto;
                }}
            </style>
            <div class="sticky-logo-container">
                <img src="data:{mime_type};base64,{logo_base64}" alt="Company Logo">
            </div>
        """
        st.markdown(sticky_logo_html, unsafe_allow_html=True)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # END OF STICKY LOGO DISPLAY BLOCK
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    st.title("Welcome to the Local Animal Shelter!")
    st.write("Here you can meet some of the animals available for adoption.")
    st.header(f"Welcome, {st.session_state['username']}! 🐾")

    # Button to display user history in the sidebar
    display_user_history_button = st.sidebar.button("My History & Points", key="display_history_button")
    if display_user_history_button:
        display_user_history()

    # --- Lost Pet Section (Found Pet Reporting) ---
    st.header("Lost Pet Reunification Hub")
    st.markdown(
        """
        This hub helps reunite lost pets with their owners.
        **Report a found pet:** Upload a photo. AI matching will attempt to find the owner. Earn points for reporting!
        **Report a lost pet:** Fill the form below with pet info and photo. Earn points for reporting! We'll notify you of matches.
        """
    )

    # --- Initialize session state for match related flags ---
    if 'match_message_visible' not in st.session_state:
        st.session_state['match_message_visible'] = False
    if 'report_message_visible' not in st.session_state:
        st.session_state['report_message_visible'] = False
    if 'found_pet_reported' not in st.session_state:
        st.session_state['found_pet_reported'] = False # Flag to track if points were awarded

    if 'potential_match_name' not in st.session_state: # Store potential match name
        st.session_state['potential_match_name'] = None
    if 'awaiting_confirmation' not in st.session_state: # Flag for user confirmation
        st.session_state['awaiting_confirmation'] = False
    if 'last_uploaded_file_hash' not in st.session_state: # To prevent re-processing same image
        st.session_state['last_uploaded_file_hash'] = None

    uploaded_found_pet_file = st.file_uploader("Upload an image of the found pet", type=["png", "jpg", "jpeg"], key="found_pet_uploader")

    if uploaded_found_pet_file is not None:
        # Generate a hash for the uploaded file content to check for changes
        # Read the file content for hashing, then seek back to 0
        file_content = uploaded_found_pet_file.getvalue()
        file_hash = hashlib.sha256(file_content).hexdigest()
        uploaded_found_pet_file.seek(0) # IMPORTANT: Reset stream position after reading for hashing

        if file_hash != st.session_state['last_uploaded_file_hash']:
            # New file uploaded or file changed, reset flags and process
            st.session_state['potential_match_name'] = None
            st.session_state['awaiting_confirmation'] = False
            st.session_state['last_uploaded_file_hash'] = file_hash
            st.session_state['found_pet_reported'] = False # Reset found pet reported flag
            # Also reset any previous confirmation to allow a fresh check
            st.session_state['last_confirmed_match'] = None

        st.image(uploaded_found_pet_file, caption="Found Pet", width=200)

        # Optional debug info
        st.write("Filename:", uploaded_found_pet_file.name)
        st.write("Size (bytes):", len(uploaded_found_pet_file.getvalue()))
        st.write("Upload timestamp:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Only run find_match if not already awaiting confirmation and no potential match yet
        # Pass the precomputed SHELTER_FEATURES_DB to find_match
        if not st.session_state['awaiting_confirmation'] and st.session_state['potential_match_name'] is None:
            with st.spinner("Searching for a match..."):
                match_name = find_match(uploaded_found_pet_file, SHELTER_FEATURES_DB)
                st.session_state['potential_match_name'] = match_name
                if match_name:
                    st.session_state['awaiting_confirmation'] = True
                else:
                    st.info("No immediate match found. Check back later.")

    # Display match confirmation only if a potential match is found and awaiting confirmation
    if st.session_state['potential_match_name'] and st.session_state['awaiting_confirmation']:
        st.success(f"Possible match found: **{st.session_state['potential_match_name']}**")
        st.write("Does this look like your pet? Please confirm to earn points!")

        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Yes, this is my pet!", key="confirm_match_yes"):
                # Check if points were already awarded for this specific confirmation
                # Ensure 'last_confirmed_match' exists in session state
                if 'last_confirmed_match' not in st.session_state:
                    st.session_state['last_confirmed_match'] = None

                if not st.session_state['last_confirmed_match'] == st.session_state['potential_match_name']:
                    report_found_pet_points() # Award points ONLY on explicit confirmation
                    st.session_state['last_confirmed_match'] = st.session_state['potential_match_name'] # Store confirmed match to prevent re-awarding
                    st.success(f"Confirmed! Thank you for helping. Points awarded.")
                else:
                    st.info("Already confirmed this match and points awarded.")

                st.session_state['potential_match_name'] = None # Clear match
                st.session_state['awaiting_confirmation'] = False # Reset confirmation state
                st.session_state['last_uploaded_file_hash'] = None # Allow re-uploading new image
                uploaded_found_pet_file = None # Clear the file uploader state (visual)
                st.rerun() # Rerun to clear messages and potential match UI
        with col_no:
            if st.button("No, this is not my pet.", key="confirm_match_no"):
                st.info("Thank you for clarifying. No points awarded for this match.")
                st.session_state['potential_match_name'] = None # Clear match
                st.session_state['awaiting_confirmation'] = False # Reset confirmation state
                st.session_state['last_uploaded_file_hash'] = None # Allow re-uploading new image
                uploaded_found_pet_file = None # Clear the file uploader state (visual)
                st.rerun() # Rerun to clear messages and potential match UI

    # --- Lost Pet Reporting Form ---
    st.subheader("Report a Lost Pet (Earn Points!)") # Updated subtitle to indicate points
    lost_pet_name = st.text_input("Pet's Name:", key="lost_pet_name_input")
    lost_pet_breed = st.text_input("Pet's Breed:", key="lost_pet_breed_input")
    lost_pet_image = st.file_uploader("Upload lost pet photo", type=["jpg", "jpeg", "png"], key="lost_pet_image_uploader")

    if lost_pet_image:
        st.image(lost_pet_image, caption="Lost Pet", width=200)

    if st.button("Report Lost Pet", key="report_lost_pet_button"):
        if not lost_pet_name or not lost_pet_breed: # Check if name or breed are empty
            st.error("Please provide the pet's name and breed.")
        elif lost_pet_image is None: # Check if image is uploaded
            st.error("Please upload an image of the lost pet to submit the report.") # Display error message
        else: # Proceed with report submission if all info is provided
            data = load_data()
            data["lost_pets"].append({
                "name": lost_pet_name,
                "breed": lost_pet_breed,
                "image": lost_pet_image.name, # Save image name
            })
            save_data(data)
            report_lost_pet_points(lost_pet_name, lost_pet_breed) # Call new function to award points
            st.rerun() # Rerun to clear the form after submission and show message


    # --- Animals Available for Adoption Section ---
    st.header("Animals Available for Adoption")

    num_cols = 4
    cols = st.columns(num_cols)
    pet_index = 0

    # --- Pagination for Animals Section ---
    # Initialize page_animals if not already in session_state
    if 'animals_page_radio' not in st.session_state:
        st.session_state['animals_page_radio'] = 1

    page_animals = st.sidebar.radio("Page", range(1, NUM_PAGES + 1), key="animals_page_radio")
    start_index_animals = (page_animals - 1) * ANIMALS_PER_PAGE
    end_index_animals = start_index_animals + ANIMALS_PER_PAGE
    paged_animals = animals[start_index_animals:end_index_animals]

    for pet in paged_animals:
        try:
            with cols[pet_index % num_cols]:
                st.markdown(f"<p class='pet-name'>{pet['name']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='pet-detail'>Breed: {pet['breed']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='pet-detail'>Age: {pet['age']}</p>", unsafe_allow_html=True)



                image_path = os.path.join(IMAGE_FOLDER_OTHER if pet.get('type') == "Other" else IMAGE_FOLDER_CATS, pet["image"])

                if os.path.exists(image_path):
                    st.image(image_path, width=150)
                else:
                    st.warning(f"Image not found: {image_path}")

                components.html(
                    f"""
                    <a href="{ADOPTION_WEBSITE}" target="_blank">
                    <button style="background-color:Green;color:white;border: none;padding: 10px 20px;text-align: center;text-decoration: none;display: inline-block;font-size: 16px;margin: 4px 2px;cursor: pointer;border-radius: 5px;">Adopt {pet['name']}!</button>
                    </a>
                    """,
                    height=50,
                )

                # --- SHARE BUTTON WITH PROMPT ---
                # Initialize share_prompt and pet_to_share for each pet if not exists
                # Using a unique key for each pet's share state
                share_prompt_key = f'share_prompt_{pet["name"]}'
                pet_to_share_key = f'pet_to_share_{pet["name"]}'

                if share_prompt_key not in st.session_state:
                    st.session_state[share_prompt_key] = False
                if pet_to_share_key not in st.session_state:
                    st.session_state[pet_to_share_key] = None

                if st.button(f"Share {pet['name']}!", key=f"share_animals_{pet['name']}"):
                    st.session_state[pet_to_share_key] = pet['name'] # store the pet name
                    st.session_state[share_prompt_key] = not st.session_state[share_prompt_key] # Toggle the prompt

                if st.session_state.get(share_prompt_key, False) and st.session_state.get(pet_to_share_key) == pet['name']:
                    st.write(f"Share {pet['name']} on:")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button("Facebook", key=f"facebook_share_{pet['name']}"):
                            share_pet(pet['name'])
                            st.session_state[share_prompt_key] = False # Hide prompt after sharing
                            st.rerun() # Rerun to update points and clear share prompt

                    with col2:
                        if st.button("X/Twitter", key=f"twitter_share_{pet['name']}"):
                            share_pet(pet['name'])
                            st.session_state[share_prompt_key] = False
                            st.rerun()

                    with col3:
                        if st.button("Snapchat", key=f"snapchat_share_{pet['name']}"):
                            share_pet(pet['name'])
                            st.session_state[share_prompt_key] = False
                            st.rerun()

                st.write("---")
        except Exception as e:
            st.error(f"Error displaying {pet['name']}: {e}")
        pet_index += 1

    # --- Adopted Animals Section (Initially Empty) ---
    st.header("Adopted Animals") # Section header - initially empty
    st.write("This section will display animals that have already been adopted. (Currently empty)")
    # --- You can add logic here later to display adopted animals if you have a separate list ---
    # --- For now, it's just a header and informational text ---

    # --- Find My Forever Home Challenge ---
    st.header("Find My Forever Home Challenge")

    # Initialize leaderboard if not already in session state (should be loaded on login)
    if 'leaderboard' not in st.session_state:
        credentials, leaderboard = load_credentials()
        st.session_state['leaderboard'] = leaderboard

    # --- Leaderboard ---
    st.subheader("Leaderboard (Top Sharers)")
    leaderboard = st.session_state['leaderboard']
    if leaderboard:
        sorted_leaderboard = sorted(leaderboard.items(), key=lambda item: item[1], reverse=True)
        for rank, (user, points) in enumerate(sorted_leaderboard, start=1):
            st.write(f"{rank}. {user}: {points} points")
    else:
        st.write("No shares yet!")

    # --- Feedback section ---
    st.subheader("🌟 Feedback")

    with st.form("feedback_form"):
        feedback_text = st.text_area("Share your feedback!")
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            if "FORMSPREE_ENDPOINT" in st.secrets: # Access secret directly from st.secrets
                form_endpoint = st.secrets["FORMSPREE_ENDPOINT"]
                try:
                    response = requests.post(form_endpoint, data={"feedback": feedback_text})
                    if response.status_code == 200:
                        st.success("Thank you for your feedback!")
                    else:
                        st.error(f"Error submitting feedback: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.error("Formspree endpoint not found! Make sure that your secrets.toml is properly updated.")



