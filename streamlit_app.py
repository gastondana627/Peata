# streamlit_app.py

import sys
print(f"Python Executable used by Streamlit: {sys.executable}")  # VERIFY PYTHON PATH - KEEP THIS

import streamlit as st
st.set_page_config(page_title="Animal Shelter", layout="wide")  # MUST come right after importing streamlit

# Diagnostics
print("Contents of st.secrets:")
print(st.secrets)

# Standard library
import os
import json
import hashlib
import datetime
import importlib
import sys

# Third-party libraries
import pandas as pd
import toml
import streamlit.components.v1 as components

# Local imports
from pet_matcher import find_match  # Ensure this path is correct

# --- Check if cv2 (OpenCV) is importable --- # NEW
try:
    import cv2
    print("OpenCV (cv2) is successfully imported within app.py!") # Success message
except ImportError as e:
    print(f"Error importing cv2 within app.py: {e}") # Error message if import fails

from google.oauth2 import service_account

try:
    creds_dict = st.secrets["vertex_ai"]
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    print("✅ GCP credentials loaded successfully.")
    st.success("✅ GCP credentials loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load GCP credentials: {e}")
    st.error(f"❌ Failed to load GCP credentials: {e}")

import urllib.parse # Add this line here

# --- IMAGE FOLDERS ---
IMAGE_FOLDER_CATS = "img/cats"
IMAGE_FOLDER_OTHER = "img/other"
ADOPTION_WEBSITE = "https://www.sa.gov/Directory/Departments/ACS/Adopt/Pet-Search"

# --- LOAD DATA ---
def load_data():
    with open("database.json", "r") as f:
        data = json.load(f)
        return data

def save_data(data):
    with open("database.json", "w") as f:
        json.dump(data, indent=4, fp=f)

# --- LOAD CREDENTIALS ---
def load_credentials():
    try:
        with open("credentials.json", "r") as f:
            credentials = json.load(f)
            # Load leaderboard from credentials file if it exists
            leaderboard = credentials.get('leaderboard', {})
            return credentials, leaderboard
    except FileNotFoundError:
        return {}, {}
    # Return empty if file not found

# --- SAVE CREDENTIALS ---
def save_credentials(credentials, leaderboard):
    credentials['leaderboard'] = leaderboard # Save leaderboard to credentials
    with open("credentials.json", "w") as f:
        json.dump(credentials, indent=4, fp=f)

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
    {"name": "Sam", "breed": "Domestic Shorthair", "age": 2, "image": "Sam.jpg"},
    {"name": "Mouse_20", "breed": "Domestic Shorthair", "age": "3", "image": "Mouse_20.jpg"},
    {"name": "Chicken", "breed": "Domestic Shorthair", "age": 1, "image": "Chicken.jpg"},
    {"name": "Callie", "breed": "Domestic Shorthair", "age": 4, "image": "Callie.jpg"},
    {"name": "Bobby", "breed": "Domestic Shorthair", "age": 2, "image": "Bobby.jpg"},
    {"name": "Kristi", "breed": "Domestic Shorthair", "age": "3", "image": "Kristi.jpg"},
    {"name": "Kylee", "breed": "Domestic Shorthair", "age": 1, "image": "Kylee.jpg"},
    {"name": "Chip_19", "breed": "Domestic Shorthair", "age": 5, "image": "Chip_19.jpg"},
    {"name": "Chip", "breed": "Domestic Shorthair", "age": 2, "image": "Chip.jpg"},
    {"name": "Lumi", "breed": "Domestic Shorthair", "age": "3", "image": "Lumi.jpg"},
    {"name": "Possum", "breed": "Domestic Shorthair", "age": 1, "image": "Possum.jpg"},
    {"name": "Victory Freedom", "breed": "Domestic Shorthair", "age": 4, "image": "Victory Freedom.jpg"},
    {"name": "Elliot", "breed": "Domestic Shorthair", "age": 2, "image": "Elliot.jpg"},
    {"name": "Miracle", "breed": "Domestic Shorthair", "age": "3", "image": "Miracle.jpg"},
    {"name": "Orange", "breed": "Domestic Shorthair", "age": 1, "image": "Orange.jpg"},
    {"name": "Tricks", "breed": "Domestic Shorthair", "age": 5, "image": "Tricks.jpg"},
    {"name": "Katsuki", "breed": "Domestic Shorthair", "age": 2, "image": "Katsuki.jpg"},
    {"name": "Felicia", "breed": "Domestic Shorthair", "age": "3", "image": "Felicia.jpg"},
    {"name": "Cream", "breed": "Domestic Shorthair", "age": 1, "image": "Cream.jpg"},
    {"name": "Sarabi", "breed": "Domestic Shorthair", "age": 4, "image": "Sarabi.jpg"},
    {"name": "Kar", "breed": "Domestic Shorthair", "age": 2, "image": "Kar.jpg"},
    {"name": "Graystone", "breed": "Domestic Shorthair", "age": "3", "image": "Graystone.jpg"},
    {"name": "Matt", "breed": "Domestic Shorthair", "age": 1, "image": "Matt.jpg"},
    {"name": "Casper", "breed": "Domestic Shorthair", "age": 5, "image": "Casper.jpg"},
    {"name": "Bubba", "breed": "Domestic Shorthair", "age": 2, "image": "Bubba.jpg"},
    {"name": "Goose", "breed": "Domestic Shorthair", "age": "3", "image": "Goose.jpg"},
    {"name": "Annie", "breed": "Domestic Shorthair", "age": 1, "image": "Annie.jpg"},
    {"name": "Jewels", "breed": "Domestic Shorthair", "age": 4, "image": "Jewels.jpg"},
    {"name": "Marquette", "breed": "Domestic Shorthair", "age": 2, "image": "Marquette.jpg"},
    {"name": "Mikey", "breed": "Domestic Shorthair", "age": "3", "image": "Mikey.jpg"},
    {"name": "Colby", "breed": "Domestic Shorthair", "age": 1, "image": "Colby.jpg"},
    {"name": "Alice", "breed": "Domestic Shorthair", "age": 5, "image": "Alice.jpg"},
    {"name": "Jerry", "breed": "Domestic Shorthair", "age": 2, "image": "Jerry.jpg"},
    {"name": "Mittens", "breed": "Domestic Shorthair", "age": "3", "image": "Mittens.jpg"},
    {"name": "Henry", "breed": "Domestic Shorthair", "age": 1, "image": "Henry.jpg"},
    {"name": "Coco", "breed": "Domestic Shorthair", "age": 4, "image": "Coco.jpg"},
    {"name": "Winston", "breed": "Domestic Shorthair", "age": 2, "image": "Winston.jpg"},
    {"name": "Wheeler", "breed": "Domestic Shorthair", "age": "3", "image": "Wheeler.jpg"},
    {"name": "Ray", "breed": "Domestic Shorthair", "age": 1, "image": "Ray.jpg"},
    {"name": "Blondie", "breed": "Domestic Shorthair", "age": 5, "image": "Blondie.jpg"},
    {"name": "BB", "breed": "Domestic Shorthair", "age": 2, "image": "BB.jpg"},
    {"name": "Frankie", "breed": "Domestic Shorthair", "age": "3", "image": "Frankie.jpg"},
    {"name": "Angel", "breed": "Domestic Shorthair", "age": 1, "image": "Angel.jpg"},
    {"name": "Winnie", "breed": "Domestic Shorthair", "age": 4, "image": "Winnie.jpg"},
    {"name": "Gary", "breed": "Domestic Shorthair", "age": 2, "image": "Gary.jpg"},
    {"name": "Peppa", "breed": "Domestic Shorthair", "age": "3", "image": "Peppa.jpg"},
    {"name": "Sammy", "breed": "Domestic Shorthair", "age": 1, "image": "Sammy.jpg"},
    {"name": "Sierra", "breed": "Domestic Shorthair", "age": 5, "image": "Sierra.jpg"},
    {"name": "Caesar", "breed": "Domestic Shorthair", "age": 2, "image": "Caesar.jpg"},
    {"name": "Baby", "breed": "Domestic Shorthair", "age": "3", "image": "Baby.jpg"},
    {"name": "Diego", "breed": "Domestic Shorthair", "age": 1, "image": "Diego.jpg"},
    {"name": "Star-Lord", "breed": "Domestic Shorthair", "age": 4, "image": "Star-Lord.jpg"},
    {"name": "Lloyd", "breed": "Domestic Shorthair", "age": 2, "image": "Lloyd.jpg"},
    {"name": "Nacho", "breed": "Domestic Shorthair", "age": "3", "image": "Nacho.jpg"},
    {"name": "Panchito", "breed": "Domestic Shorthair", "age": 1, "image": "Panchito.jpg"},
    {"name": "Rusty", "breed": "Domestic Shorthair", "age": 5, "image": "Rusty.jpg"},
    {"name": "Sky_86", "breed": "Domestic Shorthair", "age": 2, "image": "Sky_86.jpg"},
    {"name": "Jasper", "breed": "Domestic Shorthair", "age": "3", "image": "Jasper.jpg"},
    {"name": "Butter Ball", "breed": "Domestic Shorthair", "age": 1, "image": "Butter Ball.jpg"},
    {"name": "Spot", "breed": "Domestic Shorthair", "age": 4, "image": "Spot.jpg"},
    {"name": "Copper", "breed": "Domestic Shorthair", "age": 2, "image": "Copper.jpg"},
    {"name": "Roger", "breed": "Domestic Shorthair", "age": "3", "image": "Roger.jpg"},
    {"name": "Ollie", "breed": "Domestic Shorthair", "age": 1, "image": "Ollie.jpg"},
    {"name": "Trevor", "breed": "Domestic Shorthair", "age": 5, "image": "Trevor.jpg"},
    {"name": "Boo", "breed": "Domestic Shorthair", "age": 2, "image": "Boo.jpg"},
    {"name": "Shiba", "breed": "Domestic Shorthair", "age": "3", "image": "Shiba.jpg"},
    {"name": "Tiger", "breed": "Domestic Shorthair", "age": 1, "image": "Tiger.jpg"},
    {"name": "Avocado", "breed": "Domestic Shorthair", "age": 4, "image": "Avacado.jpg"},
    {"name": "Buddy", "breed": "Domestic Shorthair", "age": 2, "image": "Buddy.jpg"},
    {"name": "Mallow", "breed": "Domestic Shorthair", "age": "3", "image": "Mallow.jpg"},
    {"name": "BMO", "breed": "Domestic Shorthair", "age": 1, "image": "BMO.jpg"},
    {"name": "Momo", "breed": "Domestic Shorthair", "age": 5, "image": "Momo.jpg"},
    {"name": "Sophie", "breed": "Domestic Shorthair", "age": 2, "image": "Sophie.jpg"},
    {"name": "Piper", "breed": "Domestic Shorthair", "age": "3", "image": "Piper.jpg"},
    {"name": "Chandler", "breed": "Domestic Shorthair", "age": 1, "image": "Chandler.jpg"},
    {"name": "Pickles", "breed": "Domestic Shorthair", "age": 4, "image": "Pickles.jpg"},
    {"name": "Phoebe", "breed": "Domestic Shorthair", "age": 2, "image": "Phoebe.jpg"},
    {"name": "Joel", "breed": "Domestic Shorthair", "age": "3", "image": "Joel.jpg"},
    {"name": "Kyle", "breed": "Domestic Shorthair", "age": 1, "image": "Kyle.jpg"},
    {"name": "Lily_28", "breed": "Domestic Shorthair", "age": 5, "image": "Lily_28.jpg"},
    {"name": "Green Bean", "breed": "Domestic Shorthair", "age": 2, "image": "Green Bean.jpg"},
    {"name": "Bob", "breed": "Domestic Shorthair", "age": "3", "image": "Bob.jpg"},
    {"name": "Jada", "breed": "Domestic Shorthair", "age": 1, "image": "Jada.jpg"},
    {"name": "Cheeto", "breed": "Domestic Shorthair", "age": 4, "image": "Cheeto.jpg"},
    {"name": "Brady", "breed": "Domestic Shorthair", "age": 2, "image": "Brady.jpg"},
    {"name": "Brick", "breed": "Domestic Shorthair", "age": "3", "image": "Brick.jpg"},
    {"name": "Knight", "breed": "Domestic Shorthair", "age": 1, "image": "Knight.jpg"},
    {"name": "Saxy", "breed": "Domestic Shorthair", "age": 5, "image": "Saxy.jpg"},
    {"name": "Sylvester", "breed": "Domestic Shorthair", "age": 2, "image": "Sylvester.jpg"},
    {"name": "Luna", "breed": "Domestic Shorthair", "age": "3", "image": "Luna.jpg"},
    {"name": "Shadow", "breed": "Domestic Shorthair", "age": 1, "image": "Shadow.jpg"},
    {"name": "Pringle", "breed": "Domestic Shorthair", "age": 4, "image": "Pringle.jpg"},
    {"name": "Licius", "breed": "Domestic Shorthair", "age": 2, "image": "Licius.jpg"},
    {"name": "Onyx", "breed": "Domestic Shorthair", "age": "3", "image": "Onyx.jpg"},
    {"name": "Meko", "breed": "Domestic Shorthair", "age": 1, "image": "Meko.jpg"},
    {"name": "Scans", "breed": "Domestic Shorthair", "age": 5, "image": "Scans.jpg"},
    {"name": "Whit", "breed": "Domestic Shorthair", "age": 2, "image": "Whit.jpg"},
    {"name": "Randy", "breed": "Domestic Shorthair", "age": "3", "image": "Randy.jpg"},
    {"name": "Cereza", "breed": "Domestic Shorthair", "age": 1, "image": "Cereza.jpg"},
    {"name": "Manson", "breed": "Domestic Shorthair", "age": 4, "image": "Manson.jpg"},
    {"name": "Blue", "breed": "Domestic Shorthair", "age": 2, "image": "Blue.jpg"},
    {"name": "Fred", "breed": "Domestic Shorthair", "age": "3", "image": "Fred.jpg"},
    {"name": "Tom", "breed": "Domestic Shorthair", "age": 1, "image": "Tom.jpg"},
    {"name": "Houdini", "breed": "Domestic Shorthair", "age": 5, "image": "Houdini.jpg"},
    {"name": "Gunny", "breed": "Domestic Shorthair", "age": 2,"image": "Gunny.jpg"},
    {"name": "Sky", "breed": "Domestic Shorthair", "age": "3", "image": "Sky.jpg"},
    {"name": "Capone", "breed": "Domestic Shorthair", "age": 1, "image": "Capone.jpg"},
    {"name": "Carl", "breed": "Domestic Shorthair", "age": 4, "image": "Carl.jpg"},
    {"name": "Champ", "breed": "Domestic Shorthair", "age": 2, "image": "Champ.jpg"},
    {"name": "Joker", "breed": "Domestic Shorthair", "age": "3", "image": "Joker.jpg"},
    {"name": "Peanut", "breed": "Domestic Shorthair", "age": 1, "image": "Peanut.jpg"},
    {"name": "Parker", "breed": "Domestic Shorthair", "age": 5, "image": "Parker.jpg"},
    {"name": "Raven", "breed": "Domestic Shorthair", "age": 2, "image": "Raven.jpg"},
    {"name": "Artemis", "breed": "Domestic Shorthair", "age": "3", "image": "Artemis.jpg"},
    {"name": "Boxer", "breed": "Domestic Shorthair", "age": 1, "image": "Boxer.jpg"},
    {"name": "Stevie", "breed": "Domestic Shorthair", "age": 4, "image": "Stevie.jpg"},
    {"name": "Bella", "breed": "Domestic Shorthair", "age": 2, "image": "Bella.jpg"},
    {"name": "Mouse", "breed": "Domestic Shorthair", "age": "3", "image": "Mouse.jpg"},
    {"name": "Poppy", "breed": "Domestic Shorthair", "age": "1", "image": "Poppy.jpg"},
    {"name": "Lily", "breed": "Domestic Shorthair", "age": 5, "image": "Lily.jpg"},
    {"name": "Peter", "breed": "Domestic Shorthair", "age": 2, "image": "Peter.jpg"},
    {"name": "Cosmo", "breed": "Domestic Shorthair", "age": "3", "image": "Cosmo.jpg"},
    {"name": "River", "breed": "Domestic Shorthair", "age": "1", "image": "River.jpg"},
    {"name": "Snowball", "breed": "Domestic Shorthair", "age": 4, "image": "Snowball.jpg"},
    {"name": "Chester", "breed": "Domestic Shorthair", "age": 2, "image": "Chester.jpg"},
    {"name": "Midnight Moon", "breed": "Domestic Shorthair", "age": "3", "image": "Midnight Moon.jpg"},
    {"name": "No Name Cats", "breed": "Domestic Shorthair", "age": 1, "image": "No Name Cats.jpg"},
    {"name": "Cindy Lou Who", "breed": "Domestic Shorthair", "age": 5, "image": "Cindy Lou Who.jpg"},
    {"name": "Fey", "breed": "Domestic Shorthair", "age": 2, "image": "Fey.jpg"},
    {"name": "Mr. Whiskers", "breed": "Domestic Shorthair", "age": "3", "image": "Mr. Whiskers.jpg"},
    {"name": "Velma", "breed": "Domestic Shorthair", "age": 1, "image": "Velma.jpg"},
    {"name": "Mason", "breed": "Domestic Shorthair", "age": 4, "image": "Mason.jpg"},
    {"name": "Pixie", "breed": "Domestic Shorthair", "age": 2, "image": "Pixie.jpg"},
    {"name": "Aspen", "breed": "Domestic Shorthair", "age": "3", "image": "Aspen.jpg"},
    {"name": "Hisston Churchill", "breed": "Domestic Shorthair", "age": 1, "image": "Hisston Churchill.jpg"},
    {"name": "Saloman", "breed": "Domestic Shorthair", "age": 5, "image": "Saloman.jpg"},
    {"name": "Maple", "breed": "Domestic Shorthair", "age": 2, "image": "Maple.jpg"},
    {"name": "Bunny", "breed": "Domestic Shorthair", "age": "3", "image": "Bunny.jpg"},
    {"name": "Cherry Pie", "breed": "Domestic Shorthair", "age": 1, "image": "Cherry Pie.jpg"},
    {"name": "Milo", "breed": "Domestic Shorthair", "age": 4, "image": "Milo.jpg"},
    {"name": "Sundae", "breed": "Domestic Shorthair", "age": 2, "image": "Sundae.jpg"},
    {"name": "Leo", "breed": "Domestic Shorthair", "age": "3", "image": "Leo.jpg"},
    {"name": "Butterscotch", "breed": "Domestic Shorthair", "age": 1, "image": "Butterscotch.jpg"},
    {"name": "Peach", "breed": "Domestic Shorthair", "age": 5, "image": "Peach.jpg"},
    {"name": "Charity", "breed": "Domestic Shorthair", "age": 2, "image": "Charity.jpg"},
    {"name": "Ginger", "breed": "Domestic Shorthair", "age": "3", "image": "Ginger.jpg"},
    {"name": "Blaine", "breed": "Domestic Shorthair", "age": 1, "image": "Blaine.jpg"}
]

# --- Pagination ---
ANIMALS_PER_PAGE = 20
NUM_PAGES = (len(animals) + ANIMALS_PER_PAGE - 1) // ANIMALS_PER_PAGE

# Streamlit setup
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


st.title("Welcome to the Local Animal Shelter!")
st.write("Here you can meet some of the animals available for adoption.")

# --- User Authentication ---
if 'username' not in st.session_state:
    st.session_state['username'] = None

def create_account():
    new_username = st.text_input("New Username:")
    new_password = st.text_input("New Password:", type="password")

    # Survey Questions
    st.write("Tell us about your ideal pet:")
    has_children = st.checkbox("Do you have children?")
    has_other_pets = st.checkbox("Do you have other pets?")
    preferred_size = st.selectbox("Preferred size:", ["Small", "Medium", "Large"])
    activity_level = st.select_slider("Preferred activity level:", options=["Low", "Moderate", "High"])
    preferred_pet = st.selectbox("Do you prefer:", ["Cats", "Dogs", "Both"])

    if st.button("Create Account"):
        credentials, leaderboard = load_credentials() # Load leaderboard
        if new_username in credentials:
            st.error("Username already exists.")
        else:
            hashed_password = hash_password(new_password)
            # Hash password
            credentials[new_username] = {
                "password": hashed_password,
                "has_children": has_children,
                "has_other_pets": has_other_pets,
                "preferred_size": preferred_size,
                "activity_level": activity_level,
                "preferred_pet": preferred_pet,
                "share_history": [],
                # Initialize share history
                "lost_pet_history": [], # Initialize lost pet history,
                "found_pet_history": [], # Initialize found pet history - NEW
            }
            save_credentials(credentials, leaderboard)# Save credentials and leaderboard
            st.success("Account created! Please sign in.")
            st.session_state['username'] = new_username
            st.rerun()

def login():
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    if st.button("Sign In"):
        credentials, leaderboard = load_credentials() # Load leaderboard
        if username in credentials and credentials[username]["password"] == hash_password(password):
            st.session_state['username'] = username
            st.session_state['leaderboard'] = leaderboard # Load leaderboard on login
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid credentials.")



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
    st.session_state['leaderboard'] = leaderboard
    save_credentials(credentials, leaderboard) # SAVE - Point persistence
        #Share receipt
    st.success(f"Share Receipt:")
    st.write(f"- Pet: {pet_name}")
    st.write(f"- Date/Time: {timestamp_str}")
    st.write(f"- Points Earned: 5")
    # st.rerun() # Removed rerun to persist form data

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
        st.session_state['leaderboard'] = leaderboard

        save_credentials(credentials, leaderboard) # SAVE - Point persistence

        # Report receipt - still shown
        st.success(f"Lost Pet Reported Successfully!") # Updated success message
        st.write(f"Lost Pet Report Receipt:")
        st.write(f"- Pet Name: {pet_name}")
        st.write(f"- Pet Breed: {pet_breed}")
        st.write(f"- Date/Time: {timestamp_str}")
        st.write(f"- Points Earned: 5")
        # st.rerun() # Removed rerun to persist form data
        return # Added return to stop further execution in button click

# --- Report Found Pet Function (Points Added) - NEW ---
def report_found_pet_points():
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
        st.session_state['leaderboard'] = leaderboard

        save_credentials(credentials, leaderboard) # SAVE - Point persistence


        # Report receipt - still shown
        st.success(f"Found Pet Reported Successfully!") # Updated success message
        st.write(f"Found Pet Report Receipt:")
        st.write(f"- Date/Time: {timestamp_str}")
        st.write(f"- Points Earned: 5")
        # st.rerun() # Removed rerun to persist form data
        return # Added return to stop further execution in button click


# --- Display User History Function ---
def display_user_history():
    username = st.session_state['username']
    if username:
        credentials, leaderboard = load_credentials()
        user_data = credentials.get(username, {})
        share_history = user_data.get('share_history', [])
        lost_pet_history = user_data.get('lost_pet_history', []) # Retrieve lost pet history
        found_pet_history = user_data.get('found_pet_history', []) # Retrieve found pet history - NEW
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
        if found_pet_history: # Display found pet history if available - NEW
            st.sidebar.write("---")
            st.sidebar.subheader("Found Pet Reports:")
            df_found_pet = pd.DataFrame(found_pet_history)
            st.sidebar.dataframe(df_found_pet)
        else:
            st.sidebar.write("No shares or pet reports yet.")
    else:
        st.sidebar.info("Please log in to see your history.")

if st.session_state['username'] is None:
    menu = ["Login", "SignUp"]
    choice = st.selectbox("Menu", menu)
    if choice == "SignUp":
        create_account()
    elif choice == "Login":
        login()
else:
    display_user_history_button = st.sidebar.button("My History & Points")
    if display_user_history_button:
        display_user_history()

    # --- Lost Pet Section ---
    st.header("Lost Pet Reunification Hub")
    st.markdown(

    """
    This hub helps reunite lost pets with their owners.
    **Report a found pet:** Upload a photo. AI matching will attempt to find the owner. Earn points for reporting!
    **Report a lost pet:** Fill the form below with pet info and photo. Earn points for reporting! We'll notify you of matches.
    """
    )
    uploaded_file = st.file_uploader("Upload photo of found pet", type=["jpg", "jpeg", "png"])

    # --- Initialize session state for message visibility ---
    if 'match_message_visible' not in st.session_state:
        st.session_state['match_message_visible'] = False
    if 'report_message_visible' not in st.session_state:
        st.session_state['report_message_visible'] = False
    if 'found_pet_reported' not in st.session_state:
        st.session_state['found_pet_reported'] = False # Flag to track if points were awarded

    if 'potential_match_name' not in st.session_state: # NEW: Store potential match name
        st.session_state['potential_match_name'] = None
    if 'awaiting_confirmation' not in st.session_state: # NEW: Flag for user confirmation
        st.session_state['awaiting_confirmation'] = False

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Found Pet", width=200)
        # Only run find_match if not already awaiting confirmation
        if not st.session_state['awaiting_confirmation']:
            with st.spinner("Searching for a match..."):
                match_name = find_match(uploaded_file)
            st.session_state['potential_match_name'] = match_name # Store the result

        if st.session_state['potential_match_name']:
            st.success(f"Possible match found: **{st.session_state['potential_match_name']}**")
            st.write("Does this look like your pet? Please confirm to earn points!")
            st.session_state['awaiting_confirmation'] = True # Set flag to await confirmation

            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, this is my pet!", key="confirm_match_yes"):
                    # Check if points were already awarded for this specific confirmation
                    if not st.session_state.get('last_confirmed_match', '') == st.session_state['potential_match_name']:
                        report_found_pet_points() # Award points ONLY on explicit confirmation
                        st.session_state['last_confirmed_match'] = st.session_state['potential_match_name'] # Store confirmed match to prevent re-awarding
                        st.success(f"Confirmed! Thank you for helping. Points awarded.")
                    else:
                        st.info("Already confirmed this match and points awarded.")

                    st.session_state['potential_match_name'] = None # Clear match
                    st.session_state['awaiting_confirmation'] = False # Reset confirmation state
                    st.rerun() # Rerun to clear messages
            with col_no:
                if st.button("No, this is not my pet.", key="confirm_match_no"):
                    st.info("Thank you for clarifying. No points awarded for this match.")
                    st.session_state['potential_match_name'] = None # Clear match
                    st.session_state['awaiting_confirmation'] = False # Reset confirmation state
                    st.rerun() # Rerun to clear messages

        elif st.session_state['potential_match_name'] is None and not st.session_state['awaiting_confirmation']:
            # This block runs if no match was found initially
            st.info("No matches found. Check back later.")
            # No points awarded here.
            if st.button("Clear Search", key="clear_no_match"): # Button to clear the no match message
                st.session_state['match_message_visible'] = False # This might be redundant, but safe to keep
                st.session_state['potential_match_name'] = None
                st.session_state['awaiting_confirmation'] = False
                st.rerun()

    # Any code that was AFTER the old 'if uploaded_file is not None' block should remain.
    # This specifically means the 'if st.session_state['found_pet_reported']:' block
    # should be removed, as the new logic handles point awarding and messages directly.
    # The '--- Lost Pet Reporting Form ---' should follow after this new block.

    if st.session_state['found_pet_reported']:
        st.success(f"Found Pet Reported Successfully! +5 Points")
        st.write(f"Found Pet Report Receipt:")
        st.write(f"- Date/Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"- Points Earned: 5")
        if st.button("Continue", key="continue_report"):
            st.session_state['found_pet_reported'] = False
            st.rerun()

    # --- Lost Pet Reporting Form ---
    st.subheader("Report a Lost Pet (Earn Points!)")  # Updated subtitle to indicate points
    lost_pet_name = st.text_input("Pet's Name:")
    lost_pet_breed = st.text_input("Pet's Breed:")
    lost_pet_image = st.file_uploader("Upload lost pet photo", type=["jpg", "jpeg", "png"], key="lost_pet_image")

    if lost_pet_image:
        st.image(lost_pet_image, caption="Lost Pet", width=200)

    if st.button("Report Lost Pet"):
        if lost_pet_image is None: # **Error Handling: Check if image is uploaded**
            st.error("Please upload an image of the lost pet to submit the report.") # Display error message
        else:  # Proceed with report submission if image is uploaded
            data = load_data()
            data["lost_pets"].append({
                "name": lost_pet_name,
                "breed": lost_pet_breed,
                "image": lost_pet_image.name if lost_pet_image else None,
            })
            save_data(data)
            report_lost_pet_points(lost_pet_name, lost_pet_breed)  # Call new function to award points

    # --- Animals Available for Adoption Section ---
    st.header("Animals Available for Adoption")

    num_cols = 4
    cols = st.columns(num_cols)
    pet_index = 0

    # --- Pagination for Animals Section ---
    page_animals = st.sidebar.radio("Page", range(1, NUM_PAGES + 1), key="animals_page_radio")
    start_index_animals = (page_animals - 1) * ANIMALS_PER_PAGE
    end_index_animals = start_index_animals + ANIMALS_PER_PAGE
    paged_animals = animals[start_index_animals:end_index_animals]

    for pet in paged_animals:
        try:
            with cols[pet_index % num_cols]:
                st.write(f"**{pet['name']}**")
                st.write(f"Breed: {pet['breed']}")
                st.write(f"Age: {pet['age']}")

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
                if st.button(f"Share {pet['name']}!", key=f"share_animals_{pet['name']}"):
                    st.session_state['pet_to_share'] = pet['name']  # store the pet name
                    st.session_state['share_prompt'] = not st.session_state.get('share_prompt', False) # Toggle the prompt

                if st.session_state.get('share_prompt', False) and st.session_state.get('pet_to_share') == pet['name']:
                    st.write(f"Share {pet['name']} on:")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        components.html(
                            f"""
                            <a href="https://www.facebook.com/" target="_blank">
                            <button class="social-button">Facebook</button>
                            </a>
                            """,
                            height=50,
                        )

                    with col2:
                        components.html(
                            f"""
                            <a href="https://x.com/" target="_blank">
                            <button class="social-button">X/Twitter</button>
                            </a>
                            """,
                            height=50,
                        )

                    with col3:
                        components.html(
                            f"""
                            <a href="https://www.snapchat.com/" target="_blank">
                            <button class="social-button">Snapchat</button>
                            </a>
                            """,
                            height=50,
                        )

                st.write("---")
        except Exception as e:
            st.error(f"Error displaying {pet['name']}: {e}")
        pet_index += 1

    # --- Adopted Animals Section (Initially Empty) ---
    st.header("Adopted Animals")  # Section header - initially empty
    st.write("This section will display animals that have already been adopted. (Currently empty)")
    # ---  You can add logic here later to display adopted animals if you have a separate list ---
    # --- For now, it's just a header and informational text ---

    # --- Find My Forever Home Challenge ---
    st.header("Find My Forever Home Challenge")

    # Initialize leaderboard
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
    # secrets = load_secrets_toml() #this line is unnecessary

    st.subheader("🌟 Feedback")

    with st.form("feedback_form"):
        feedback_text = st.text_area("Share your feedback!")
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            if "FORMSPREE_ENDPOINT" in st.secrets:  # Access secret directly from st.secrets
                form_endpoint = st.secrets["FORMSPREE_ENDPOINT"]
                try:
                    import requests  # Import requests here

                    response = requests.post(form_endpoint, data={"feedback": feedback_text})
                    if response.status_code == 200:
                        st.success("Thank you for your feedback!")
                    else:
                        st.error(f"Error submitting feedback: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.error("Formspree endpoint not found! Make sure that your secrets.toml is properly updated.")
                
