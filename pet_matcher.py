# pet_matcher.py
import cv2
import os
import numpy as np
from google.oauth2 import service_account
from google.cloud import aiplatform
import toml

IMAGE_FOLDER_CATS = "img/cats"
IMAGE_FOLDER_OTHER = "img/other"

def load_gcp_credentials():
    """Loads GCP credentials from the service account JSON."""
    try:
        credentials = service_account.Credentials.from_service_account_file('your-service-account.json')
        return credentials
    except FileNotFoundError:
        print("Error: GCP service account JSON file not found.")
        return None

def load_secret_toml():
    try:
        data = toml.load("secrets.toml")
        return data
    except FileNotFoundError:
        print("Error: secrets.toml file not found.")
        return None

def initialize_vertex_ai(credentials, project_id, location):
    """Initializes Vertex AI with the provided credentials."""
    aiplatform.init(project=project_id, location=location, credentials=credentials)

# Example of use.
gcp_creds = load_gcp_credentials()
secret_data = load_secret_toml()

if gcp_creds and secret_data:
    project_id = secret_data['vertex_ai']['project_id']
    location = secret_data['vertex_ai']['location']
    initialize_vertex_ai(gcp_creds, project_id, location)
    print("Vertex AI Initialized")
else:
    print("GCP or toml credentials failed to load")

def find_match(uploaded_image):
    """
    Finds the best match for the uploaded image among shelter pet images
    using ORB feature matching with ratio test.

    Args:
        uploaded_image: Uploaded file object (Streamlit UploadedFile).

    Returns:
        str: Name of the best matching pet, or None if no good match found.
    """
    best_match_name = None
    max_good_matches = 0
    MIN_MATCH_COUNT = 15  # Increased minimum matches
    RATIO_THRESHOLD = 0.75 # Threshold for Lowe's ratio test

    # Initialize ORB detector and BFMatcher
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    if uploaded_image is None:
        print("No image uploaded. Returning None.")
        return None

    try:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        uploaded_image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if uploaded_image_cv is None:
            print("Failed to decode uploaded image.")
            return None

        keypoints_uploaded, descriptors_uploaded = orb.detectAndCompute(uploaded_image_cv, None)
        if descriptors_uploaded is None or len(keypoints_uploaded) < 3:
            print("Not enough keypoints detected in the uploaded image.")
            return None

        def find_good_matches(kp1, des1, kp2, des2, ratio_thresh):
            if des1 is None or des2 is None:
                return []
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
            return len(good_matches)

        # Iterate through cat images
        for filename in os.listdir(IMAGE_FOLDER_CATS):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(IMAGE_FOLDER_CATS, filename)
                shelter_image = cv2.imread(image_path)
                if shelter_image is not None:
                    keypoints_shelter, descriptors_shelter = orb.detectAndCompute(shelter_image, None)
                    if descriptors_shelter is not None and len(keypoints_shelter) >= 3:
                        good_matches = find_good_matches(keypoints_uploaded, descriptors_uploaded, keypoints_shelter, descriptors_shelter, RATIO_THRESHOLD)
                        if good_matches > max_good_matches and good_matches > MIN_MATCH_COUNT:
                            max_good_matches = good_matches
                            best_match_name = filename

        # Iterate through other images
        for filename in os.listdir(IMAGE_FOLDER_OTHER):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(IMAGE_FOLDER_OTHER, filename)
                shelter_image = cv2.imread(image_path)
                if shelter_image is not None:
                    keypoints_shelter, descriptors_shelter = orb.detectAndCompute(shelter_image, None)
                    if descriptors_shelter is not None and len(keypoints_shelter) >= 3:
                        good_matches = find_good_matches(keypoints_uploaded, descriptors_uploaded, keypoints_shelter, descriptors_shelter, RATIO_THRESHOLD)
                        if good_matches > max_good_matches and good_matches > MIN_MATCH_COUNT:
                            max_good_matches = good_matches
                            best_match_name = filename

    except Exception as e:
        print(f"Error in find_match function: {e}")
        return None

    if best_match_name:
        return best_match_name
    else:
        return None

