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
    """Loads the secrets.toml file."""
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
    using ORB feature matching with ratio test and homography estimation.
    """
    best_match_name = None
    max_inliers = 0
    MIN_INLIERS = 15  # Increased minimum number of inlier matches required
    RATIO_THRESHOLD = 0.7 # Strict ratio threshold

    # Initialize ORB detector and BFMatcher with crossCheck
    orb = cv2.ORB_create(nfeatures=2000) # Increase number of features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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

        def find_good_matches_homography(kp1, des1, kp2, des2, ratio_thresh):
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10: # Increased min keypoints
                return 0, None
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

            if len(good_matches) > MIN_INLIERS:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None and mask is not None:
                    inlier_matches = np.sum(mask)
                    return inlier_matches, M
            return 0, None

        # Iterate through all pet images
        for folder_name in [IMAGE_FOLDER_CATS, IMAGE_FOLDER_OTHER]:
            for filename in os.listdir(folder_name):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_name, filename)
                    shelter_image = cv2.imread(image_path)
                    if shelter_image is not None:
                        keypoints_shelter, descriptors_shelter = orb.detectAndCompute(shelter_image, None)
                        if descriptors_shelter is not None and len(keypoints_shelter) >= 10: # Increased min keypoints
                            inlier_count, _ = find_good_matches_homography(
                                keypoints_uploaded, descriptors_uploaded, keypoints_shelter, descriptors_shelter, RATIO_THRESHOLD
                            )
                            if inlier_count > max_inliers and inlier_count > MIN_INLIERS:
                                max_inliers = inlier_count
                                best_match_name = filename

    except Exception as e:
        print(f"Error in find_match function: {e}")
        return None

    if best_match_name:
        return best_match_name
    else:
        return None