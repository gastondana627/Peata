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
    using ORB feature matching with adjusted parameters for better recall.
    """
    best_match_name = None
    max_inliers = 0
    # Adjusted parameters for better recall (finding more matches)
    MIN_INLIERS = 18  # Slightly reduced from 20 (original 15). Try 15 if still too strict.
    RATIO_THRESHOLD = 0.70 # Slightly increased from 0.65 (original 0.75). Try 0.75 if needed.

    # Initialize ORB detector and BFMatcher with crossCheck
    orb = cv2.ORB_create(nfeatures=2500) # Keep high number of features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    if uploaded_image is None:
        print("No image uploaded. Returning None.")
        return None

    try:
        # Important: Reset stream position to the beginning if it's already been read (e.g., by st.image)
        uploaded_image.seek(0)
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        uploaded_image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if uploaded_image_cv is None:
            print("Failed to decode uploaded image using cv2.imdecode")
            return None

        # Convert to grayscale for feature detection as ORB is often more stable with it
        uploaded_image_gray = cv2.cvtColor(uploaded_image_cv, cv2.COLOR_BGR2GRAY)
        keypoints_uploaded, descriptors_uploaded = orb.detectAndCompute(uploaded_image_gray, None)

        # Check if enough keypoints are detected in the uploaded image
        # Using a slightly lower threshold for this initial check compared to MIN_INLIERS
        if descriptors_uploaded is None or len(keypoints_uploaded) < 10:
            print(f"Not enough keypoints detected in the uploaded image ({len(keypoints_uploaded)}).")
            return None

        def find_good_matches_homography(kp1, des1, kp2, des2, ratio_thresh):
            # Ensure enough keypoints for homography calculation before matching
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                return 0, None, None

            try:
                # Perform k-NN matching
                matches = bf.knnMatch(des1, des2, k=2)
            except cv2.error as e:
                print(f"Error in knnMatch: {e}. Descriptors might be empty or invalid for matching.")
                return 0, None, None

            good_matches = []
            for pair in matches:
                # Ensure knnMatch returned two nearest neighbors for ratio test, or one for crossCheck=True
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)
                elif len(pair) == 1: # This case is typically for bf.match with crossCheck=True
                    good_matches.append(pair[0])

            # Only proceed with homography if enough good matches are found
            if len(good_matches) >= MIN_INLIERS: # Use main MIN_INLIERS as the bar here
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # RANSAC threshold 5.0
                if M is not None and mask is not None:
                    inlier_matches = np.sum(mask) # Sum of the boolean mask gives count of inliers
                    return inlier_matches, M, mask
            return 0, None, None # Return 0 inliers if not enough good matches or homography fails

        # Iterate through all pet images in both folders
        for folder_name in [IMAGE_FOLDER_CATS, IMAGE_FOLDER_OTHER]:
            for filename in os.listdir(folder_name):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_name, filename)
                    shelter_image = cv2.imread(image_path)
                    if shelter_image is not None:
                        shelter_image_gray = cv2.cvtColor(shelter_image, cv2.COLOR_BGR2GRAY)
                        keypoints_shelter, descriptors_shelter = orb.detectAndCompute(shelter_image_gray, None)
                        # Check if enough keypoints are detected in the shelter image
                        if descriptors_shelter is not None and len(keypoints_shelter) >= 10:
                            inlier_count, _, _ = find_good_matches_homography(
                                keypoints_uploaded, descriptors_uploaded,
                                keypoints_shelter, descriptors_shelter,
                                RATIO_THRESHOLD
                            )
                            # Update best match if current image has more inliers AND meets the MIN_INLIERS threshold
                            if inlier_count > max_inliers and inlier_count >= MIN_INLIERS:
                                max_inliers = inlier_count
                                best_match_name = filename

    except Exception as e:
        print(f"Error in find_match function: {e}")
        return None

    if best_match_name:
        # Return only the name without extension for cleaner display
        return os.path.splitext(best_match_name)[0]
    else:
        return None