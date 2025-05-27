# pet_matcher.py

import cv2
import os
import numpy as np
from PIL import Image
import imagehash
import io
# from google.oauth2 import service_account # Not needed here, handled by streamlit_app
# from google.cloud import aiplatform     # Not needed here, handled by streamlit_app
# import toml                            # Not needed here, handled by streamlit_app

# Define constants for image folders (good practice)
IMAGE_FOLDER_CATS = "img/Cats_Q2_2025"
IMAGE_FOLDER_OTHER = "img/other"
# You should ensure your img/cats and img/other directories exist and contain images.

# --- ORB Parameters (Define these here, not in the find_match function itself) ---
# These values are examples; you might need to tune them based on your images
ORB_FEATURES = 500 # Max number of features to detect
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8
ORB_EDGE_THRESHOLD = 31
ORB_FIRST_LEVEL = 0
ORB_WTA_K = 2
ORB_SCORE_TYPE = cv2.ORB_FAST_SCORE
ORB_PATCH_SIZE = 31
ORB_FAST_THRESHOLD = 20

PHASH_THRESHOLD = 8 # Maximum Hamming distance for pHash match (tune this)
RATIO_THRESHOLD_ORB = 0.75 # Ratio test for good matches (tune this)
MIN_INLIERS_FINAL_ORB = 10 # Minimum inliers required for a homography match (tune this)
MAX_AVG_REPROJECTION_ERROR_ORB = 5.0 # Max average reprojection error (tune this)


# Initialize ORB detector and Brute-Force Matcher once
# This is crucial for efficiency and to avoid re-initializing on every call
orb = cv2.ORB_create(
    nfeatures=ORB_FEATURES,
    scaleFactor=ORB_SCALE_FACTOR,
    nlevels=ORB_N_LEVELS,
    edgeThreshold=ORB_EDGE_THRESHOLD,
    firstLevel=ORB_FIRST_LEVEL,
    WTA_K=ORB_WTA_K,
    scoreType=ORB_SCORE_TYPE,
    patchSize=ORB_PATCH_SIZE,
    fastThreshold=ORB_FAST_THRESHOLD
)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


# --- Image Preprocessing/Feature Extraction Helper ---
# This function is where you would process images from your database
# and extract their features (hashes, ORB descriptors, etc.)
# It should be called once at app startup or when the DB changes.
def precompute_shelter_image_features(image_folders):
    """
    Precomputes pHashes and ORB features for all images in the specified folders.
    Returns a dictionary mapping image_path to { 'phash': hash_value, 'kp': keypoints, 'des': descriptors }
    """
    shelter_features = {}
    print("Precomputing features for shelter animals...")
    for folder_path in image_folders:
        if not os.path.isdir(folder_path):
            print(f"Warning: Image folder not found: {folder_path}")
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(folder_path, filename)
            try:
                # For pHash
                pil_image = Image.open(image_path)
                current_phash = imagehash.phash(pil_image)

                # For ORB
                cv_image = cv2.imread(image_path)
                if cv_image is None:
                    print(f"Warning: Could not read image for OpenCV: {image_path}")
                    continue
                cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                kp, des = orb.detectAndCompute(cv_image_gray, None)

                shelter_features[image_path] = {
                    'phash': current_phash,
                    'kp': kp,
                    'des': des,
                    'name': os.path.splitext(filename)[0] # Store just the name for return
                }
                # print(f"  Processed: {filename}")
            except Exception as e:
                print(f"Error processing {image_path} for precomputation: {e}")
                continue
    print("Finished precomputing shelter animal features.")
    return shelter_features


# --- Main Matching Function ---
def find_match(uploaded_file_bytesio, shelter_features_db):
    """
    Attempts to find a match for the uploaded image against known shelter animals
    using a two-stage process: Perceptual Hashing (pHash) and ORB feature matching.
    """
    best_match_name = None
    best_match_score = float('inf') # For pHash: lower is better
    orb_best_match_name = None
    orb_max_inliers = 0
    orb_min_avg_reprojection_error = float('inf')

    try:
        # --- Prepare uploaded image for both pHash and ORB ---
        uploaded_file_bytesio.seek(0) # IMPORTANT: Reset stream position
        pil_image = Image.open(uploaded_file_bytesio) # PIL can open BytesIO directly
        
        # Convert PIL Image to OpenCV format
        uploaded_image_cv = np.array(pil_image.convert('RGB')) # Convert to RGB if needed
        uploaded_image_cv = uploaded_image_cv[:, :, ::-1].copy() # Convert RGB to BGR for OpenCV
        
        if uploaded_image_cv is None:
            print("Failed to convert uploaded image to OpenCV format.")
            return None

        # --- Stage 1: Perceptual Hashing (pHash) ---
        print("Stage 1: Performing Perceptual Hashing...")
        uploaded_hash = imagehash.phash(pil_image)

        for image_path, features in shelter_features_db.items():
            candidate_hash = features['phash']
            hamming_distance = uploaded_hash - candidate_hash
            # print(f"  pHash: {features['name']} distance = {hamming_distance}")

            if hamming_distance < best_match_score:
                best_match_score = hamming_distance
                best_match_name = features['name'] # Tentative best match from pHash

        if best_match_name and best_match_score <= PHASH_THRESHOLD:
            print(f"Stage 1 Success: Found strong pHash match: {best_match_name} (Distance: {best_match_score})")
            return best_match_name
        else:
            print(f"Stage 1: No strong pHash match found (Best Distance: {best_match_score}). Proceeding to Stage 2.")


        # --- Stage 2: ORB + Homography (Only if pHash doesn't find a strong match) ---
        print("Stage 2: Performing ORB + Homography matching...")

        uploaded_image_gray = cv2.cvtColor(uploaded_image_cv, cv2.COLOR_BGR2GRAY)
        keypoints_uploaded, descriptors_uploaded = orb.detectAndCompute(uploaded_image_gray, None)

        if descriptors_uploaded is None or len(keypoints_uploaded) < MIN_INLIERS_FINAL_ORB:
            print(f"Uploaded image has too few keypoints ({len(keypoints_uploaded) if keypoints_uploaded else 0}) for ORB.")
            return None # No match if not enough keypoints for ORB

        def perform_homography_match(kp1, des1, kp2, des2, ratio_thresh, min_inliers_count):
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                return 0, float('inf') # Return 0 inliers, infinite error

            try:
                # Use knnMatch for ORB
                matches = bf.knnMatch(des1, des2, k=2)
            except cv2.error as e:
                # print(f"Error in knnMatch (ORB): {e}") # Suppress verbose error for expected cases
                return 0, float('inf')

            good_matches = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)
                # else: # If k=1 was used for some reason, or only one match found
                #     good_matches.append(pair[0])

            if len(good_matches) < min_inliers_count:
                return 0, float('inf')

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None or mask is None:
                return 0, float('inf')

            inlier_matches = np.sum(mask)

            if inlier_matches > 0:
                inlier_src_pts = src_pts[mask.ravel() == 1]
                inlier_dst_pts = dst_pts[mask.ravel() == 1]
                reprojected_pts = cv2.perspectiveTransform(inlier_src_pts, M)
                errors = np.linalg.norm(inlier_dst_pts - reprojected_pts, axis=2)
                avg_error = np.mean(errors)
            else:
                avg_error = float('inf')

            return inlier_matches, avg_error


        # Iterate through all precomputed shelter pet features for ORB matching
        for image_path, features in shelter_features_db.items():
            keypoints_shelter = features['kp']
            descriptors_shelter = features['des']
            shelter_name = features['name']

            if descriptors_shelter is None or len(keypoints_shelter) < MIN_INLIERS_FINAL_ORB:
                continue # Skip if shelter image has too few keypoints for ORB

            inlier_count, avg_error = perform_homography_match(
                keypoints_uploaded, descriptors_uploaded,
                keypoints_shelter, descriptors_shelter,
                RATIO_THRESHOLD_ORB, MIN_INLIERS_FINAL_ORB
            )

            # Update best ORB match based on inliers and reprojection error
            # Prioritize more inliers, then lower error
            if inlier_count > orb_max_inliers:
                orb_max_inliers = inlier_count
                orb_min_avg_reprojection_error = avg_error
                orb_best_match_name = shelter_name
            elif inlier_count == orb_max_inliers and avg_error < orb_min_avg_reprojection_error:
                orb_min_avg_reprojection_error = avg_error
                orb_best_match_name = shelter_name

        # Final decision for Stage 2
        if orb_best_match_name and orb_max_inliers >= MIN_INLIERS_FINAL_ORB \
           and orb_min_avg_reprojection_error <= MAX_AVG_REPROJECTION_ERROR_ORB:
            print(f"Stage 2 Success: Found ORB match: {orb_best_match_name} (Inliers: {orb_max_inliers}, Error: {orb_min_avg_reprojection_error:.2f})")
            return orb_best_match_name
        else:
            print("Stage 2: No robust ORB match found.")
            return None


    except Exception as e:
        print(f"Critical error in find_match function: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None