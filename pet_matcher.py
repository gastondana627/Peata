import cv2
import os
import numpy as np
from google.oauth2 import service_account
from google.cloud import aiplatform
import toml

from PIL import Image # For imagehash
import imagehash     # For perceptual hashing
import io            # For imagehash

IMAGE_FOLDER_CATS = "img/cats"
IMAGE_FOLDER_OTHER = "img/other"

# Helper: Safe image loader
def load_images_from_folder(folder_path):
    try:
        return [
            os.path.join(folder_path, img)
            for img in os.listdir(folder_path)
            if img.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    except FileNotFoundError:
        st.warning(f"Directory not found: {folder_path}")
        return []

# Streamlit App - Display Cat Images
def display_cat_gallery():
    st.title("🐱 Cat Gallery")

    if not os.path.exists(IMAGE_FOLDER_CATS):
        st.warning("Cat image folder not found!")
        return

    cat_images = [f for f in os.listdir(IMAGE_FOLDER_CATS) if f.endswith(".jpg")]
    cat_images.sort()

    if not cat_images:
        st.info("No cat images found.")
    else:
        for img_name in cat_images:
            img_path = os.path.join(IMAGE_FOLDER_CATS, img_name)
            try:
                image = Image.open(img_path)
                st.image(image, caption=img_name, use_column_width=True)
            except Exception as e:
                st.warning(f"Could not open {img_name}: {e}")

# Run the gallery if you're not modularizing the script
if __name__ == "__main__":
    display_cat_gallery()

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
    import imagehash
    from PIL import Image
    import os

    print("Stage 1 - Hashing uploaded image")
    uploaded_hash = imagehash.phash(Image.open(uploaded_image))

    print("Stage 2 - Searching folders...")
    best_match = None
    lowest_distance = float("inf")

    for folder_name in os.listdir("img"):
        folder_path = os.path.join("img", folder_name)

        # Skip non-directory entries
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue  # Skip non-image files

            image_path = os.path.join(folder_path, filename)
            try:
                candidate_image = Image.open(image_path)
                candidate_hash = imagehash.phash(candidate_image)

                distance = abs(uploaded_hash - candidate_hash)
                print(f"Comparing to {image_path} — Distance: {distance}")

                if distance < lowest_distance:
                    lowest_distance = distance
                    best_match = image_path

            except Exception as e:
                print(f"Error reading image at {image_path}: {e}")
                continue

    if best_match:
        print(f"Best match: {best_match} (distance: {lowest_distance})")
        return best_match
    else:
        print("No match found.")
        return None

    try:
        # --- Prepare uploaded image for both pHash and ORB ---
        uploaded_image.seek(0) # Reset stream position for reading
        pil_image = Image.open(io.BytesIO(uploaded_image.read()))
        uploaded_image_cv = cv2.imdecode(np.asarray(bytearray(pil_image.tobytes()), dtype=np.uint8), cv2.IMREAD_COLOR) # Convert PIL to OpenCV

        if uploaded_image_cv is None:
            print("Failed to decode uploaded image for OpenCV.")
            return None

        # --- Stage 1: Perceptual Hashing ---
        print("Stage 1: Performing Perceptual Hashing...")
        uploaded_hash = imagehash.phash(pil_image)

        for folder_name in [IMAGE_FOLDER_CATS, IMAGE_FOLDER_OTHER]:
            for filename in os.listdir(folder_name):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_name, filename)
                    try:
                        shelter_pil_image = Image.open(image_path)
                        shelter_hash = imagehash.phash(shelter_pil_image)
                        
                        hamming_distance = uploaded_hash - shelter_hash
                        if hamming_distance < best_match_score:
                            best_match_score = hamming_distance
                            best_match_name = filename # Tentative best match from pHash

                    except Exception as e:
                        print(f"Error processing {filename} for pHash: {e}")
                        continue
        
        if best_match_name and best_match_score <= PHASH_THRESHOLD:
            print(f"Stage 1 Success: Found strong pHash match: {best_match_name} (Distance: {best_match_score})")
            return os.path.splitext(best_match_name)[0]
        else:
            print(f"Stage 1: No strong pHash match found (Best Distance: {best_match_score}). Proceeding to Stage 2.")


        # --- Stage 2: ORB + Homography (Only if pHash doesn't find a strong match) ---
        print("Stage 2: Performing ORB + Homography matching...")

        # Convert uploaded image to grayscale for ORB
        uploaded_image_gray = cv2.cvtColor(uploaded_image_cv, cv2.COLOR_BGR2GRAY)
        keypoints_uploaded, descriptors_uploaded = orb.detectAndCompute(uploaded_image_gray, None)

        if descriptors_uploaded is None or len(keypoints_uploaded) < MIN_INLIERS_FINAL_ORB:
            print(f"Uploaded image has too few keypoints ({len(keypoints_uploaded)}) for ORB.")
            return None # No match if not enough keypoints for ORB

        def perform_homography_match(kp1, des1, kp2, des2, ratio_thresh, min_inliers_count):
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                return 0, float('inf') # Return 0 inliers, infinite error

            try:
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
                elif len(pair) == 1:
                    good_matches.append(pair[0])

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


        # Iterate through all pet images in both folders for ORB matching
        orb_best_match_name = None
        orb_max_inliers = 0
        orb_min_avg_reprojection_error = float('inf')

        for folder_name in [IMAGE_FOLDER_CATS, IMAGE_FOLDER_OTHER]:
            for filename in os.listdir(folder_name):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_name, filename)
                    shelter_image_cv = cv2.imread(image_path)
                    if shelter_image_cv is None:
                        continue

                    shelter_image_gray = cv2.cvtColor(shelter_image_cv, cv2.COLOR_BGR2GRAY)
                    keypoints_shelter, descriptors_shelter = orb.detectAndCompute(shelter_image_gray, None)

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
                        orb_best_match_name = filename
                    elif inlier_count == orb_max_inliers and avg_error < orb_min_avg_reprojection_error:
                        orb_min_avg_reprojection_error = avg_error
                        orb_best_match_name = filename

        # Final decision for Stage 2
        if orb_best_match_name and orb_max_inliers >= MIN_INLIERS_FINAL_ORB \
           and orb_min_avg_reprojection_error <= MAX_AVG_REPROJECTION_ERROR_ORB:
            print(f"Stage 2 Success: Found ORB match: {orb_best_match_name} (Inliers: {orb_max_inliers}, Error: {orb_min_avg_reprojection_error:.2f})")
            return os.path.splitext(orb_best_match_name)[0]
        else:
            print("Stage 2: No robust ORB match found.")
            return None


    except Exception as e:
        print(f"Critical error in find_match function: {e}")
        return None