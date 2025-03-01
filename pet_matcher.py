import cv2
import os
import numpy as np # ADDED: Import numpy

IMAGE_FOLDER_CATS = "img/cats"  # Define image folder paths here to match app.py
IMAGE_FOLDER_OTHER = "img/other"

def find_match(uploaded_image):
    """
    Finds the best match for the uploaded image among shelter pet images
    using ORB feature matching with OpenCV.

    Args:
        uploaded_image: Uploaded file object (Streamlit UploadedFile).

    Returns:
        str: Name of the best matching pet, or None if no good match found.
    """

    best_match_name = None
    max_matches = 0

    # Initialize ORB detector and Brute-Force matcher
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Convert uploaded image to OpenCV format and grayscale
    file_bytes = uploaded_image.getvalue()
    np_array = np.frombuffer(file_bytes, dtype=np.uint8) # MODIFIED LINE: Use numpy.frombuffer and dtype=np.uint8

    found_pet_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if found_pet_image is None:
        print(f"Error: Could not decode uploaded image.") # Debugging
        return None # Handle decode failure
    found_pet_gray = cv2.cvtColor(found_pet_image, cv2.COLOR_BGR2GRAY)
    found_pet_kp, found_pet_des = orb.detectAndCompute(found_pet_gray, None)

    if found_pet_kp is None or found_pet_des is None:
        print(f"Warning: No ORB features detected in uploaded image.") # Debugging
        return None # No features, cannot match

    # Load shelter pet images and compare
    image_folders = [IMAGE_FOLDER_CATS, IMAGE_FOLDER_OTHER]
    for folder in image_folders:
        for filename in os.listdir(folder):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): # Basic image file check
                continue

            pet_name = os.path.splitext(filename)[0] # Extract name from filename (without extension)
            pet_image_path = os.path.join(folder, filename)
            shelter_pet_image = cv2.imread(pet_image_path)
            if shelter_pet_image is None:
                print(f"Error: Could not read shelter pet image: {pet_image_path}") # Debugging
                continue # Skip to next image if load fails

            shelter_pet_gray = cv2.cvtColor(shelter_pet_image, cv2.COLOR_BGR2GRAY)
            shelter_pet_kp, shelter_pet_des = orb.detectAndCompute(shelter_pet_gray, None)

            if shelter_pet_kp is None or shelter_pet_des is None:
                print(f"Warning: No ORB features detected in shelter pet image: {pet_image_path}") # Debugging
                continue # No features, cannot match

            if found_pet_des is not None and shelter_pet_des is not None: # Ensure descriptors exist
                matches = bf.match(found_pet_des, shelter_pet_des)
                good_matches = [m for m in matches if m.distance < 50]  # Adjust distance threshold as needed

                if len(good_matches) > max_matches:
                    max_matches = len(good_matches)
                    best_match_name = pet_name
                    print(f"Potential match found: {best_match_name} with {max_matches} matches.") # Debugging

    if best_match_name:
        print(f"Best match found: {best_match_name} with {max_matches} matches.") # Debugging
        return best_match_name
    else:
        print("No good match found.") # Debugging
        return None

if __name__ == '__main__':
    # Example usage for testing - Place test images in 'test_images' folder
    test_image_folder = "test_images" # Create a folder named 'test_images' and put test images inside
    if not os.path.exists(test_image_folder):
        os.makedirs(test_image_folder) # Create if it doesn't exist

    # Example: Test with an image from test_images folder
    test_image_path = os.path.join(test_image_folder, "cat_test_image.jpg") # Place a test image here
    if os.path.exists(test_image_path):
        with open(test_image_path, "rb") as f: # Open in binary read mode
            test_uploaded_file = f.read() # Read bytes
        # Need to simulate Streamlit UploadedFile for testing outside of Streamlit
        class MockUploadedFile:
            def getvalue(self):
                return test_uploaded_file # Return bytes

        mock_uploaded_file = MockUploadedFile()
        match_name = find_match(mock_uploaded_file)
        if match_name:
            print(f"Test Match found: {match_name}")
        else:
            print("Test No match found.")
    else:
        print(f"Please place a test image 'cat_test_image.jpg' in '{test_image_folder}' folder for testing.")

