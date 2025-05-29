
1. Project Title: Peata


2. Elevator Pitch / Short Description:

Every year, millions of beloved pets go missing, causing immense distress to their families and overwhelming animal shelters. Peata is a community-focused web application that leverages AI-powered image matching to rapidly connect found pets with their worried owners and to help shelter animals find loving forever homes. By simply uploading a photo, users can tap into a smart system that significantly increases the chances of a happy reunion or a new beginning.

(Optional: If you have your 30-second video hosted, you could add: "Watch our 30-second overview: [Link to Video]")



3. Problem Statement:

The separation of a pet from its owner is a deeply emotional and stressful event. Traditional methods of finding lost pets – flyers, social media posts, and shelter visits – can be time-consuming, fragmented, and often rely on luck. Animal shelters, while doing incredible work, face challenges in quickly identifying found animals, managing intake, and promoting adoptions effectively. There's a need for a centralized, intelligent platform that can bridge these gaps and utilize technology to expedite the reunification and adoption processes.




4. Solution: How Peata Solves It

Peata offers a user-friendly, centralized platform where the community and shelters can collaborate:

For Lost Pets: Owners can quickly create detailed reports for their missing pets, including photos and descriptions.
For Found Pets: Individuals who find a lost animal can upload its photo. Peata's AI matching engine then compares this photo against the database of reported lost pets and known shelter animals, looking for potential matches based on visual features.
For Adoptions: Peata showcases animals currently in shelters that are available for adoption, providing key details and images to help them find new homes.
Community Engagement: A points and leaderboard system encourages active participation in reporting pets, sharing profiles, and contributing to the platform's success.
The core innovation lies in the application of image recognition technology (Perceptual Hashing and ORB feature matching) to provide a more efficient and potentially faster way to identify animals than relying solely on manual description matching.




5. Key Features:

User Authentication: Secure login and account creation for users.
AI-Powered Image Matching:
Users can upload an image of a found pet.
The system uses a two-stage process (Perceptual Hashing for a quick similarity check and ORB feature matching with homography for more robust verification) to compare the uploaded image against a precomputed database of known shelter animal images.
Presents potential matches to the user for confirmation.
Lost Pet Reporting: Users can submit detailed reports for pets they have lost, including name, breed, and photos.
Found Pet Reporting: Users can report pets they have found, initiating the AI matching process and earning points.
Adoptable Animals Showcase: Displays a browsable list of animals currently available for adoption, with images, names, breeds, and ages. Users can click through to an external adoption website.
Points & Leaderboard System: Users earn points for activities like sharing pet profiles (via simulated social media buttons), reporting lost pets, and confirming found pet matches. A leaderboard displays top contributors.
User Profile & History: Users can view their accumulated points and a log of their sharing and reporting activities.
Responsive UI: Custom-styled interface with attention to user experience, including responsive videos on login/signup and a sticky logo for branding in the main app.
Feedback Mechanism: Integrated Formspree form for users to submit feedback.
GCP Vertex AI Integration (Initialized): Foundational setup for potential future expansion with more advanced cloud AI services (though the current matching is local with OpenCV).


6. Technology Stack:

Backend & Frontend: Python with Streamlit (a rapid web application development framework)
Image Processing & AI Matching:
OpenCV (cv2): For image manipulation, ORB feature detection and description.
imagehash: For Perceptual Hashing (pHash).
Pillow (PIL): For image loading and resizing.
NumPy: For numerical operations, especially with image data.
Data Storage (Prototype Level):
JSON files (credentials.json, database.json): For user credentials, leaderboard, and lost pet reports.
Styling: Custom CSS injected via Streamlit's st.markdown(unsafe_allow_html=True).
Image & Media Handling:
Base64 encoding for embedding local videos (login/signup animation) and the sticky SVG logo directly in HTML.
Local file storage for shelter animal images (img/Cats_Q1_2025/, img/Cats_Q2_2025/, img/other/).
Version Control: Git & GitHub.
Deployment (Cloud): Streamlit Community Cloud.
Secrets Management: Streamlit Secrets (secrets.toml).
Form Handling (Feedback): Formspree.
Cloud AI (Initialization): Google Cloud Platform (GCP) Vertex AI SDK (initialized, for future use).



7. How to Run/Setup (Briefly, for local development understanding):

Clone the Repository: git clone (https://github.com/gastondana627/Peata)
Navigate to Directory: cd Peata
Create & Activate Virtual Environment (Recommended):
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
Bash
pip install -r requirements.txt
(Ensure requirements.txt includes all necessary packages like streamlit, opencv-python, imagehash, Pillow, numpy, pandas, toml, requests, google-cloud-aiplatform, google-auth-oauthlib)
Setup secrets.toml: Create a .streamlit/secrets.toml file with necessary API keys/credentials (e.g., for GCP Vertex AI, Formspree). Example:
Init, TOML
# .streamlit/secrets.toml
FORMSPREE_ENDPOINT = "YOUR_FORMSPREE_ENDPOINT_HERE"

[vertex_ai]
type = "service_account"
project_id = "YOUR_GCP_PROJECT_ID"
# ... other service account details ...
Ensure Image Folders Exist: Create the assets/, img/Cats_Q1_2025/, img/Cats_Q2_2025/, and img/other/ directories and populate them with relevant images if running locally and they are not part of the repo, or ensure your code points to the correct image paths committed in the repo.

Run the Streamlit App:
Bash
streamlit run streamlit_app.py





8. Live Application Link:

https://peata-sao.streamlit.app/ 



9. Accomplishments/What Was Built:

As a solo developer, I successfully designed and built a comprehensive prototype for "Peata," an AI-enhanced platform for pet reunification and adoption. This involved:

Developing a full user authentication system (signup/login).
Implementing a two-stage AI image matching engine from scratch using Perceptual Hashing and ORB feature detection for identifying found pets against a shelter database.
Creating a system for users to report lost and found pets.
Building a browsable showcase for adoptable animals.
Integrating a gamification aspect with a points and leaderboard system to encourage community participation.
Designing and implementing a multi-view user interface with custom styling, responsive elements (like videos on auth pages), and a persistent brand logo.
Setting up data persistence for user accounts, pet reports, and leaderboard data using JSON files.
Successfully deploying the application to Streamlit Community Cloud.
Initiating the integration with Google Cloud Vertex AI for future scalability.



10. Challenges Faced (Optional, but good to include for a hackathon to show resilience):

UI Responsiveness & State Management: Ensuring a smooth user experience, especially with dynamic content like image uploads and match confirmations, required careful management of Streamlit's session state and rerun behavior (e.g., fixing UI elements not resetting, resolving "no-op rerun" warnings).
Image Processing Performance: Initial image matching caused application freezes. This was addressed by implementing image resizing before feature extraction in the AI pipeline and adding user feedback with spinners.
File Organization & Path Management: Reorganizing a large dataset of local images into a new structure necessitated significant updates to data references and path handling logic throughout the codebase.
Error Handling: Initial versions of the app were prone to crashing on unhandled exceptions. Robust try-except blocks were implemented across the main application view to improve stability and provide better error diagnostics.
Solo Development Scope: Managing the full stack – from UI/UX design and frontend development in Streamlit to backend logic, AI model implementation, data handling, and deployment – as a single developer was a significant undertaking requiring prioritization and efficient use of tools.



11. Future Plans:

Database Migration: Transition from JSON files to a more robust and scalable database solution (e.g., SQLite for simplicity, or a cloud database like Firestore/PostgreSQL for larger scale).
Enhanced AI Matching:
Explore more advanced deep learning models for image recognition to improve accuracy and robustness.
Train custom models on specific animal breeds or features.
Incorporate other metadata (breed, color, location) into the matching algorithm.
Direct Shelter Integration: Develop APIs or systems for shelters to directly manage their list of adoptable animals and receive notifications for potential matches of found pets.
Geolocation Features: Allow users to specify location for lost/found pets and filter searches by proximity.
Mobile Responsiveness & PWA: Further improve mobile usability and explore Progressive Web App (PWA) features for better offline access or app-like experience.
Expanded Community Features: Notifications for users when a pet matching their lost report is found, user-to-user messaging (with privacy considerations).
Full Vertex AI Utilization: Leverage Vertex AI for model training, deployment, and potentially other MLOps capabilities.
12. Author: GasMan

Built by: [Gaston Dana - gastondana627]
This project was developed as a solo effort.
