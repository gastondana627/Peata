# Peata: The AI-Powered Rescue & Reunite Hub

![Peata Logo](assets/Peata_Logo.svg)

Peata is a community-focused web application designed to tackle the heartbreaking issue of lost pets. By leveraging a dual AI-powered system, Peata provides a centralized, intelligent platform for reuniting lost pets with their owners and helping shelter animals find their forever homes.

## 🐾 Project Overview

Every year, millions of beloved pets go missing, causing immense distress to their families and overwhelming animal shelters. Traditional methods of finding lost pets—flyers, social media posts, and shelter visits—can be time-consuming and fragmented. Peata bridges this gap with technology.

### How Peata Solves It

-   **AI-Powered Image Matching**: Users who find a pet can upload its photo. Peata uses a two-stage image recognition process to compare the photo against a database of lost and shelter pets, significantly increasing the chance of a match.
-   **Dual AI Chatbot**: Peata includes a sophisticated chatbot that can operate in two modes:
    -   **Online Mode**: Powered by Google's powerful **Vertex AI (Gemini 2.0 Flash)** for fast and comprehensive responses.
    -   **Offline Mode**: Powered by **Gemma 2B-IT**, a state-of-the-art model from Google, running locally. This ensures the chatbot is always available, even if cloud services are interrupted.
-   **Community Engagement**: A points and leaderboard system gamifies the process of reporting pets and sharing profiles, encouraging active community participation.
-   **Centralized Hub**: Peata provides a single place for reporting lost pets, reporting found pets, and browsing adoptable animals from the shelter.

---

## 🛠️ Technology Stack

Peata is built with a modern, robust stack designed for performance and scalability.

-   **Backend & Frontend**: Python with [Streamlit](https://streamlit.io/)
-   **AI Chatbot**:
    -   **Online**: Google Cloud Vertex AI (Gemini 2.0 Flash)
    -   **Offline**: Hugging Face Transformers with `google/gemma-2b-it`
-   **AI Image Matching**:
    -   **Stage 1 (Fast Scan)**: Perceptual Hashing (`imagehash`)
    -   **Stage 2 (Deep Scan)**: ORB Feature Detection & Homography (`opencv-python`)
-   **Image Processing**: Pillow (PIL), NumPy
-   **Data Storage**: Flat-file JSON for user credentials, reports, and leaderboards.
-   **Deployment**: Streamlit Community Cloud
-   **Styling**: Custom CSS injected via Streamlit.

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

-   Python 3.8+
-   A virtual environment manager (e.g., `venv`)

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/Peata.git
    cd Peata
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    The required Python packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

### Configuration: Secrets Management

Peata uses Streamlit Secrets for managing API keys and credentials. You must create a `secrets.toml` file in a `.streamlit` directory.

1.  Create the directory:
    ```bash
    mkdir .streamlit
    ```

2.  Create the secrets file:
    ```bash
    touch .streamlit/secrets.toml
    ```

3.  Add the following content to `.streamlit/secrets.toml`, replacing the placeholder values with your actual credentials.

    ```toml
    # .streamlit/secrets.toml

    # --- Hugging Face Token (for Gemma offline model) ---
    # Required to download the gated Gemma 2B-IT model.
    # Get your token from https://huggingface.co/settings/tokens
    HF_TOKEN = "hf_YOUR_HUGGING_FACE_TOKEN"

    # --- Google Cloud Vertex AI Credentials ---
    # Credentials for the online Gemini 2.0 Flash model.
    # This is the content of your GCP service account JSON key file.
    [vertex_ai]
    type = "service_account"
    project_id = "your-gcp-project-id"
    private_key_id = "your-private-key-id"
    private_key = "-----BEGIN PRIVATE KEY-----\\nYOUR_PRIVATE_KEY\\n-----END PRIVATE KEY-----\\n"
    client_email = "your-service-account-email@your-project-id.iam.gserviceaccount.com"
    client_id = "your-client-id"
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account-email.iam.gserviceaccount.com"

    # --- Formspree Endpoint (for feedback form) ---
    FORMSPREE_ENDPOINT = "https://formspree.io/f/YOUR_ENDPOINT_ID"
    ```

### Running the Application

Once the dependencies are installed and your secrets are configured, you can run the app with a single command:

```bash
streamlit run streamlit_app.py
```

Your web browser should open a new tab with the Peata application running locally.

---

## ✨ Core Features Explained

### 1. Dual AI Chatbot

The chatbot is the heart of user interaction. It can be switched between two modes from the sidebar.

-   **Online Mode (Vertex AI)**: Default mode. Uses a powerful cloud-based model for the best performance.
-   **Offline Mode (Gemma)**: A fallback that runs entirely on your local machine. The first time you switch to this mode, it will download the model (approx. 5 GB), which may take a few minutes. Subsequent uses will be much faster.

### 2. AI Image Matching

When a user uploads a photo of a found pet, a two-stage process begins:

1.  **pHash Scan**: The system first generates a "perceptual hash" of the uploaded image and quickly compares it to the hashes of all pets in the database. If a very close match is found, the process stops and suggests the match. This is extremely fast.
2.  **ORB Deep Scan**: If the pHash scan doesn't yield a confident match, the system proceeds to a more detailed analysis using **Oriented FAST and Rotated BRIEF (ORB)** feature detection. It identifies hundreds of key points (like the corner of an eye or the tip of an ear) and compares them to the key points of images in the database, even if the pet is in a different pose or lighting. This is computationally more intensive but far more accurate for difficult matches.

### 3. User Authentication and Gamification

-   **Secure Accounts**: Users can create accounts to report lost pets, track their history, and earn points.
-   **Leaderboard**: A leaderboard tracks users who contribute the most by reporting pets and sharing profiles, fostering a helpful community.

---

## 📂 Project Structure

```
.
├── app_core/
│   └── ai_service.py       # Handles all chatbot logic (Vertex AI & Gemma)
├── assets/                 # Logos, videos, and other static assets
├── img/
│   ├── Cats_Q2_2025/       # Image database for shelter cats
│   └── other/              # Image database for other animals
├── .streamlit/
│   └── secrets.toml        # (You create this) API keys and credentials
├── streamlit_app.py        # Main application file, handles UI and flow
├── pet_matcher.py          # Core logic for the AI image matching
├── requirements.txt        # Python package dependencies
└── README.md               # This file
```

---

## ✍️ Author

Built by: **Gaston Dana**

This project was developed as a solo effort.
