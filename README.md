Vertex AI-Powered Pet Adoption & Reunification App

Overview

This application is designed to help a local animal shelter find homes for pets and reunite lost pets using Google Vertex AI for image recognition. It includes:

Lost Pet Reunification Hub

Find My Forever Home Challenge (adoption system)

Shelter Management Dashboard

The app is built using Streamlit, with Google Cloud Vertex AI handling image matching and JSON storage for lightweight data management.

Features

AI-powered pet matching using Vertex AI.

Lost pet identification via image uploads.

Gamified adoption system to encourage pet adoptions.

Admin dashboard for shelters to manage pet listings.

Installation

Prerequisites

Python 3.8+

Google Cloud SDK installed and configured

Streamlit installed (pip install streamlit)

Vertex AI API enabled in Google Cloud

Service Account JSON key with proper permissions

Clone the Repository

git clone https://github.com/your-repo-name.git
cd your-repo-name

Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows

Install Dependencies

pip install -r requirements.txt

Configuration

1. Set Up Google Cloud Authentication

You need to store your Google Cloud credentials securely.

Local Setup (secrets.toml)

Create a ~/.streamlit/secrets.toml file and add:

[vertex_ai]
private_key = """-----BEGIN PRIVATE KEY-----
<your-private-key-content-here>
-----END PRIVATE KEY-----"""
client_email = "your_service_account_email_here"

Streamlit Cloud Secrets Setup

Go to Streamlit Cloud.

Open your app's Secrets settings.

Add the same credentials in the secrets.toml format.

Running the Application

Locally

streamlit run app.py

Deploying to Streamlit Cloud

Push your changes to GitHub.

Connect your repo to Streamlit Cloud.

Deploy the app and configure secrets.

Deployment Considerations

Git Ignore Rules

Ensure the following files are ignored to prevent accidental exposure of credentials:

# Ignore Google Cloud SDK and CLI installation files
google-cloud-sdk-latest-darwin-x86_64.tar.gz
google-cloud-cli-darwin-x86_64.tar.gz
google-cloud-sdk/

Security Best Practices

Never expose private_key or service_account.json in your repository.

Use environment variables or Streamlit secrets to manage credentials.

Regularly review Google Cloud IAM permissions to ensure minimal access rights.

Future Enhancements

Upgrade to FastAPI & Next.js for better scalability.

Real-time pet status updates using WebSockets.

Improved AI model for pet matching with user feedback.

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
