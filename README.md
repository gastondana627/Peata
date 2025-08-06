# Peata: The AI-Powered Rescue & Reunite Hub
### *A Submission for the Gemma 3n Hackathon*

**Public Project Link**: [https://peata-sao.streamlit.app/](https://peata-sao.streamlit.app/)
**Video Demo Link**: [To be added]

---

## 1. The Problem: A Community Disconnected

Every year, millions of pets go missing, leaving families distraught and animal shelters overwhelmed. The traditional methods for recovery—paper flyers, fragmented social media posts, and desperate shelter visits—are inefficient and rely heavily on luck. This disconnection costs precious time, prolonging the suffering of both pets and their owners. Peata was built to bridge this gap with an intelligent, centralized, and community-driven solution.

## 2. The Solution: An AI-Powered Hub for Hope

Peata is a web application that serves as a central hub for pet reunification and adoption. It leverages a powerful dual-AI system to intelligently connect lost pets with their owners and help shelter animals find new homes.

-   **AI Image Matching**: Found a pet? Upload its photo, and Peata's two-stage AI matching engine gets to work, comparing it against a database of lost and shelter animals to find a potential match.
-   **Dual AI Chatbot**: Get answers and assistance from Peata's versatile chatbot, which can operate in two modes:
    -   **Online Mode**: Powered by **Vertex AI (Gemini 2.0 Flash)** for fast, comprehensive responses.
    -   **Offline Mode**: Powered by **Gemma 2B-IT**, ensuring the app's core chat functionality is always available, even without an internet connection.
-   **Community-Driven Platform**: Users can create accounts, report lost pets, view found animals, and earn points for participating, creating a gamified and engaging experience that encourages community involvement.

---
## 3. Key Features

-   **Secure User Authentication**: Users can sign up and log in to a persistent account to track their activity and points.
-   **Lost & Found Pet Reporting**: Simple forms allow users to report a pet they have lost or one they have found, including uploading photos.
-   **AI-Powered Image Matching**: When a pet is reported as found, the system automatically runs a two-stage image analysis to find potential matches from the shelter and lost pet database.
-   **Dual-Mode AI Chatbot**: A helpful chatbot, powered by Vertex AI online and Gemma 2B-IT offline, is available to answer user questions.
-   **Gamified Community Hub**:
    -   **Points System**: Users earn points for reporting pets and sharing profiles on social media.
    -   **Leaderboard**: A public leaderboard showcases the most active and helpful community members.
-   **Adoptable Animals Showcase**: A browsable, paginated gallery of all animals currently available for adoption at the shelter.
-   **User Profile & History**: A dedicated page where users can see their total points and a log of their reporting and sharing activities.

---

## 4. App Architecture

Peata is designed with a modular and scalable architecture, primarily built around a Streamlit frontend that communicates with distinct AI and data management services.

```
+--------------------------------+
|      Streamlit Frontend        |
|      (streamlit_app.py)        |
+--------------------------------+
|  |                           |
|  v                           v
+-----------------+     +----------------------+
|  User & Data    |     |   AI Services        |
|  Management     |     | (app_core/ai_service.py|
+-----------------+     +----------------------+
| - credentials.json  |     | - Vertex AI (Online) |
| - database.json     |     | - Gemma 2B-IT (Offline)|
+-----------------+     +----------------------+
|                                |
|                                v
|                      +----------------------+
|                      |  Image Matching      |
|                      |   (pet_matcher.py)   |
|                      +----------------------+
|                      | - pHash (Fast Scan)  |
|                      | - ORB (Deep Scan)    |
|                      +----------------------+
```

1.  **Streamlit Frontend (`streamlit_app.py`):** This is the single source of truth for the user interface. It handles user authentication, page views, and orchestrates calls to the other services.
2.  **User & Data Management:** Simple JSON files (`credentials.json`, `database.json`) act as a lightweight database for storing user accounts, lost pet reports, and the community leaderboard. This was chosen for rapid prototyping.
3.  **AI Services (`app_core/ai_service.py`):** This module contains all the logic for the chatbot. It is responsible for initializing the AI models and handling the fallback logic between the online and offline modes.
4.  **Image Matching (`pet_matcher.py`):** This specialized module handles the heavy lifting of image recognition. It pre-computes features for all shelter animals at startup and runs the two-stage matching algorithm when a user uploads a photo of a found pet.

---

## 4. Specific Use of Gemma 2B-IT

The integration of Gemma is a cornerstone of Peata's resilience, providing a robust fallback to ensure the chatbot is always available to the user.

-   **Model Used**: We are using `google/gemma-2b-it`, a powerful yet efficient instruction-tuned model that is well-suited for a helpful assistant role. Its balance of performance and size makes it feasible to run on a local CPU.

-   **Loading and Invocation**:
    1.  **Loading**: The model is loaded from Hugging Face using the `transformers` library. To optimize the user experience, we implemented a **pre-loading strategy**. The `load_gemma_model` function is decorated with Streamlit's `@st.cache_resource`, which downloads the model and caches it on disk. This function is called immediately after a user logs in, so the multi-gigabyte download and initial load happen in the background while the user explores the app.
    2.  **Invocation**: When the user switches to "Offline Mode," the `get_chatbot_response` function routes the prompt to the already-loaded Gemma model. The model generates a response using a standard `model.generate()` call.

-   **Fallback Logic**: The application is designed to be "online-first."
    1.  The app first checks if it is in "Online Mode."
    2.  If yes, it attempts to connect to Vertex AI. If this connection succeeds, it uses the Gemini model.
    3.  If the app is in "Offline Mode," OR if the online connection fails for any reason (e.g., no internet, API error), the application seamlessly **falls back** to using the local Gemma model. This ensures maximum uptime for the core chatbot feature.

---

## 5. Tech Stack Reasoning

Every technology in Peata was chosen for a specific purpose, balancing rapid development with powerful features.

-   **Python & Streamlit**: Chosen for its speed of development. Streamlit allows for the creation of beautiful, interactive data and AI applications with pure Python, which was perfect for a hackathon timeline.
-   **Gemma 2B-IT**: The ideal choice for the offline model. It is powerful enough to provide genuinely helpful responses while being small enough to feasibly run on a user's local machine or a standard cloud container CPU, which is critical for a public-facing application.
-   **Vertex AI (Gemini 2.0 Flash)**: Used as the primary online model to provide the best possible performance and response quality without taxing the user's or the server's local resources.
-   **OpenCV & ImageHash**: This two-stage approach to image matching is highly efficient. `imagehash` provides a near-instantaneous way to find potential matches, while OpenCV's ORB feature detection provides a much more robust (but slower) deep scan, giving us the best of both worlds.
-   **JSON Database**: For a prototype, a full-fledged database would be overkill. Using simple JSON files allowed for rapid implementation of user accounts and data persistence without the overhead of database management.

---

## 6. Challenges Overcome

Development was not without its challenges. Here are some of the key hurdles and how they were solved:

-   **Challenge: Slow Offline Model Loading.** The Gemma 2B-IT model is over 5 GB, and the initial download and loading process took several minutes, creating a terrible user experience.
    -   **Solution:** We implemented a **pre-loading and caching** strategy. By calling the cached `load_gemma_model` function immediately after user login, the long wait happens in the background. By the time the user wants to use the offline chatbot, it's already in memory and ready to go.

-   **Challenge: Offline Mode Crashing Without Internet.** An early version of the app would crash if the internet was disconnected, even when in offline mode.
    -   **Solution:** We diagnosed a logic flaw where the app was attempting to initialize the online Vertex AI service regardless of the mode. We refactored the logic to be truly conditional, ensuring that the app **only** attempts to connect to online services when it is explicitly in "Online Mode." This made the offline feature truly robust.

-   **Challenge: Accurate Image Matching with Varied Photos.** Pets in photos can be in different poses, lighting conditions, and distances from the camera.
    -   **Solution:** We implemented a two-stage matching system. The initial fast pHash scan handles the easy matches, and the more powerful ORB feature detection provides a deep-scan fallback. This allows the system to be both fast and accurate, correctly identifying pets even in challenging photos.

-   **Challenge: Securing Credentials.** The app requires API keys for both Hugging Face and Google Cloud.
    -   **Solution:** We used Streamlit's built-in Secrets Management (`secrets.toml`), ensuring that no sensitive keys were ever hardcoded into the source code, following security best practices.
