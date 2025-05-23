# AI-Powered Multimodal Plant Health Assistant - Backend

This backend server powers a sophisticated, AI-driven chatbot designed to assist users in diagnosing plant health issues. It supports multimodal interactions (text, voice, image), leverages multiple Large Language Models (local Gemma and cloud-based Gemini), and incorporates features like session management, RAG, conversation summarization, and real-time speech processing. Given that the setup can fit on a 6gb vram setup, this chatbot can be easily hosted by a farmer on a laptop under 8gb of VRAM, which can be accessed by the farmer's phone for mobile use through exposing the local backend server and front-end client with tunnelling tools like pinggy or ngrok

Next plan in the near future is adding in the ability for users to create their own "Knowledge Base" as grounding truth for the model. The model itself also will be grounded on a custom dataset with 38 classes of plant diseases on over 70k samples: https://huggingface.co/datasets/minhhungg/plant-disease-dataset. More details of the making of the dataset is here: https://github.com/alberttrann/plant-disease-dataset

Here is the link to the front-end client: (https://github.com/alberttrann/react-vite-ts)

## Table of Contents

1.  [Overview](#overview)
2.  [Features](#features)
3.  [Architecture](#architecture)
    *   [Core Components](#core-components)
    *   [Data Flow](#data-flow)
4.  [Technologies Used](#technologies-used)
5.  [Setup and Installation](#setup-and-installation)
    *   [Prerequisites](#prerequisites)
    *   [Backend Setup](#backend-setup)
    *   [Environment Variables & Configuration](#environment-variables--configuration)
6.  [Running the Server](#running-the-server)
7.  [Deployment & Public Exposure](#Deployment-&-Public-Exposure)
8.  [Key Backend Modules/Classes](#key-backend-modulesclasses)
9.  [API Key Management](#api-key-management)
10.  [Developer Toggles](#developer-toggles)
11. [Folder Structure (Backend)](#folder-structure-backend)
12. [Troubleshooting](#troubleshooting)
13. [Future Enhancements](#future-enhancements)

## 1. Overview

The backend provides a WebSocket endpoint for real-time communication with the frontend client. It handles user inputs, orchestrates calls to various AI/ML models, manages conversation state and history, and persists data. The core functionality revolves around processing user queries about plant health, potentially analyzing uploaded images, and generating informative responses or clarifying questions.

## 2. Features

*   **Real-time WebSocket Communication:** For interactive chat experience.
*   **Multimodal Input Processing:**
    *   Text queries.
    *   Image uploads/captures (passed as base64).
    *   Real-time voice input (streamed from client).
*   **Dual LLM Backend:**
    *   **Local Gemma-3-4B-IT:** Default model for image description and main chat responses.
    *   **Google Gemini 2.0 Flash API:** Alternative model for main chat responses (user-toggleable) and for "Evaluation Mode."
*   **AI-Powered Image Analysis:** Gemma (or Gemini) generates textual descriptions of plant images.
*   **Conversational Diagnosis:** Engages in iterative dialogue, asking clarifying questions.
*   **Retrieval Augmented Generation (RAG):**
    *   Uses `SentenceTransformer` (all-MiniLM-L6-v2) for embeddings.
    *   Stores and retrieves conversational context (messages, image descriptions, summaries) from `ChromaDB` vector store.
*   **Conversation Summarization:**
    *   Utilizes `Qwen2.5-0.5B-Instruct` to periodically generate structured summaries of ongoing conversations.
    *   Summaries are used in RAG.
*   **Speech Services:**
    *   **VAD (Voice Activity Detection):** Custom `AudioSegmentDetector` for identifying speech in audio streams.
    *   **ASR (Automatic Speech Recognition):** `Whisper-small` for transcribing voice to text.
    *   **TTS (Text-to-Speech):** `Kokoro-82M` for vocalizing AI responses.
*   **Session Management:**
    *   Creation, retrieval, renaming, and deletion of chat sessions.
    *   Persistence of sessions and messages in an SQLite database.
*   **Persistent API Key Storage:** Gemini API key stored in `server_settings.json`.
*   **Developer Toggles:** Client-controlled toggles to switch LLMs, activate evaluation mode, and (planned) grounding.
*   **Comprehensive Logging:** For debugging and monitoring.

## 3. Architecture

### Core Components

*   **WebSocket Server (`handle_client`):** Manages client connections and communication tasks.
*   **Request Router (`receive_data_from_client`):** Parses incoming messages and directs them.
*   **State Manager:** Global variables for API keys and toggle states, persisted in `server_settings.json`.
*   **Main Processor (`process_user_input_and_respond`):** Orchestrates the response generation pipeline.
*   **LLM Processors (`GemmaMultimodalProcessor`, `GeminiAPIProcessor`):** Abstractions for interacting with the respective LLMs.
*   **AI Services:** VAD, ASR, TTS, Summarizer, RAG System.
*   **Data Stores:** SQLite, ChromaDB, File System (for media and settings).

*(Refer to the Architecture Diagram provided separately for a visual representation)*

### Data Flow (Simplified for a text/image query)

1.  Client sends `text_input` (with optional image) via WebSocket.
2.  `handle_client` receives, `receive_data_from_client` routes to `process_user_input_and_respond`.
3.  User message and image (if any) are saved. User turn added to Gemma's history manager.
4.  If Gemini toggle OFF:
    *   Gemma describes the image (if present); description sent to client & saved.
    *   Gemma generates main response using RAG, user query, and image description.
5.  If Gemini toggle ON (and API key valid):
    *   RAG context is fetched.
    *   Gemini API is called with system prompt, RAG context, user query, and raw image bytes.
6.  Main AI response sent to client.
7.  If Eval toggle ON (and API key valid):
    *   Latest summary, user query, and image are sent to Gemini for an eval response.
    *   Eval response sent to client.
8.  All relevant messages/responses are saved to SQLite and embedded into ChromaDB.
9.  TTS is triggered if requested.
10. Summarization is triggered if conversation length threshold is met.

## 4. Technologies Used

*   **Python 3.10+** with **AsyncIO**
*   **`websockets`**: WebSocket server implementation.
*   **Hugging Face Libraries:**
    *   `transformers`: For loading and using Gemma, Whisper, Qwen, and managing tokenizers/processors.
    *   `sentence-transformers`: For text embeddings.
    *   `accelerate` (implied by `device_map="auto"`).
    *   `bitsandbytes`: For 4-bit quantization.
*   **AI Models:**
    *   Gemma-3-4B-IT (Unsloth quantized version)
    *   Whisper-small
    *   Kokoro-82M (TTS)
    *   Qwen2.5-0.5B-Instruct (Summarization)
    *   all-MiniLM-L6-v2 (Embeddings)
*   **Google Gemini API:**
    *   `google-genai` library for Python.
    *   `gemini-2.0-flash` model.
*   **Data Storage:**
    *   `sqlite3`: For relational data (sessions, messages, summaries).
    *   `chromadb`: For vector storage and similarity search (RAG).
    *   File System (`os`, `shutil`): For image files and `server_settings.json`.
*   **Other Python Libraries:**
    *   `Pillow` (PIL): Image processing.
    *   `numpy`: Numerical operations.
    *   `json`, `logging`, `re`, `uuid`, `datetime`, `functools`.

## 5. Setup and Installation

### Prerequisites

*   Python 3.10 or higher.
*   `pip` for package installation.
*   NVIDIA GPU with CUDA installed (recommended for local model performance).
*   A Google Gemini API Key (for Gemini features).

### Backend Setup

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone https://github.com/alberttrann/plantdiseaseserver.git
    cd plantdiseaseserver
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # Activate it:
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    A `requirements.txt` file would be ideal. Based on the imports, you'll need at least:
    ```bash
    pip install websockets torch torchvision torchaudio transformers sentence-transformers Pillow numpy accelerate bitsandbytes google-genai chromadb uvicorn # If using uvicorn for serving
    # For Kokoro TTS, installation might be more specific if not on PyPI directly
    pip install kokoro
    ```
    *(Ensure you install versions compatible with your CUDA setup if using GPU.)*

4.  **Directory Setup:**
    *   The script will automatically create `media_storage/` and `F:\CHROMA` (for ChromaDB, path is hardcoded, consider making it configurable) and `server_settings.json` if they don't exist. Ensure the parent directory (`F:\` in Chroma's case) is writable.

### Environment Variables & Configuration

*   **Gemini API Key:**
    *   The system will prompt for this via the client UI if not found in `server_settings.json`.
    *   Alternatively, you can pre-populate `server_settings.json` in the same directory as `main-gemma3.py`:
        ```json
        {
            "gemini_api_key": "YOUR_GEMINI_API_KEY_HERE"
        }
        ```
*   **Model Paths (Hardcoded):** Model IDs for Hugging Face models are currently hardcoded in `main()`.
*   **ChromaDB Path (`CHROMA_DB_PATH`):** Currently hardcoded to `F:\CHROMA`. Modify this constant in the script if needed.
*   **Other Constants:** Review constants at the top of `main-gemma3.py` for default behaviors (e.g., `SUMMARY_TRIGGER_THRESHOLD`).

## 6. Running the Server

1.  **Activate your virtual environment.**
2.  **Navigate to the directory containing `main-gemma3.py`.**
3.  **Run the script:**
    ```bash
    python main-gemma3.py
    ```
4.  The server will start, pre-load models, and listen on `0.0.0.0:9073` by default.
5.  Connect your frontend client to `ws://127.0.0.1:9073`.

## 7. Deployment & Public Exposure 

This section provides guidance on how to expose your locally running backend server to the public internet for testing with a publicly accessible frontend, primarily using tunneling services like ngrok or Pinggy. This is suitable for development, prototyping, and small-sized personal use. 

**Note:** For actual large-scale production deployment, consider containerization (Docker), cloud platforms (AWS, GCP, Azure), and proper reverse proxies (Nginx, Caddy) with SSL certificate management.

### Using Tunneling Services (ngrok, Pinggy, etc.)

Tunneling services create a secure tunnel from a public URL to your local machine.

1.  **Choose a Tunneling Service:**
    *   **ngrok:** Popular and well-documented. Offers HTTP/S and TCP tunnels. Free tier provides dynamic URLs.
        *   Website: [ngrok.com](https://ngrok.com/)
    *   **Pinggy:** Another option, also with free tiers.
        *   Website: [pinggy.io](https://pinggy.io/)
    *   Others: `localtunnel`, `serveo` (availability may vary).

2.  **Install the Tunneling Client:** Follow the instructions on the chosen service's website to download and set up their client.

3.  **Expose Your Backend WebSocket Server:**
    *   Your backend Python server runs (by default) on `localhost:9073`.
    *   **Using ngrok (Recommended for WebSockets on HTTP/S or TCP):**
        *   **Option A (HTTP/S Tunnel - Handles WebSocket Upgrades):**
            This is often preferred if you want a `wss://` (secure WebSocket) URL.
            ```bash
            ngrok http 9073
            ```
            ngrok will output forwarding URLs, e.g.:
            `Forwarding                    https://<random-id>.ngrok-free.app -> http://localhost:9073`
            Your public WebSocket URL will be: `wss://<random-id>.ngrok-free.app`
        *   **Option B (TCP Tunnel - More Generic):**
            This forwards raw TCP traffic. Good if HTTP/S upgrade handling by ngrok causes issues, or if your WS server is not on a standard HTTP port.
            ```bash
            ngrok tcp 9073
            ```
            ngrok will output a TCP address, e.g.:
            `Forwarding                    tcp://0.tcp.ngrok.io:<random-port> -> localhost:9073`
            Your public WebSocket URL will be: `ws://0.tcp.ngrok.io:<random-port>` (Note: `ws://` not `wss://` unless you configure TLS termination separately).

    *   **Using Pinggy (Refer to their specific commands):**
        You mentioned using `pinggy.link`. The command would be specific to how Pinggy handles TCP or WebSocket traffic.
        Example (hypothetical, **check Pinggy docs**):
        ```bash
        # For a raw TCP tunnel which WebSockets can use:
        pinggy tcp -p 9073 wss://your-desired-subdomain.a.free.pinggy.link 
        # Or it might give you a generic tcp.pinggy.io:PORT address
        ```
        Ensure the Pinggy tunnel type supports raw TCP or explicit WebSocket proxying.

4.  **Update Frontend Configuration:**
    *   In your frontend application (`src/App.tsx` or via environment variables like `.env`), set the `WEBSOCKET_URL` to the public URL provided by your tunneling service for the backend.
    *   Example for ngrok HTTP tunnel: `const WEBSOCKET_URL = "wss://<random-id>.ngrok-free.app";`
    *   Example for ngrok TCP tunnel: `const WEBSOCKET_URL = "ws://0.tcp.ngrok.io:<random-port>";`
    *   Example for your Pinggy URL: `const WEBSOCKET_URL = "wss://rnapn-42-117-46-84.a.free.pinggy.link";` (if this is indeed your backend tunnel).

5.  **Configure Backend `allowed_origins`:**
    *   Your Python WebSocket server (`main-gemma3.py`) uses the `origins` parameter in `websockets.serve()` to restrict which frontend origins can connect.
    *   If your frontend is also tunneled (e.g., Vite dev server accessed via `https://my-frontend.ngrok-free.app`), you **must** add this frontend's public origin to the `allowed_origins` list in `main-gemma3.py`.
        ```python
        # In main-gemma3.py, inside the main() function
        allowed_origins = [
            "http://localhost:5173",      # Local Vite dev server
            "http://127.0.0.1:5173",    # Also for local Vite
            "https://your-frontend-public-url.ngrok-free.app", # Replace with your frontend's ngrok/pinggy URL
            "https://another-frontend-public-url.pinggy.link"  # If using pinggy for frontend
        ]
        # ...
        server = await websockets.serve(
            handle_client, 
            addr, 
            port, 
            # ... other params ...
            origins=allowed_origins
        )
        ```
    *   **Important:** The origin is the scheme, hostname, and port (if not default). For `https://my-frontend.ngrok-free.app/some/path`, the origin is `https://my-frontend.ngrok-free.app`.

6.  **Configure Frontend Dev Server `allowedHosts` (Vite):**
    *   If you are also tunneling your Vite frontend dev server and accessing it via its public ngrok/pinggy URL, you must add that public hostname to your `vite.config.js` (or `.ts`) `server.allowedHosts` array. This allows Vite to serve requests to that hostname.
        ```javascript
        // vite.config.js
        export default defineConfig({
          // ...
          server: {
            host: true, // Listen on all interfaces
            allowedHosts: [
              'your-frontend-public-hostname.ngrok-free.app',
              'your-frontend-public-hostname.pinggy.link',
            ],
          },
        });
        ```

7.  **Restart Servers:**
    *   After making changes to backend `allowed_origins` or frontend `WEBSOCKET_URL` or `vite.config.js`, restart the respective servers.

### Common Issues & Tips when Tunneling:

*   **HTTP vs. TCP Tunnels:** For WebSockets, TCP tunnels are often more reliable as they forward raw traffic. HTTP/S tunnels need to correctly handle the WebSocket `Upgrade` handshake. Most modern versions of ngrok handle this well for HTTP/S tunnels. Verify your chosen service's WebSocket support.
*   **`ws://` vs. `wss://`:**
    *   If your tunnel provides an `https://` endpoint, use `wss://` for your WebSocket URL.
    *   If your tunnel provides a raw `tcp://` endpoint, use `ws://`.
    *   Mixing these (e.g., `ws://` to an `https` ngrok endpoint) will likely fail.
*   **Dynamic Tunnel URLs:** Free tiers of services like ngrok provide dynamic URLs that change each time you restart the ngrok client. You'll need to update your frontend's `WEBSOCKET_URL` (and potentially backend's `allowed_origins` if your frontend URL also changes) accordingly. Using environment variables for `WEBSOCKET_URL` in the frontend can make this easier to manage.
*   **CORS vs. WebSocket Origins:**
    *   `allowedHosts` in Vite is a dev server security feature.
    *   `origins` in the Python `websockets` library is a WebSocket security feature similar to CORS, controlling which frontend origins can establish a WebSocket connection.
*   **Firewalls:** Ensure your local machine's firewall allows the tunneling client to make outbound connections and your Python server to accept connections on the specified port (e.g., `9073`) from the tunneling client (usually from `localhost` or `127.0.0.1` as the tunnel client runs locally).
*   **Tunnel Inspector:** Use the web inspector provided by your tunneling service (e.g., `http://localhost:4040` for ngrok) to see incoming requests to your tunneled backend port. This is invaluable for debugging if connections are even reaching the tunnel.

## 8. Key Backend Modules/Classes

*   **`main-gemma3.py`:** The main script containing all logic.
*   **`GeminiAPIProcessor`:** Handles all interactions with the Google Gemini API.
*   **`GemmaMultimodalProcessor`:** Handles interactions with the local Gemma model.
*   **`SummarizationProcessor`:** Manages conversation summarization using the Qwen model.
*   **`WhisperTranscriber`:** Transcribes audio to text.
*   **`KokoroTTSProcessor`:** Synthesizes text to speech.
*   **`AudioSegmentDetector`:** Performs Voice Activity Detection.
*   **Database Functions (`init_db`, `save_message_db`, etc.):** Manage SQLite operations.
*   **`handle_client`:** Manages individual WebSocket client connections and their lifecycles.
*   **`process_user_input_and_respond`:** Core logic for processing user queries and generating AI responses.

## 9. API Key Management

*   The Gemini API key is required for "Use Gemini" and "Eval Mode" toggles.
*   The client UI should prompt the user for the key if needed.
*   The key is sent to the backend via a `set_api_key` WebSocket message.
*   The backend stores this key in `server_settings.json` for persistence across server restarts.
*   The `GEMINI_API_KEY_STORE` global variable holds the key in memory during runtime.

## 10. Developer Toggles

Controlled via `update_toggle_state` WebSocket messages from the client. Their states are stored in global backend variables:

*   **`GLOBAL_USE_GEMINI_MODEL` (boolean):** If `True` and API key is set, Gemini API is used for main responses. Otherwise, local Gemma is used.
*   **`GLOBAL_EVAL_MODE_ACTIVE` (boolean):** If `True` and API key is set, an additional "evaluation" response is generated by Gemini after the main AI response.
*   **`GLOBAL_GROUNDING_ACTIVE` (boolean):** Placeholder for future knowledge base grounding feature.

## 11. Folder Structure (Backend - Simplified)
```
alberttran/plantdiseaseserver
├── main-gemma3.py # Main backend script
├── server_settings.json # Stores Gemini API key (created by script)
├── chat_history.db # SQLite database (created by script)
├── media_storage/ # Stores uploaded images (created by script)
│ └── <session_id>/
│ └── <image_filename.jpg>
├── F:/CHROMA/ # ChromaDB persistence path (as configured)
└── .venv/ # Python virtual environment
```
## 12. Troubleshooting

*   **`AttributeError` on startup or during calls:**
    *   Ensure all Python dependencies are installed correctly.
    *   Verify class definitions are complete and appear before their usage in the script.
    *   Delete `__pycache__` directories and restart.
*   **Model Loading Failures (OOM, etc.):**
    *   Ensure you have sufficient VRAM/RAM. 4-bit quantized models still require significant resources.
    *   Check CUDA and PyTorch compatibility.
    *   Try reducing the number of models loaded simultaneously if VRAM is an issue (e.g., by disabling TTS or using a smaller Whisper model if not already).
*   **Gemini API Errors:**
    *   Check that `GEMINI_API_KEY_STORE` is correctly set and the key is valid with permissions for `gemini-2.0-flash`.
    *   Look for specific error messages from the Gemini API in the backend logs.
*   **Slow Performance:**
    *   Local LLM inference (especially Gemma-3-4B on limited hardware) can be slow. Timeouts have been increased, but further optimization (like Unsloth's `FastLanguageModel` if reconsidered) or hardware upgrades might be needed for production-like speeds.
*   **WebSocket Connection Issues:**
    *   Ensure the backend server is running and accessible on the configured host/port.
    *   Check for firewall issues.

## 13. Future Enhancements

*   Implement the "Grounding" feature using a custom Huggingface dataset, with data distilled from a Gemini model
*   More robust UI synchronization for toggle states.
*   User-specific API key storage if deploying for multiple users.
*   Frontend UI for "Secrets" panel to manage API keys.
*   Performance optimization for local LLM inference.
*   Streamed responses from LLMs back to the client for better perceived responsiveness.
*   More sophisticated error handling and user feedback on the frontend.
*   Configuration options for model selection, paths, etc., via a config file instead of hardcoding.


Some images from the web app:
  
![Screenshot 2025-05-17 000228](https://github.com/user-attachments/assets/a2be00d5-656d-4e55-8278-e982ba8b2677)

![Screenshot 2025-05-17 000110](https://github.com/user-attachments/assets/f28e54c4-a8ae-4df4-8a1c-3439e07dc7b6)

![Screenshot 2025-05-17 011027](https://github.com/user-attachments/assets/bf909c93-7fc7-41e6-822f-859b95dc9409)




