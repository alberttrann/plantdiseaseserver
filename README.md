# AI-Powered Multimodal Plant Health Assistant - Backend

This backend server powers a sophisticated, AI-driven chatbot designed to assist users in diagnosing plant health issues. It supports multimodal interactions (text, voice, image), leverages multiple Large Language Models (local Gemma and cloud-based Gemini), and incorporates features like session management, RAG, conversation summarization, and real-time speech processing.

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
7.  [Key Backend Modules/Classes](#key-backend-modulesclasses)
8.  [API Key Management](#api-key-management)
9.  [Developer Toggles](#developer-toggles)
10. [Folder Structure (Backend)](#folder-structure-backend)
11. [Troubleshooting](#troubleshooting)
12. [Future Enhancements](#future-enhancements)

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

## 7. Key Backend Modules/Classes

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

## 8. API Key Management

*   The Gemini API key is required for "Use Gemini" and "Eval Mode" toggles.
*   The client UI should prompt the user for the key if needed.
*   The key is sent to the backend via a `set_api_key` WebSocket message.
*   The backend stores this key in `server_settings.json` for persistence across server restarts.
*   The `GEMINI_API_KEY_STORE` global variable holds the key in memory during runtime.

## 9. Developer Toggles

Controlled via `update_toggle_state` WebSocket messages from the client. Their states are stored in global backend variables:

*   **`GLOBAL_USE_GEMINI_MODEL` (boolean):** If `True` and API key is set, Gemini API is used for main responses. Otherwise, local Gemma is used.
*   **`GLOBAL_EVAL_MODE_ACTIVE` (boolean):** If `True` and API key is set, an additional "evaluation" response is generated by Gemini after the main AI response.
*   **`GLOBAL_GROUNDING_ACTIVE` (boolean):** Placeholder for future knowledge base grounding feature.

## 10. Folder Structure (Backend - Simplified)
your-project-root/
├── main-gemma3.py # Main backend script
├── server_settings.json # Stores Gemini API key (created by script)
├── chat_history.db # SQLite database (created by script)
├── media_storage/ # Stores uploaded images (created by script)
│ └── <session_id>/
│ └── <image_filename.jpg>
├── F:/CHROMA/ # ChromaDB persistence path (as configured)
└── .venv/ # Python virtual environment

## 11. Troubleshooting

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

## 12. Future Enhancements

*   Implement the "Grounding" feature using the provided JSON knowledge base.
*   More robust UI synchronization for toggle states.
*   User-specific API key storage if deploying for multiple users.
*   Frontend UI for "Secrets" panel to manage API keys.
*   Performance optimization for local LLM inference.
*   Streamed responses from LLMs back to the client for better perceived responsiveness.
*   More sophisticated error handling and user feedback on the frontend.
*   Configuration options for model selection, paths, etc., via a config file instead of hardcoding.
