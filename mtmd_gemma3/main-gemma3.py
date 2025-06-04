import asyncio
import json
import websockets
import base64
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    BitsAndBytesConfig,
    AutoModelForCausalLM, 
    AutoModelForImageTextToText,
    GenerationConfig,
    StoppingCriteria, StoppingCriteriaList,
    AutoTokenizer
)
import numpy as np
import logging
import sys
import io
from PIL import Image
import time
import os
from datetime import datetime
from kokoro import KPipeline
import re
import uuid
import sqlite3
import shutil
import functools

from sentence_transformers import SentenceTransformer
from websockets.connection import State as WebSocketConnectionState
from google import genai
from google.genai import types
import chromadb

# --- Constants ---
SETTINGS_FILE_PATH = "settings.json"
DATABASE_PATH = "chat_history.db"
MEDIA_STORAGE_DIR = "media_storage"
CHROMA_DB_PATH = "F:\\CHROMA"
RAG_MAX_RESULTS = 3
SHORT_TERM_MEMORY_MAX_TOKENS = 1500
SUMMARY_TRIGGER_THRESHOLD = 4 

REVISED_SIMPLIFIED_SYSTEM_PROMPT = """You are a helpful plant health assistant.
- If an image description is provided, use it.
- If critical info (plant name, symptoms, location, weather) is missing for diagnosis, ask for it.
- If enough info, give a preliminary analysis.
- If off-topic, state your role and guide to plant topics.
- Use relevant past conversation context.
- Provide concise, clear responses.
"""


GEMMA_IMAGE_DESCRIBER_PROMPT_TEMPLATE = """You are an image analysis expert specializing in plant health. Describe the following image of a plant component in detail. Focus on visual characteristics such as color, shape, texture, lesions, spots, wilting, or any abnormalities. Be concise (around 3 sentences) but informational. This description will be used by another AI to help diagnose plant issues.
User's original query for context: '{user_query}'

Detailed image description of the provided image:
"""
IMAGE_CLARIFICATION_GUIDANCE_PROMPT_TEMPLATE = """You are a plant-disease assistant.
An image was provided by the user, and you (or another AI module) have generated a description of it.
Your current task is to use this image description AND the user's original prompt/query to ask smart, targeted clarifying questions. These questions should help gather more information needed for a reliable diagnosis later.
Do NOT provide any diagnosis or treatment advice in this step. Only ask 1-3 clarifying questions. Be concise

User's Original Prompt/Query:
{user_query}

AI-Generated Image Description:
{image_description}

Based on the above, what specific clarifying questions (1-3) should be asked to the user to get more details for a diagnosis?
Your questions:
"""

SUMMARIZER_SYSTEM_PROMPT = """You are a text summarization agent. Your task is to extract key information from the conversation and structure it.

Follow this template STRICTLY for your output. If information for a field is not present, use "Not specified" or "N/A". Don't assume anything not mentioned in the conversation. Only write 1-3 sentences per field.
---
Plant & Variety: 
Age: 
Location: 
Weather: 
Symptoms (User Reported & Image Derived): 
Key AI Questions: [List 1-2 most important unanswered questions by AI, or last questions asked]
Key User Answers: [List key answers user provided to AI questions]
Potential AI Diagnosis: [If AI offered a preliminary diagnosis]
AI Recommendations: [Key advices by AI]
Other Notable Points: [Other crucial brief facts]
---

Instructions for Updating:
- If <Existing Summary> is provided, UPDATE the fields based on <New Turns>. Do NOT just append. Replace or add to existing field values.
- If no <Existing Summary>, create a new summary using the template based on <Conversation Turns>.
- Keep field values VERY concise (phrases, short sentences).
- Output ONLY the completed template.
"""

GEMINI_CHAT_SYSTEM_PROMPT_FOR_MAIN_RESPONSE = """**Your Role:**
You are an AI-powered Specialist Plant Disease Assistant. Your primary function is to help users identify potential diseases, pests, or abiotic disorders affecting their plants. Your goal is to provide a preliminary analysis and actionable, safe advice based on the information provided. Maintain a helpful, empathetic, and knowledgeable tone.

**Core Operational Guidelines:**

1.  **Image Description Utilization:**
    *   If an image or a detailed description of an image is provided by the user, meticulously analyze it for visual cues. Focus on:
        *   **Symptom Type:** e.g., leaf spots, wilting, discoloration (yellowing, browning, blackening), lesions, cankers, galls, powdery mildew, rust, rot, abnormal growths.
        *   **Symptom Location:** e.g., on new leaves, old leaves, stems, roots (if visible/described), flowers, fruits, veins, leaf margins.
        *   **Symptom Pattern:** e.g., uniform, random, concentric rings, along veins, spreading from a point.
        *   **Presence of Pests or Fungal Structures:** e.g., insects, mites, webbing, mycelium, spores, fruiting bodies.

2.  **Information Gathering (If Critical Information is Missing for Diagnosis):**
    *   If the information provided is insufficient for even a preliminary analysis, proactively ask clear, targeted questions to gather **critical diagnostic details**. Prioritize:
        *   **A. Precise Plant Identification:**
            *   Ask for the common name and, if possible, the species or variety of the affected plant (e.g., "Tomato, variety 'Beefsteak'" not just "vegetable").
        *   **B. Detailed Symptom Description (if not clear from image/initial description):**
            *   **What exactly do you see?** (e.g., "Small brown spots with yellow halos," "Leaves are wilting from the bottom up," "White powdery substance on leaves").
            *   **Where on the plant are symptoms most prominent?** (e.g., "Only on new growth," "Primarily on the undersides of leaves").
            *   **When did symptoms first appear, and how have they progressed?** (e.g., "Started 2 days ago, spreading rapidly," "Been slow for weeks").
            *   **What percentage or part of the plant is affected?** (Severity).
        *   **C. Environmental and Care Context:**
            *   **Geographic Location:** General region or climate zone (e.g., "Pacific Northwest, USA," "Southern Florida").
            *   **Growing Conditions:** Indoor, outdoor, greenhouse, container (material/size), raised bed, in-ground.
            *   **Recent Weather Patterns:** Significant changes in temperature (heatwaves, frost), humidity, rainfall, prolonged leaf wetness, strong winds.
            *   **Watering Practices:** Frequency, method (overhead, base), amount, recent changes.
            *   **Soil/Potting Mix:** Type (if known), drainage.
            *   **Fertilization:** Type, frequency, last application.
            *   **Recent Treatments/Changes:** Any pesticides, fungicides, herbicides used recently? Repotting, pruning, or other disturbances?
            *   **Nearby Plants:** Are other plants (same or different species) showing similar symptoms?

3.  **Analysis and Advice (If Sufficient Information is Available):**
    *   Provide a **preliminary analysis**, clearly stating it as such.
    *   Suggest **1-3 most likely diseases, pests, or abiotic disorders** based on the evidence.
    *   **Explain your reasoning** by correlating the provided symptoms and conditions with known characteristics of the suggested issues.
    *   Offer **general management recommendations**, prioritizing Integrated Pest Management (IPM) principles:
        *   Cultural controls (e.g., sanitation, pruning, improving air circulation, adjusting watering).
        *   Biological controls (if appropriate and feasible).
        *   Less-toxic chemical controls as a last resort, always advising to read and follow label instructions carefully and test on a small area first.
    *   Provide **preventative tips** for future plant health.
    *   **Crucial Disclaimer:** Always explicitly state that your analysis is preliminary and based on the information provided. For a definitive diagnosis and critical situations, recommend consulting a local horticultural extension office, certified arborist, or plant pathology lab.

4.  **Contextual Continuity:**
    *   Actively use relevant information and questions from the **past conversation history** to avoid repetition and build upon existing knowledge. Refer to previous details if they are pertinent to the current query.

5.  **Scope Management:**
    *   If the user's query is **off-topic** (e.g., asking for recipes, general gardening advice not related to a specific plant health problem, non-plant topics), politely state your specialized role as a "Plant Disease Assistant." Gently redirect the conversation back to plant health issues, disease identification, or pest problems.

6.  **Response Style:**
    *   Provide **concise, clear, and actionable responses.**
    *   Use bullet points or numbered lists for questions, symptoms, or recommendations to enhance readability.
    *   Avoid overly technical jargon unless explained.
    *   Maintain a professional, supportive, and understanding tone.""" # Ensure this is the full prompt
GEMINI_API_KEY_STORE = None # Simple in-memory store for API key for now

# --- Global States for Toggles (default to False) ---
GLOBAL_USE_GEMINI_MODEL = False
GLOBAL_EVAL_MODE_ACTIVE = False
GLOBAL_GROUNDING_ACTIVE = False # Placeholder for later

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(process)d - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

GLOBAL_WHISPER_TRANSCRIBER = None
GLOBAL_KOKORO_TTS_PROCESSOR = None
GLOBAL_GEMMA_MODEL = None
GLOBAL_GEMMA_PROCESSOR = None
GLOBAL_EMBEDDING_MODEL = None
GLOBAL_CHROMA_CLIENT = None
GLOBAL_SMOL_MODEL = None     
GLOBAL_SMOL_TOKENIZER = None 
GLOBAL_SUMMARIZER = None      
GLOBAL_LLM_INFERENCE_LOCK = asyncio.Lock()

# Located after global variables and before other major classes/functions
class GeminiAPIProcessor:
    def __init__(self, api_key: str):
        self.instance_id_log = str(id(self))[-6:]
        if not api_key:
            logger.error(f"GeminiAPIProc Inst:{self.instance_id_log} Initialization failed: API key is missing.")
            raise ValueError("Gemini API key is required for GeminiAPIProcessor.")
        
        try:
            self.client = genai.Client(api_key=api_key) # Key passed here as per sample
            # Optional: Test client by listing models, as done before
            # _ = list(self.client.list_models()) 
            logger.info(f"GeminiAPIProc Inst:{self.instance_id_log} GenAI client CREATED with API key.")
        except Exception as e:
            logger.error(f"GeminiAPIProc Inst:{self.instance_id_log} Failed to create GenAI client: {e}", exc_info=True)
            self.client = None 
            raise 
        
        self.default_model_name = "gemini-2.0-flash" # From your sample code
        self.generation_count = 0
        logger.info(f"GeminiAPIProc Inst:{self.instance_id_log} initialized. Default model: {self.default_model_name}")

    async def _generate_content_with_gemini(self, model_name_override: str = None, system_instruction_text: str = None, user_parts: list = None, tools: list = None, stream: bool = False, response_mime_type="text/plain"):
        self.generation_count += 1
        model_to_use_for_api = model_name_override if model_name_override else self.default_model_name
            
        log_prefix = f"GeminiAPIProc Inst:{self.instance_id_log} GenCall {self.generation_count} (Model: {model_to_use_for_api}):"
        
        if not self.client:
            logger.error(f"{log_prefix} Gemini client not initialized.")
            return "[Error: Gemini client not initialized]"

        try:
            gen_config_parts = {}
            if system_instruction_text:
                gen_config_parts["system_instruction"] = types.Content(parts=[types.Part.from_text(text=system_instruction_text)])
            if tools:
                gen_config_parts["tools"] = tools 
            if response_mime_type:
                 gen_config_parts["response_mime_type"] = response_mime_type
            
            generation_config_api = types.GenerateContentConfig(**gen_config_parts)
            contents_for_api = [types.Content(role="user", parts=user_parts if user_parts else [])]

            logger.debug(f"{log_prefix} Preparing for Gemini call. Model: {model_to_use_for_api}, SystemInstruction: '{system_instruction_text[:100] if system_instruction_text else 'None'}...', UserParts: {len(user_parts if user_parts else [])}")

            loop = asyncio.get_event_loop()
            generated_text = ""

            # Define the synchronous function to be run in executor
            def sync_call_gemini():
                full_text = ""
                # This directly matches your sample's generation call structure for streaming
                response_stream = self.client.models.generate_content_stream(
                    model=model_to_use_for_api, # e.g., "gemini-2.0-flash"
                    contents=contents_for_api,
                    config=generation_config_api 
                )
                for chunk in response_stream:
                    # According to your sample, chunk should directly have .text
                    # If chunk.parts exists, it's more complex; but sample implies chunk.text
                    current_chunk_text = ""
                    if hasattr(chunk, 'text') and chunk.text:
                        current_chunk_text = chunk.text
                    elif chunk.parts: # Fallback if structure is more complex
                        for part in chunk.parts:
                            if hasattr(part, 'text') and part.text:
                                current_chunk_text += part.text
                    
                    if current_chunk_text:
                        if stream:
                            # If truly streaming back to client, would yield here.
                            # For now, we collect for both stream=True and stream=False cases.
                            # logger.debug(f"{log_prefix} Stream chunk: {current_chunk_text[:50]}...")
                            pass # Placeholder if we were to implement true async streaming yield
                        full_text += current_chunk_text
                return full_text.strip()

            generated_text = await loop.run_in_executor(None, sync_call_gemini)
            
            if stream:
                logger.info(f"{log_prefix} Collected streamed response (length: {len(generated_text)}).")
            else:
                logger.info(f"{log_prefix} Non-streamed response (collected from stream wrapper) received (length: {len(generated_text)}).")
            
            logger.info(f"{log_prefix} Gemini generated text (final, len {len(generated_text)}): '{generated_text[:200]}...'")
            return generated_text

        except AttributeError as ae:
            logger.error(f"{log_prefix} AttributeError during Gemini API call setup or processing: {ae}.", exc_info=True)
            return f"[Error: Gemini client library usage error - {ae}]"
        except Exception as e:
            logger.error(f"{log_prefix} Error during Gemini API call: {e}", exc_info=True)
            if "API_KEY_INVALID" in str(e) or "API key not valid" in str(e) or "PermissionDenied" in str(e):
                return "[Error: Gemini API key is not valid or lacks permissions. Please check your key.]"
            if "pleaseAuthenticated" in str(e):
                 return "[Error: Gemini API authentication failed. Please check your API key.]"
            return f"[Error: Gemini API call failed - {type(e).__name__}]"

    # generate_response_for_chat and generate_eval_response remain the same as they call the corrected _generate_content_with_gemini
    async def generate_response_for_chat(self, system_prompt_text: str, user_query_text: str, image_bytes: bytes = None, rag_context: str = ""):
        user_parts_list = []
        full_user_text_prompt = ""
        if rag_context: 
            full_user_text_prompt += f"{rag_context}\n\n---\n\nUser Query:\n{user_query_text}"
        else: 
            full_user_text_prompt += f"User Query:\n{user_query_text}"
        
        user_parts_list.append(types.Part(text=full_user_text_prompt))

        if image_bytes:
            # Construct Part with inline_data as a dictionary
            image_part_data = {
                "mime_type": "image/jpeg", # Or the actual mime type
                "data": image_bytes
            }
            # Create the Part object, assigning the dictionary to inline_data
            # This assumes types.Part() will correctly create the nested structure,
            # or that inline_data can be directly assigned a dict that matches the expected schema.
            # If types.Part has an inline_data parameter in its constructor:
            try:
                 # Attempt 1: Pass as a kwarg if Part constructor supports it
                image_part = types.Part(inline_data=image_part_data)
            except TypeError: 
                # Attempt 2: Create Part then assign if constructor doesn't take inline_data directly
                logger.warning("GeminiAPIProc: types.Part constructor might not take inline_data directly. Trying assignment.")
                image_part = types.Part()
                image_part.inline_data = image_part_data # This might still fail if inline_data needs to be an object
            
            user_parts_list.append(image_part)
            logger.debug(f"GeminiAPIProc: Added image part. Mime: image/jpeg, Size: {len(image_bytes)}")
            
        return await self._generate_content_with_gemini(
            system_instruction_text=system_prompt_text,
            user_parts=user_parts_list,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            stream=False 
        )


    async def generate_eval_response(self, newest_summary: str, latest_user_query: str, latest_image_bytes: bytes = None):
        eval_system_prompt_template = """<EVAL>
You're a plant-disease expert, with great knowledge base. Your job is to provide an enhanced "clarification" answer that aims to resolve insufficient answering from a smaller model, based on a summarization of the chat session and the latest uploaded image """ 
        user_parts_for_eval = []
        eval_user_text = "Please provide an enhanced analysis based on the context in the system prompt and the provided image (if any)."
        user_parts_for_eval.append(types.Part(text=eval_user_text))

        if latest_image_bytes:
            image_part_data_eval = {
                "mime_type": "image/jpeg",
                "data": latest_image_bytes
            }
            try:
                image_part_eval = types.Part(inline_data=image_part_data_eval)
            except TypeError:
                logger.warning("GeminiAPIProc (Eval): types.Part constructor might not take inline_data directly. Trying assignment.")
                image_part_eval = types.Part()
                image_part_eval.inline_data = image_part_data_eval

            user_parts_for_eval.append(image_part_eval)
            logger.debug(f"GeminiAPIProc (Eval): Added image part. Mime: image/jpeg, Size: {len(latest_image_bytes)}")

        formatted_eval_system_prompt = eval_system_prompt_template.format(
            NEWEST_SUMMARIZATION=newest_summary, USER_QUERY_FOR_EVAL=latest_user_query
        )
        
        generated_text = await self._generate_content_with_gemini(
            model_name_override="gemini-2.0-flash", 
            system_instruction_text=formatted_eval_system_prompt,
            user_parts=user_parts_for_eval,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            stream=False
        )
        if generated_text and not generated_text.strip().startswith("<EVAL>"): return f"<EVAL>\n{generated_text.strip()}"
        elif not generated_text: return "<EVAL>\n[Gemini Eval: No response generated]"
        return generated_text

# >>> PLACE SETTINGS FUNCTIONS HERE <<<
def load_server_settings():
    global GEMINI_API_KEY_STORE # Allow modification of the global variable
    try:
        if os.path.exists(SETTINGS_FILE_PATH):
            with open(SETTINGS_FILE_PATH, 'r') as f:
                settings = json.load(f)
                GEMINI_API_KEY_STORE = settings.get("gemini_api_key") # Directly assign
                if GEMINI_API_KEY_STORE:
                    logger.info(f"Loaded Gemini API Key from {SETTINGS_FILE_PATH} (length: {len(GEMINI_API_KEY_STORE)}).")
                else:
                    logger.info(f"{SETTINGS_FILE_PATH} found, but no 'gemini_api_key' present or it's empty.")
        else:
            logger.info(f"{SETTINGS_FILE_PATH} not found. API key will need to be set by the client.")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {SETTINGS_FILE_PATH}: {e}. API key will be None.")
        GEMINI_API_KEY_STORE = None
    except Exception as e:
        logger.error(f"Error loading settings from {SETTINGS_FILE_PATH}: {e}", exc_info=True)
        GEMINI_API_KEY_STORE = None

def save_server_settings():
    global GEMINI_API_KEY_STORE # Access the global variable to save its current state
    settings_to_save = {
        "gemini_api_key": GEMINI_API_KEY_STORE # Save current value
    }
    try:
        with open(SETTINGS_FILE_PATH, 'w') as f:
            json.dump(settings_to_save, f, indent=4)
        logger.info(f"Server settings (API key) saved to {SETTINGS_FILE_PATH}.")
    except Exception as e:
        logger.error(f"Error saving settings to {SETTINGS_FILE_PATH}: {e}", exc_info=True)

load_server_settings()

def strip_markdown_for_tts(text):
    if not text: return ""
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text); text = re.sub(r'___(.*?)___', r'\1', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text); text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text); text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\*\-\+]\s+', '', text, flags=re.MULTILINE); text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'`(.*?)`', r'\1', text); text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text); text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'^\s*([-*_]){3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\n\s*', '\n', text); text = re.sub(r'\n{2,}', '. \n', text)
    text = text.replace('\n', '. '); text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\.\s*\.', '.', text); text = text.replace(' .', '.'); return text.strip()

def init_db():
    logger.info(f"Initializing database at {os.path.abspath(DATABASE_PATH)}...")
    if not os.path.exists(MEDIA_STORAGE_DIR):
        try: os.makedirs(MEDIA_STORAGE_DIR); logger.info(f"Created media storage directory: {MEDIA_STORAGE_DIR}")
        except OSError as e: logger.error(f"Failed to create media storage directory {MEDIA_STORAGE_DIR}: {e}")

    conn = sqlite3.connect(DATABASE_PATH); cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, name TEXT, created_at INTEGER, last_updated_at INTEGER )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS messages (id TEXT PRIMARY KEY, session_id TEXT, sender TEXT, timestamp INTEGER, text_content TEXT, image_filename TEXT, data_type TEXT DEFAULT 'text', llm_model_used TEXT, tts_audio_filename TEXT, FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE )''')
    cursor.execute('''CREATE INDEX IF NOT EXISTS idx_messages_session_id_timestamp ON messages (session_id, timestamp)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS session_summaries (
                        session_id TEXT PRIMARY KEY, 
                        summary_text TEXT, 
                        last_summarized_message_id TEXT, 
                        last_updated INTEGER,
                        FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                     )''')
    conn.commit(); conn.close(); logger.info(f"SQL Database initialized/checked at {DATABASE_PATH}")

    global GLOBAL_CHROMA_CLIENT
    try:
        if not os.path.exists(CHROMA_DB_PATH):
             logger.info(f"ChromaDB path {CHROMA_DB_PATH} does not exist. It will be created by ChromaDB.")
        GLOBAL_CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        logger.info(f"ChromaDB client initialized. Data will be persisted at: {os.path.abspath(CHROMA_DB_PATH)}")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB client: {e}")
        GLOBAL_CHROMA_CLIENT = None

def db_execute(query, params=(), fetchone=False, fetchall=False, commit=False):
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH); conn.row_factory = sqlite3.Row; cursor = conn.cursor()
        cursor.execute(query, params); result = None
        if fetchone: row = cursor.fetchone(); result = dict(row) if row else None
        if fetchall: rows = cursor.fetchall(); result = [dict(row) for row in rows] if rows else []
        if commit: conn.commit()
        return result
    except sqlite3.Error as e: logger.error(f"DB ERROR: {e} for query: {query} with params: {params}"); return None
    finally:
        if conn: conn.close()

def create_new_session_db(session_id, name, timestamp):
    if db_execute("SELECT id FROM sessions WHERE id = ?", (session_id,), fetchone=True):
        update_session_timestamp_db(session_id, timestamp); return True
    try:
        db_execute("INSERT INTO sessions (id, name, created_at, last_updated_at) VALUES (?, ?, ?, ?)", (session_id, name, timestamp, timestamp), commit=True)
        session_media_path = os.path.join(MEDIA_STORAGE_DIR, session_id)
        if not os.path.exists(session_media_path): os.makedirs(session_media_path)
        logger.info(f"DB: Successfully created new session {session_id} - '{name}'")
        if GLOBAL_CHROMA_CLIENT:
            try:
                GLOBAL_CHROMA_CLIENT.get_or_create_collection(name=session_id, embedding_function=None)
                logger.info(f"ChromaDB collection ensured for session: {session_id}")
            except Exception as e_chroma_coll:
                logger.error(f"Failed to ensure ChromaDB collection for session {session_id}: {e_chroma_coll}")
        return True
    except Exception as e: logger.error(f"DB: Error creating session {session_id}: {e}"); return False

def update_session_timestamp_db(session_id, timestamp):
    db_execute("UPDATE sessions SET last_updated_at = ? WHERE id = ?", (timestamp, session_id), commit=True)

def get_all_sessions_db():
    return db_execute("SELECT id, name, created_at, last_updated_at FROM sessions ORDER BY last_updated_at DESC", fetchall=True) or []

async def save_message_db(msg_id, session_id, sender, timestamp, text_content=None, image_filename=None, data_type="text", llm_model_used=None, tts_audio_filename=None):
    if not db_execute("SELECT id FROM sessions WHERE id = ?", (session_id,), fetchone=True): return False
    try:
        db_execute("INSERT INTO messages (id, session_id, sender, timestamp, text_content, image_filename, data_type, llm_model_used, tts_audio_filename) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (msg_id, session_id, sender, timestamp, text_content, image_filename, data_type, llm_model_used, tts_audio_filename), commit=True)
        update_session_timestamp_db(session_id, timestamp)

        if GLOBAL_CHROMA_CLIENT and GLOBAL_EMBEDDING_MODEL and text_content:
            if data_type not in ["ai_image_description", "running_summary_debug"]: 
                try:
                    collection = GLOBAL_CHROMA_CLIENT.get_or_create_collection(name=session_id)
                    doc_to_embed = f"{sender}: {text_content}"
                    if image_filename and sender == "User": 
                        doc_to_embed += f" [Image filename: {image_filename}]"

                    loop = asyncio.get_event_loop()
                    embedding_array = await loop.run_in_executor(None, GLOBAL_EMBEDDING_MODEL.encode, doc_to_embed)
                    embedding = embedding_array.tolist()

                    func_to_run = functools.partial(
                        collection.add,
                        embeddings=[embedding],
                        documents=[doc_to_embed],
                        metadatas=[{"sender": sender, "timestamp": timestamp, "sqlite_msg_id": msg_id, "data_type": data_type}],
                        ids=[msg_id]
                    )
                    await loop.run_in_executor(None, func_to_run)
                    logger.debug(f"ChromaDB: Added message {msg_id} (type: {data_type}) to collection {session_id}")
                except Exception as e_chroma_add:
                    logger.error(f"ChromaDB: Error adding message {msg_id} to collection {session_id}: {e_chroma_add}", exc_info=True)
        return True
    except Exception as e:
        logger.error(f"DB: Error saving message to session {session_id}: {e}", exc_info=True)
        return False

def get_messages_for_session_db(session_id):
    rows = db_execute(
        "SELECT id, sender, timestamp, text_content, image_filename, data_type, llm_model_used, tts_audio_filename FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
        (session_id,),
        fetchall=True
    )
    messages_for_client = []
    if rows:
        for row_dict in rows:
            client_msg = {
                "id": row_dict["id"], "sender": row_dict["sender"], "timestamp": row_dict["timestamp"],
                "data": {"text": row_dict["text_content"]}, "image_filename": row_dict["image_filename"],
                "tts_audio_filename": row_dict["tts_audio_filename"], "data_type": row_dict["data_type"],
                "llm_model_used": row_dict["llm_model_used"]
            }
            if row_dict["image_filename"] and row_dict["sender"] == "User":
                client_msg["image_url"] = f"{MEDIA_STORAGE_DIR}/{session_id}/{row_dict['image_filename']}"
            messages_for_client.append(client_msg)
    logger.debug(f"GET_MESSAGES_DB: For session {session_id}, found {len(messages_for_client)} messages.")
    return messages_for_client

def rename_session_db(session_id, new_name, timestamp):
    try: db_execute("UPDATE sessions SET name = ?, last_updated_at = ? WHERE id = ?", (new_name, timestamp, session_id), commit=True); return True
    except Exception as e: logger.error(f"DB: Error renaming session {session_id}: {e}"); return False

def delete_session_db(session_id):
    try:
        session_media_path = os.path.join(MEDIA_STORAGE_DIR, session_id)
        if os.path.exists(session_media_path): shutil.rmtree(session_media_path)
        db_execute("DELETE FROM sessions WHERE id = ?", (session_id,), commit=True) 
        db_execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,), commit=True)
        if GLOBAL_CHROMA_CLIENT:
            try:
                collections = GLOBAL_CHROMA_CLIENT.list_collections()
                if any(c.name == session_id for c in collections):
                    GLOBAL_CHROMA_CLIENT.delete_collection(name=session_id)
                    logger.info(f"ChromaDB collection deleted for session: {session_id}")
                else:
                    logger.info(f"ChromaDB collection for session {session_id} not found, no deletion needed.")
            except Exception as e_chroma_del:
                logger.warning(f"ChromaDB: Error during delete_collection for {session_id}: {e_chroma_del}")
        return True
    except Exception as e: 
        logger.error(f"DB: Error deleting session {session_id}: {e}", exc_info=True)
        return False

def load_formatted_history_for_gemma_from_db(session_id: str, max_pairs: int):
    query = "SELECT sender, text_content, image_filename, data_type FROM messages WHERE session_id = ? AND text_content IS NOT NULL AND text_content != '' ORDER BY timestamp ASC"
    all_relevant_rows = db_execute(query, (session_id,), fetchall=True)
    
    raw_formatted_history = []
    if all_relevant_rows:
        for row_dict in all_relevant_rows:
            role = None
            content_list = []
            data_type = row_dict.get("data_type", "text")

            if row_dict["sender"] == "User":
                role = "user"
                if row_dict["image_filename"]: 
                    content_list.append({"type": "image_marker"})
                if row_dict["text_content"]:
                    content_list.append({"type": "text", "text": row_dict["text_content"]})
            elif row_dict["sender"] == "AI":
                # Include main diagnostic/clarification/error responses and explicit image descriptions by Gemma
                if data_type in ["ai_diagnostic_response", "ai_image_description", "ai_clarification_question", "ai_error", "ai_internal_error"]:
                    role = "assistant"
                    if row_dict["text_content"]:
                        text_for_history = row_dict["text_content"]
                        if data_type == "ai_image_description": # Mark image descriptions for potential special handling if needed
                             text_for_history = f"[Image Description by Assistant]: {row_dict['text_content']}"
                        content_list.append({"type": "text", "text": text_for_history})
            
            if role and content_list:
                raw_formatted_history.append({"role": role, "content": content_list})

    final_history_for_gemma_state = []
    if raw_formatted_history:
        temp_reversed_history = []
        last_assistant_idx = -1
        for i in range(len(raw_formatted_history) - 1, -1, -1):
            if raw_formatted_history[i]["role"] == "assistant":
                last_assistant_idx = i
                break
        
        if last_assistant_idx != -1:
            temp_reversed_history.append(raw_formatted_history[last_assistant_idx])
            expected_role = "user"
            current_pairs = 0
            for i in range(last_assistant_idx - 1, -1, -1):
                if current_pairs >= max_pairs: break
                current_msg = raw_formatted_history[i]
                if current_msg["role"] == expected_role:
                    temp_reversed_history.append(current_msg)
                    if expected_role == "user": expected_role = "assistant"
                    else: expected_role = "user"; current_pairs +=1
            final_history_for_gemma_state = list(reversed(temp_reversed_history))
            if final_history_for_gemma_state and final_history_for_gemma_state[0]["role"] == "assistant":
                final_history_for_gemma_state.pop(0)
        
    logger.debug(f"DB_HISTORY_LOAD (for Gemma state): Loaded {len(final_history_for_gemma_state)} items for session {session_id} after filtering and alternation.")
    return final_history_for_gemma_state


class AudioSegmentDetector: 
    def __init__(self, sample_rate=16000, energy_threshold=0.015, silence_duration=0.8, min_speech_duration=0.8, max_speech_duration=30, vad_initial_cooldown=1.0):
        self.sample_rate = sample_rate; self.energy_threshold = energy_threshold
        self.silence_samples = int(silence_duration * sample_rate); self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate); self.audio_buffer = bytearray()
        self.is_speech_active = False; self.silence_counter = 0; self.speech_start_idx = 0
        self.lock = asyncio.Lock(); self.segment_queue = asyncio.Queue(); self.segments_detected = 0
        self.tts_playing = False; self.tts_lock = asyncio.Lock()
        self.current_generation_task = None; self.current_tts_task = None; self.task_lock = asyncio.Lock()
        self.processing_lock = asyncio.Lock(); self.ignore_input_until = 0.0
        self.vad_process_cooldown = vad_initial_cooldown; self.id_for_log = str(id(self))[-6:]
    async def _is_tts_active(self):
        async with self.tts_lock: return self.tts_playing
    async def set_tts_playing(self, is_playing): 
        async with self.tts_lock: self.tts_playing = is_playing
        logger.debug(f"Detector {self.id_for_log}: TTS playing set to {is_playing}")
    async def cancel_current_tasks(self):
        async with self.task_lock:
            cancelled_something = False
            if self.current_generation_task and not self.current_generation_task.done(): 
                self.current_generation_task.cancel(); cancelled_something = True
                try: await self.current_generation_task
                except asyncio.CancelledError: logger.info(f"Detector {self.id_for_log}: Gen task cancelled.") 
                self.current_generation_task = None
            if self.current_tts_task and not self.current_tts_task.done(): 
                self.current_tts_task.cancel(); cancelled_something = True
                try: await self.current_tts_task
                except asyncio.CancelledError: logger.info(f"Detector {self.id_for_log}: TTS task cancelled.")
                self.current_tts_task = None
            if cancelled_something: logger.info(f"Detector {self.id_for_log}: Cancellation for active tasks.")
            await self.set_tts_playing(False) 
    async def set_current_tasks(self, gen_task=None, tts_task=None):
        async with self.task_lock: 
            if gen_task is not None: self.current_generation_task = gen_task
            if tts_task is not None: self.current_tts_task = tts_task
    async def set_vad_cooldown(self, duration=None):
        cooldown_duration = duration if duration is not None else self.vad_process_cooldown
        async with self.processing_lock: 
            self.ignore_input_until = time.time() + cooldown_duration
            logger.info(f"Detector {self.id_for_log}: VAD input cooldown for {cooldown_duration:.2f}s (until {self.ignore_input_until:.2f})")
    async def extend_vad_cooldown(self, required_until_time):
        async with self.processing_lock:
            if required_until_time > self.ignore_input_until: 
                self.ignore_input_until = required_until_time
                logger.info(f"Detector {self.id_for_log}: VAD output cooldown extended until {self.ignore_input_until:.2f}")
    async def clear_vad_cooldown(self):
        async with self.processing_lock:
            if self.ignore_input_until != 0.0: logger.info(f"Detector {self.id_for_log}: VAD cooldown cleared (was active until {self.ignore_input_until:.2f}).")
            self.ignore_input_until = 0.0
    async def add_audio(self, audio_bytes):
        if await self._is_tts_active(): logger.debug(f"Detector {self.id_for_log}: TTS is active, ignoring incoming audio chunk for VAD processing."); return None
        segment_to_queue = None; ignore_this_chunk = False
        async with self.processing_lock: ignore_until = self.ignore_input_until
        if time.time() < ignore_until: ignore_this_chunk = True
        if ignore_this_chunk: logger.debug(f"Detector {self.id_for_log}: VAD cooldown active, ignoring chunk."); return None
        async with self.lock:
            self.audio_buffer.extend(audio_bytes); current_buffer_len = len(self.audio_buffer)
            analysis_chunk_samples = int(0.05 * self.sample_rate); analysis_chunk_bytes = analysis_chunk_samples * 2
            start_index = max(0, current_buffer_len - analysis_chunk_bytes); recent_audio_bytes = bytes(self.audio_buffer[start_index:])
            if len(recent_audio_bytes) >= 2:
                audio_array = np.frombuffer(recent_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                energy = 0.0
                if len(audio_array) > 0: energy = np.sqrt(np.mean(audio_array**2))
                if not self.is_speech_active:
                    if energy > self.energy_threshold:
                        self.is_speech_active = True; buffer_offset = int(0.1 * self.sample_rate * 2)
                        self.speech_start_idx = max(0, current_buffer_len - len(recent_audio_bytes) - buffer_offset); self.silence_counter = 0
                        logger.info(f"Detector {self.id_for_log}: Speech start (energy: {energy:.6f}) at index {self.speech_start_idx}")
                        is_tts_playing_now_for_interrupt_check = False; 
                        async with self.tts_lock: is_tts_playing_now_for_interrupt_check = self.tts_playing
                        if is_tts_playing_now_for_interrupt_check: logger.info(f"Detector {self.id_for_log}: User speech during TTS (interrupt)! Triggering Interrupt!"); asyncio.create_task(self.cancel_current_tasks()); asyncio.create_task(self.clear_vad_cooldown())
                else: 
                    current_speech_len_bytes = current_buffer_len - self.speech_start_idx; current_speech_len_samples = current_speech_len_bytes // 2
                    if energy > self.energy_threshold: self.silence_counter = 0
                    else: self.silence_counter += analysis_chunk_samples
                    if self.silence_counter >= self.silence_samples:
                        speech_end_idx_bytes = current_buffer_len - (self.silence_counter * 2); speech_end_idx_bytes = max(self.speech_start_idx, speech_end_idx_bytes)
                        segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx_bytes]); segment_len_samples = len(segment_bytes) // 2
                        logger.info(f"Detector {self.id_for_log}: Silence met ({self.silence_counter}/{self.silence_samples} samples)")
                        self.is_speech_active = False; self.silence_counter = 0; self.audio_buffer = self.audio_buffer[speech_end_idx_bytes:]; self.speech_start_idx = 0
                        if segment_len_samples >= self.min_speech_samples: segment_to_queue = segment_bytes; logger.info(f"Detector {self.id_for_log}: Speech segment (Silence End): {segment_len_samples / self.sample_rate:.2f}s")
                        else: logger.info(f"Detector {self.id_for_log}: Discarding short segment (Silence End): {segment_len_samples / self.sample_rate:.2f}s")
                    elif current_speech_len_samples >= self.max_speech_samples: 
                        speech_end_idx_bytes = self.speech_start_idx + (self.max_speech_samples * 2)
                        segment_bytes = bytes(self.audio_buffer[self.speech_start_idx:speech_end_idx_bytes]); segment_len_samples = len(segment_bytes) // 2
                        logger.info(f"Detector {self.id_for_log}: Max duration met ({current_speech_len_samples}/{self.max_speech_samples} samples)")
                        segment_to_queue = segment_bytes; logger.info(f"Detector {self.id_for_log}: Speech segment (Max Duration): {segment_len_samples / self.sample_rate:.2f}s")
                        self.audio_buffer = self.audio_buffer[speech_end_idx_bytes:]; self.speech_start_idx = 0; self.silence_counter = 0
        if segment_to_queue: 
            self.segments_detected += 1
            await self.segment_queue.put(segment_to_queue)
        async with self.lock:
            if not self.is_speech_active and len(self.audio_buffer) > self.max_speech_samples * 5 * 2: 
                keep_bytes = self.max_speech_samples * 2; trim_point = len(self.audio_buffer) - keep_bytes
                self.audio_buffer = self.audio_buffer[trim_point:]; self.speech_start_idx = 0
                logger.info(f"Detector {self.id_for_log}: Trimmed long inactive audio buffer to {len(self.audio_buffer)} bytes.")
        return None
    async def get_next_segment(self): 
        try: return await asyncio.wait_for(self.segment_queue.get(), timeout=0.1)
        except asyncio.TimeoutError: return None

class WhisperTranscriber: # Unchanged
    _instance = None
    @classmethod
    def get_instance(cls, model=None, processor=None):
        if cls._instance is None:
            if model and processor: cls._instance = cls(model=model, processor=processor, load_new=False)
            else: cls._instance = cls(load_new=True)
        return cls._instance
    def __init__(self, model=None, processor=None, load_new=True):
        if load_new:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"; logger.info(f"Whisper (NEW INSTANCE): Using device: {self.device}")
            compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()) else torch.float16 if torch.cuda.is_available() else torch.float32
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype)
            model_id = "openai/whisper-small"; 
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, quantization_config=quant_config, low_cpu_mem_usage=True, use_safetensors=True)
            self.processor = AutoProcessor.from_pretrained(model_id)
        else: self.model = model; self.processor = processor; logger.info(f"Whisper: Using pre-loaded model.")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            stride_length_s=[6,0]
        )
        logger.info(f"Whisper: ASR pipeline ready."); self.transcription_count = 0
    async def transcribe(self, audio_bytes, sample_rate=16000):
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_array) < 1000: return ""
            result = await asyncio.get_event_loop().run_in_executor(None, lambda: self.pipe( {"array": audio_array, "sampling_rate": sample_rate}, generate_kwargs={"task": "transcribe", "language": "english", "temperature": 0.0}))
            text = result.get("text", "").strip() if result else ""
            self.transcription_count += 1; logger.info(f"Whisper Tx {self.transcription_count}: '{text}'")
            return text
        except Exception as e: logger.exception(f"Whisper tx error: {e}"); return ""

class StopOnTokens(StoppingCriteria): # Unchanged
    def __init__(self, stop_ids_list): 
        super().__init__(); self.stop_sequences = []
        for item in stop_ids_list:
            if isinstance(item, list) and all(isinstance(i, int) for i in item): self.stop_sequences.append(torch.tensor(item, dtype=torch.long))
            elif isinstance(item, int): self.stop_sequences.append(torch.tensor([item], dtype=torch.long))
        logger.info(f"StopOnTokens Initialized. Will stop on: {[seq.tolist() for seq in self.stop_sequences]}")
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        current_sequence = input_ids[0] 
        for stop_seq_tensor in self.stop_sequences:
            if len(current_sequence) >= len(stop_seq_tensor):
                if torch.equal(current_sequence[-len(stop_seq_tensor):], stop_seq_tensor.to(current_sequence.device)):
                    logger.debug(f"Stopping criteria met: matches {stop_seq_tensor.tolist()}"); return True
        return False

class KokoroTTSProcessor: # Unchanged
    _instance = None
    @classmethod
    def get_instance(cls, pipeline_instance=None):
        if cls._instance is None:
            if pipeline_instance: cls._instance = cls(pipeline_instance=pipeline_instance, load_new=False)
            else: cls._instance = cls(load_new=True)
        return cls._instance
    def __init__(self, pipeline_instance=None, load_new=True):
        if load_new:
            try: self.pipeline = KPipeline(lang_code='a'); logger.info("Kokoro: KPipeline loaded.")
            except Exception as e: logger.exception(f"FATAL: Kokoro KPipeline init error: {e}"); self.pipeline = None; raise
        else: self.pipeline = pipeline_instance; logger.info("Kokoro: Using pre-loaded KPipeline.")
        if self.pipeline: self.target_sr, self.default_voice, self.synthesis_count = 24000, 'af_sarah', 0; logger.info(f"Kokoro TTS ready. SR:{self.target_sr}, Voice:{self.default_voice}")
        else: raise RuntimeError("Kokoro pipeline failed to initialize")
    async def synthesize_initial_speech(self, text):
        if not text or not self.pipeline: return None
        try:
            logger.info(f"TTS: Synthesizing for text: '{text[:80]}...'")
            audio_segments_np = []
            def sync_synthesize():
                processed_segments = []
                for _gs, _ps, audio_chunk_data in self.pipeline(text, voice=self.default_voice, speed=1.0, split_pattern=None): 
                    if audio_chunk_data is None: continue
                    current_audio_np = audio_chunk_data.cpu().numpy() if isinstance(audio_chunk_data, torch.Tensor) else (audio_chunk_data if isinstance(audio_chunk_data, np.ndarray) else None)
                    if current_audio_np is None: logger.error(f"TTS: Kokoro unexpected type: {type(audio_chunk_data)}"); continue
                    if current_audio_np.size > 0:
                        if current_audio_np.dtype != np.float32: current_audio_np = current_audio_np.astype(np.float32)
                        processed_segments.append(current_audio_np)
                return processed_segments
            audio_segments_np = await asyncio.get_event_loop().run_in_executor(None, sync_synthesize)
            if not audio_segments_np: return None
            combined_audio = np.concatenate(audio_segments_np); self.synthesis_count += 1
            logger.info(f"TTS Synthesis {self.synthesis_count} complete: {len(combined_audio)/self.target_sr:.2f}s")
            return combined_audio
        except Exception as e: logger.exception(f"TTS synthesis error: {e}"); return None

class SummarizationProcessor:
    def __init__(self, model, tokenizer): # Assuming model is Qwen, tokenizer is Qwen's tokenizer
        self.model = model
        self.tokenizer = tokenizer
        self.device = self.model.device 
        self.summarization_count = 0
        if self.tokenizer.pad_token_id is None:
            # Qwen models usually have an eos_token_id. Some might use a specific pad_token.
            # If tokenizer.eos_token_id is a list (like for Gemma), pick one.
            # For Qwen, it's typically a single int or None.
            eos_id = self.tokenizer.eos_token_id
            if isinstance(eos_id, list) and eos_id: # Handle if it's a list
                eos_id = eos_id[0]
            
            self.tokenizer.pad_token_id = eos_id if eos_id is not None else self.tokenizer.vocab_size -1 # A common fallback for pad_token
            logger.info(f"SummarizationProcessor: Summarizer pad_token_id set to {self.tokenizer.pad_token_id}.")
    
    async def summarize_text(self, text_to_summarize: str, current_summary: str = None, max_summary_length=300, min_summary_length=50) -> str:
        self.summarization_count += 1
        log_prefix = f"Summarizer Call {self.summarization_count} (Qwen):"
        
        # Using the new SUMMARIZER_SYSTEM_PROMPT
        if current_summary: # current_summary is the previous template-filled summary
            prompt = f"{SUMMARIZER_SYSTEM_PROMPT}\n\n<Existing Summary>\n{current_summary}\n</Existing Summary>\n\n<New Turns>\n{text_to_summarize}\n</New Turns>\n\nUpdated Summary (using the template):"
        else:
            prompt = f"{SUMMARIZER_SYSTEM_PROMPT}\n\n<Conversation Turns>\n{text_to_summarize}\n</Conversation Turns>\n\nNew Summary (using the template):"
        
        logger.info(f"{log_prefix} Summarizing. Input text for SmolLM (first 100 chars of new turns): '{text_to_summarize[:100]}...' Current summary provided: {bool(current_summary)}")
        logger.debug(f"{log_prefix} Full prompt for SmolLM: {prompt[:500]}...") # Log more of the prompt

        async with GLOBAL_LLM_INFERENCE_LOCK:
            logger.debug(f"{log_prefix} Acquired LLM inference lock for SmolLM.")
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device) # Max length for Qwen input
                
                summary_ids = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.generate(
                        **inputs,
                        max_new_tokens=max_summary_length,
                        min_new_tokens=min_summary_length, # Ensure it generates something meaningful
                        num_beams=3, # Increased beams slightly
                        early_stopping=True,
                        no_repeat_ngram_size=3, # Stricter no_repeat
                        temperature=0.6, # More deterministic
                        do_sample=True, # Still allow some sampling
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                )
                summary_text = self.tokenizer.decode(summary_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                if "---" in summary_text: # If template ending "---" is still there
                    summary_text = summary_text.split("---")[0].strip()
                if "Updated Summary:" in summary_text: # If it repeats part of the prompt
                    summary_text = summary_text.split("Updated Summary:")[-1].strip()
                if "New Summary:" in summary_text:
                    summary_text = summary_text.split("New Summary:")[-1].strip()
                logger.info(f"{log_prefix} Generated SmolLM summary (full): '{summary_text}'") # Log full summary
                return summary_text
            except Exception as e:
                logger.error(f"{log_prefix} Error during Qwen summarization: {e}", exc_info=True)
                return ""
            finally:
                logger.debug(f"{log_prefix} Released LLM inference lock after Qwen.")

# GemmaMultimodalProcessor, process_user_input_and_respond, update_session_summary, handle_client, main
# These were pasted in the previous response.
# Ensure all `await GLOBAL_LLM_INFERENCE_LOCK` are in place for Gemma's `generate` calls.
# Ensure `_update_history_with_complete_response` and `_build_messages_for_main_generation` are the LATEST versions.
# Ensure `process_user_input_and_respond` calls the modified `_update_history_with_complete_response` correctly.
# Ensure `main` loads SmolLM.

# The classes and functions below are pasted from the previous response, including their latest fixes.
# Double check against your file that these are the versions you intend.

class GemmaMultimodalProcessor:
    class EmptyStreamer:
        def __iter__(self): return self
        def __next__(self): raise StopIteration

    def __init__(self, shared_model, shared_processor, shared_hf_tokenizer):
        self.instance_id_log = str(id(self))[-6:]
        self.model = shared_model
        self.processor = shared_processor
        self.hf_tokenizer = shared_hf_tokenizer
        self.device = self.model.device
        logger.info(f"GemmaProc Inst:{self.instance_id_log} initialized with SHARED model on device: {self.device}")
        # Ensure GLOBAL_GEMMA_MODEL.generation_config exists and is valid
        if hasattr(GLOBAL_GEMMA_MODEL, 'generation_config') and GLOBAL_GEMMA_MODEL.generation_config is not None:
            self.generation_config = GenerationConfig.from_dict(GLOBAL_GEMMA_MODEL.generation_config.to_dict())
        else:
            logger.warning(f"GemmaProc Inst:{self.instance_id_log} GLOBAL_GEMMA_MODEL.generation_config missing. Using default.")
            self.generation_config = GenerationConfig(eos_token_id=self.hf_tokenizer.eos_token_id, pad_token_id=self.hf_tokenizer.pad_token_id or self.hf_tokenizer.eos_token_id)

        logger.info(f"GemmaProc Inst:{self.instance_id_log} Using G.CFG (copied or default): {str(self.generation_config).replace(chr(10), ' ')}")
        self.session_states = {}
        self.max_history_pairs = 3 
        self.generation_count = 0
        stop_sequences_strings = ["<end_of_turn>\n", "<end_of_turn>"]
        self.stop_token_ids_list = []
        if self.hf_tokenizer:
            for stop_str in stop_sequences_strings:
                try:
                    token_ids = self.hf_tokenizer.encode(stop_str, add_special_tokens=False)
                    if token_ids:
                        if isinstance(token_ids, int): token_ids = [token_ids]
                        if token_ids not in self.stop_token_ids_list : self.stop_token_ids_list.append(token_ids); logger.info(f"GemmaProc Inst:{self.instance_id_log} Registered stop sequence: '{stop_str}' -> Tokens: {token_ids}")
                except Exception as e_tok_encode:
                    logger.error(f"GemmaProc Inst:{self.instance_id_log} Error encoding stop string '{stop_str}': {e_tok_encode}")
            
            configured_eos_ids = getattr(self.generation_config, "eos_token_id", None)
            eos_ids_to_check = []
            if isinstance(configured_eos_ids, int): eos_ids_to_check.append([configured_eos_ids])
            elif isinstance(configured_eos_ids, list):
                 for eid in configured_eos_ids:
                     if isinstance(eid, int): eos_ids_to_check.append([eid])
            for eos_list in eos_ids_to_check:
                 if eos_list not in self.stop_token_ids_list: self.stop_token_ids_list.append(eos_list); logger.info(f"GemmaProc Inst:{self.instance_id_log} Added G.CFG EOS to custom stop: {eos_list}")
            
            if self.stop_token_ids_list: self.stopping_criteria = StoppingCriteriaList([StopOnTokens(self.stop_token_ids_list)]); logger.info(f"GemmaProc Inst:{self.instance_id_log} Custom stopping criteria INITIALIZED with: {self.stop_token_ids_list}")
            else: self.stopping_criteria = None; logger.warning(f"GemmaProc Inst:{self.instance_id_log} No valid stop sequences for custom stopping criteria.")
        else:
            self.stopping_criteria = None; logger.error(f"GemmaProc Inst:{self.instance_id_log} Tokenizer not available for stopping criteria init.")

    def _get_or_create_session_state(self, client_session_id: str):
        if client_session_id not in self.session_states:
            logger.info(f"GemmaProc Inst:{self.instance_id_log} Creating new state for session_id: {client_session_id}")
            loaded_history_for_gemma_state = load_formatted_history_for_gemma_from_db(client_session_id, self.max_history_pairs)
            summary_row = db_execute("SELECT summary_text FROM session_summaries WHERE session_id = ?", (client_session_id,), fetchone=True)
            current_running_summary = summary_row["summary_text"] if summary_row else ""
            self.session_states[client_session_id] = {
                "message_history": loaded_history_for_gemma_state,
                "last_image_pil_for_description": None, # Store the PIL object
                "last_image_user_query_context": None, # Store user text that came with image
                "pending_image_description": None, # Store the generated description
                "messages_since_last_summary": 0, 
                "current_running_summary": current_running_summary 
            }
            logger.info(f"GemmaProc Inst:{self.instance_id_log} Initialized session {client_session_id} with {len(loaded_history_for_gemma_state)} history items. Initial summary length: {len(current_running_summary)}")
        return self.session_states[client_session_id]

    async def set_image(self, image_data_bytes, client_session_id: str, user_query_for_context: str): # Ensure this line is correctly indented under the class
        # This line should be indented correctly as part of the method
        session_state = self._get_or_create_session_state(client_session_id) 
        try:
            image = Image.open(io.BytesIO(image_data_bytes))
            session_state["last_image_pil_for_description"] = image 
            session_state["last_image_user_query_context"] = user_query_for_context
            session_state["pending_image_description"] = None # Clear any old pending one
            logger.info(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} Image PIL and query context buffered, awaiting description generation trigger.")
            return True 
        except Exception as e: 
            logger.error(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} Error processing image for buffering: {e}")
            session_state["last_image_pil_for_description"] = None
            session_state["last_image_user_query_context"] = None
            return False

    def _get_short_term_memory(self, client_session_id: str):
        # This fetches already processed history suitable for direct inclusion before the current user turn.
        # It relies on load_formatted_history_for_gemma_from_db to ensure alternation.
        history = load_formatted_history_for_gemma_from_db(client_session_id, self.max_history_pairs)
        logger.debug(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} _get_short_term_memory: returning {len(history)} items.")
        return history

    async def _retrieve_rag_context(self, query_text: str, client_session_id: str) -> str:
        # ... (Unchanged from your latest working version with functools.partial) ...
        if not GLOBAL_CHROMA_CLIENT or not GLOBAL_EMBEDDING_MODEL:
            logger.warning(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} ChromaDB or Embedding model not available for RAG.")
            return ""
        rag_parts = []
        session_state = self._get_or_create_session_state(client_session_id)
        current_running_summary = session_state.get("current_running_summary", "")
        if current_running_summary:
            rag_parts.append(f"---\nConversation Summary So Far:\n{current_running_summary}\n---")
        try:
            collection = GLOBAL_CHROMA_CLIENT.get_or_create_collection(name=client_session_id)
            loop = asyncio.get_event_loop()
            if await loop.run_in_executor(None, collection.count) > 0 :
                embedding_array = await loop.run_in_executor(None, GLOBAL_EMBEDDING_MODEL.encode, query_text)
                query_embedding = embedding_array.tolist()
                current_collection_count = await loop.run_in_executor(None, collection.count)
                n_results_to_fetch = RAG_MAX_RESULTS + 5 
                actual_n_results = min(n_results_to_fetch , current_collection_count)
                if actual_n_results > 0:
                    query_func = functools.partial(
                        collection.query,
                        query_embeddings=[query_embedding], n_results=actual_n_results,
                        include=["documents", "metadatas"]
                    )
                    results = await loop.run_in_executor(None, query_func)
                    retrieved_docs_text = []
                    if results and results.get('documents') and results['documents'][0]:
                        for i, doc_text in enumerate(results['documents'][0]):
                            if len(retrieved_docs_text) >= RAG_MAX_RESULTS: break
                            meta = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] and len(results['metadatas'][0]) > i else {}
                            if meta.get("type") not in ["running_summary", "image_description_embedding"]:
                                retrieved_docs_text.append(doc_text)
                        if retrieved_docs_text:
                           rag_parts.append("---\nRelevant Past Conversation Snippets (for context):\n" + "\n\n".join(retrieved_docs_text) + "\n---")
                           logger.info(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} RAG: Retrieved {len(retrieved_docs_text)} snippets for RAG context.")
                    else: logger.info(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} RAG: No relevant message documents found in Chroma query results.")
            if not rag_parts: logger.info(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} RAG: No context generated."); return ""
            final_rag_context = "\n\n".join(rag_parts)
            logger.debug(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} RAG Context Preview: {final_rag_context[:500]}...")
            return final_rag_context
        except Exception as e_rag:
            logger.error(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} RAG retrieval error: {e_rag}", exc_info=True)
            if current_running_summary: return f"---\nConversation Summary So Far:\n{current_running_summary}\n---"
            return ""

    def _update_history_with_complete_response(
        self, 
        client_session_id: str,
        user_prompt_text: str, 
        assistant_response_text: str, 
        user_had_image: bool, 
        is_assistant_image_description: bool = False, # <<<< RENAMED HERE
        is_assistant_clarification_turn: bool = False 
    ):
        session_state = self._get_or_create_session_state(client_session_id)
        logger.debug(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} _update_history_WITH_EXCHANGE: user_text='{str(user_prompt_text)[:30]}...', assist_text='{str(assistant_response_text)[:30]}...', user_had_img={user_had_image}, is_assist_img_desc={is_assistant_image_description}, is_clarify_q={is_assistant_clarification_turn}") # Use new name in log

        # --- Handle User Turn ---
        add_this_user_turn = True
        # If this call is specifically to add an AI's image description, the "user turn" is the original user message that provided the image.
        # That user message should have been added to history already when it was first processed.
        # So, for an AI image description turn, we typically only add the assistant part.
        if is_assistant_image_description: # if assistant_response_text is an image description
            add_this_user_turn = False # Don't add a new "user" turn for the image description step.
                                     # The original user turn (with image_marker) is already in history.
            logger.debug(f"GemmaProc Inst:{self.instance_id_log} _update_history: This is an AI image description turn; user part already logged or implicit.")


        if add_this_user_turn: # Only add user turn if it's a genuine user message for this exchange point
            user_turn_content_for_gemma_state = []
            if user_had_image: 
                user_turn_content_for_gemma_state.append({"type": "image_marker"})
            
            user_prompt_text_stripped = str(user_prompt_text).strip() if user_prompt_text else ""
            if user_prompt_text_stripped: 
                user_turn_content_for_gemma_state.append({"type": "text", "text": user_prompt_text_stripped})
            elif user_had_image and not user_prompt_text_stripped: 
                user_turn_content_for_gemma_state.append({"type": "text", "text": "[User sent an image this turn]"})
            
            if user_turn_content_for_gemma_state:
                if not (session_state["message_history"] and \
                        session_state["message_history"][-1]["role"] == "user" and \
                        session_state["message_history"][-1]["content"] == user_turn_content_for_gemma_state):
                    if session_state["message_history"] and session_state["message_history"][-1]["role"] == "user":
                        logger.warning(f"GemmaProc Inst:{self.instance_id_log} _update_history: Popping last user message to ensure alternation before adding new user turn. This might indicate a logic flow issue if unexpected.")
                        session_state["message_history"].pop()
                    session_state["message_history"].append({"role": "user", "content": user_turn_content_for_gemma_state})
                    logger.debug(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} _update_history: Added user turn to Gemma history.")
                else:
                    logger.debug(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} _update_history: Skipped adding identical user message to Gemma history.")
        
        # --- Handle Assistant Turn ---
        normalized_assistant_response = assistant_response_text
        error_prefixes = ("[Critical error during Gemma processing:", "Error: Gemma prompt construction failed", "[Error:")
        if any(str(assistant_response_text).startswith(prefix) for prefix in error_prefixes):
            normalized_assistant_response = "[AI failed to generate a response]"
        
        assistant_response_stripped = str(normalized_assistant_response).strip() if normalized_assistant_response else ""
        if assistant_response_stripped:
             assistant_content_type_hint = "text"
             if is_assistant_image_description: assistant_content_type_hint = "ai_image_description"
             elif is_assistant_clarification_turn: assistant_content_type_hint = "ai_clarification_question"
            
             # Ensure last turn was user before adding assistant, unless history is empty (first turn for AI)
             if not session_state["message_history"] or session_state["message_history"][-1]["role"] == "user":
                session_state["message_history"].append({"role": "assistant", "content": [{"type": "text", "text": assistant_response_stripped, "data_type_hint": assistant_content_type_hint }]})
                logger.debug(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} _update_history: Added assistant turn ({assistant_content_type_hint}).")
             else: # Attempting to add assistant after assistant - indicates a flow issue.
                logger.warning(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} _update_history: Last history item was not 'user' when trying to add assistant turn. Current AI content type: {assistant_content_type_hint}. History: {[m['role'] for m in session_state['message_history'][-2:]]}")
                # As a recovery, if the last was assistant and this is NOT an image description (which should follow user),
                # it might be better to replace the last assistant message if it was just a placeholder or error.
                # For now, just log the warning. The history builder for LLM needs to be robust.
        else: 
            logger.warning(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} _update_history: Assistant response (normalized) empty, not adding to Gemma history.")
        
        max_history_items = self.max_history_pairs * 2 + 4 
        if len(session_state["message_history"]) > max_history_items:
            session_state["message_history"] = session_state["message_history"][-max_history_items:]
        logger.debug(f"GemmaProc Inst:{self.instance_id_log} SESS:{client_session_id} _update_history: EXIT. History length: {len(session_state['message_history'])}. Last roles: {[m['role'] for m in session_state['message_history'][-3:]]}")


    async def describe_image(self, image_pil: Image.Image, client_session_id: str, original_user_query: str) -> str:
        self.generation_count += 1
        log_prefix = f"GemmaProc Inst:{self.instance_id_log} DescribeImage Call {self.generation_count} SESS:{client_session_id}:"
        logger.info(f"{log_prefix} Generating description for image. User query context: '{original_user_query[:100]}...'")
        
        prompt_text = GEMMA_IMAGE_DESCRIBER_PROMPT_TEMPLATE.format(user_query=original_user_query)
        messages_for_desc = [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image", "image": image_pil}]}]
        
        desc_gen_config = GenerationConfig(
            max_new_tokens=256, temperature=0.2, top_p=0.9, do_sample=True,     
            pad_token_id=self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else self.processor.tokenizer.eos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id
        )
        description_text = "[Error: Could not generate image description]"
        async with GLOBAL_LLM_INFERENCE_LOCK:
            logger.debug(f"{log_prefix} Acquired LLM inference lock for image description.")
            try:
                inputs = self.processor.apply_chat_template(messages_for_desc, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.device)
                generation_args_desc = {"input_ids": inputs["input_ids"], "attention_mask": inputs.get("attention_mask"), "generation_config": desc_gen_config}
                if "pixel_values" in inputs and inputs["pixel_values"] is not None: generation_args_desc["pixel_values"] = inputs["pixel_values"]
                
                loop = asyncio.get_event_loop(); DESC_TIMEOUT = 500
                
                generated_ids = await asyncio.wait_for(loop.run_in_executor(None, lambda: self.model.generate(**generation_args_desc)), timeout=DESC_TIMEOUT)
                
                input_ids_length = inputs["input_ids"].shape[1]
                newly_generated_ids_tensor = torch.tensor([], device=generated_ids.device, dtype=generated_ids.dtype)
                if generated_ids.shape[1] > input_ids_length: newly_generated_ids_tensor = generated_ids[0][input_ids_length:]
                description_text = self.processor.tokenizer.decode(newly_generated_ids_tensor, skip_special_tokens=True).strip()
                if not description_text: description_text = "[Warning: Image description was empty]"
                logger.info(f"{log_prefix} Generated image description (Gemma): {description_text[:150]}...")
            except asyncio.TimeoutError: logger.error(f"{log_prefix} Image description generation timed out."); description_text = "[Error: Image description generation timed out]"
            except Exception as e: logger.exception(f"{log_prefix} Error during image description: {e}"); description_text = "[Error: Could not generate image description due to exception]"
            finally: logger.debug(f"{log_prefix} Released LLM inference lock after image description.")
        return description_text

    async def generate_clarification_questions(self, original_user_query: str, image_description: str, client_session_id: str) -> str:
        self.generation_count += 1
        log_prefix = f"GemmaProc Inst:{self.instance_id_log} ClarifyQuestion Call {self.generation_count} SESS:{client_session_id}:"
        logger.info(f"{log_prefix} Generating clarification questions. User Query: '{original_user_query[:50]}...', Image Desc: '{image_description[:50]}...'")

        prompt_text = IMAGE_CLARIFICATION_GUIDANCE_PROMPT_TEMPLATE.format(user_query=original_user_query, image_description=image_description)
        
        # CORRECTED: Content must be a list of dicts
        messages_for_clarify = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]

        clarify_gen_config = GenerationConfig(
            max_new_tokens=150, temperature=0.5, top_p=0.9, do_sample=True,
            pad_token_id=self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else self.processor.tokenizer.eos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id
        )
        
        clarification_text = "[Error: Could not generate clarification questions]"
        async with GLOBAL_LLM_INFERENCE_LOCK:
            logger.debug(f"{log_prefix} Acquired LLM inference lock for clarification questions.")
            try:
                inputs = self.processor.apply_chat_template(
                    messages_for_clarify, add_generation_prompt=True, 
                    tokenize=True, return_dict=True, return_tensors="pt"
                ).to(self.device)
                generation_args_clarify = {"input_ids": inputs["input_ids"], "attention_mask": inputs.get("attention_mask"), "generation_config": clarify_gen_config}
                
                loop = asyncio.get_event_loop(); CLARIFY_TIMEOUT = 500
                generated_ids = await asyncio.wait_for(loop.run_in_executor(None, lambda: self.model.generate(**generation_args_clarify)), timeout=CLARIFY_TIMEOUT)
                
                input_ids_length = inputs["input_ids"].shape[1]
                newly_generated_ids_tensor = torch.tensor([], device=generated_ids.device, dtype=generated_ids.dtype)
                if generated_ids.shape[1] > input_ids_length: newly_generated_ids_tensor = generated_ids[0][input_ids_length:]
                clarification_text = self.processor.tokenizer.decode(newly_generated_ids_tensor, skip_special_tokens=True).strip()
                if not clarification_text: clarification_text = "[Warning: Clarification questions generation returned empty]"
                logger.info(f"{log_prefix} Generated clarification questions (Gemma): {clarification_text[:150]}...")
            except asyncio.TimeoutError:
                logger.error(f"{log_prefix} Clarification questions generation timed out.")
                clarification_text = "[Error: Clarification questions generation timed out]"
            except Exception as e:
                logger.exception(f"{log_prefix} Error during clarification questions generation: {e}")
                clarification_text = "[Error: Could not generate clarification questions due to exception]"
            finally:
                logger.debug(f"{log_prefix} Released LLM inference lock after clarification questions.")
        return clarification_text

    async def add_image_summary_to_vector_db(self, session_id: str, original_user_msg_id: str, image_description_text: str): # Unchanged
        if not GLOBAL_CHROMA_CLIENT or not GLOBAL_EMBEDDING_MODEL or not image_description_text:
            return
        try:
            collection = GLOBAL_CHROMA_CLIENT.get_or_create_collection(name=session_id)
            summary_doc_id = f"img_desc_for_{original_user_msg_id}" 
            doc_to_embed = f"Visual description of image provided with user message {original_user_msg_id}: {image_description_text}"
            loop = asyncio.get_event_loop()
            embedding_array = await loop.run_in_executor(None, GLOBAL_EMBEDDING_MODEL.encode, doc_to_embed)
            embedding = embedding_array.tolist()
            func_to_run = functools.partial(
                collection.upsert, 
                ids=[summary_doc_id], embeddings=[embedding], documents=[doc_to_embed],
                metadatas=[{"type": "image_description_embedding", "original_msg_id": original_user_msg_id, "timestamp": int(time.time() * 1000)}]
            )
            await loop.run_in_executor(None, func_to_run)
            logger.info(f"ChromaDB: Added/Updated image description embedding '{summary_doc_id}' to collection '{session_id}'")
        except Exception as e:
            logger.error(f"ChromaDB: Error adding image description embedding for msg '{original_user_msg_id}' to collection '{session_id}': {e}", exc_info=True)
            
    async def generate_streaming(self, user_text_for_db: str, client_session_id: str, image_description: str = None): # user_text_for_db is the original user text for THIS turn
        try:
            t_start_generate_streaming = time.time()
            self.generation_count +=1
            log_prefix = f"GemmaProc Inst:{self.instance_id_log} GenerateStream Call {self.generation_count} SESS:{client_session_id}:"
            logger.info(f"{log_prefix} User text for this turn: '{user_text_for_db[:100]}...' Image Desc (for this turn's image, if any): {image_description is not None}")

            session_state = self._get_or_create_session_state(client_session_id)
            # current_image_pil_for_main_gen is the PIL object if an image was uploaded *as part of this specific interaction/call*
            # This is typically set in process_user_input_and_respond if image_data_b64 was present
            current_image_pil_for_main_gen = session_state.get("last_image_pil_for_description") # Using the buffered image if this is part of image flow
                                                                                            # Or will be None if this is a text-only follow-up

            # --- Assemble the prompt for Gemma ---
            messages_for_processor_final = []

            # 1. Get prior conversational history (alternating user/assistant turns)
            # This history should NOT include the current user_text_for_db, as that's part of the "current user turn"
            prior_history = self._get_short_term_memory(client_session_id) 
            messages_for_processor_final.extend(prior_history)
            logger.debug(f"{log_prefix} Extended with {len(prior_history)} prior history messages.")

            # 2. Construct the initial context block: System Prompt + RAG
            initial_context_block = REVISED_SIMPLIFIED_SYSTEM_PROMPT 
            # RAG based on the original user text for this turn to find relevant past snippets/summary
            rag_context_str = await self._retrieve_rag_context(user_text_for_db, client_session_id) 
            if rag_context_str:
                initial_context_block += "\n\n--- Relevant Context From Past Conversation ---\n" + rag_context_str + "\n--- End Context ---"
        
            # 3. Construct the current user's turn content for the LLM
            current_user_turn_llm_formatted_content = []

            # Add PIL image if it's part of this logical turn (e.g., user uploaded image and this call is to generate diagnostic)
            if current_image_pil_for_main_gen: 
                current_user_turn_llm_formatted_content.append({"type": "image", "image": current_image_pil_for_main_gen})
                logger.debug(f"{log_prefix} Added PIL image to current user turn content.")
        
            # Text part for current user turn (System Instructions, RAG, Image Description, Actual User Query)
            current_turn_user_text_parts = []
            current_turn_user_text_parts.append(initial_context_block) # System Prompt and RAG first

            if image_description: # This is the AI-generated description of an image uploaded in THIS logical turn
                current_turn_user_text_parts.append(f"\n\nImage Description (for current image):\n{image_description}")
        
            user_text_stripped = user_text_for_db.strip() if user_text_for_db else ""
            if user_text_stripped:
                current_turn_user_text_parts.append(f"\n\nUser's Current Query:\n{user_text_stripped}")
            # If no actual user text but there was an image and description, the prompt already guides AI.
            # If no image, no desc, no user text (empty input), add a placeholder:
            elif not image_description and not current_image_pil_for_main_gen and not user_text_stripped:
                current_turn_user_text_parts.append("\n\nUser's Current Query: (No text provided, please respond based on history and context)")

            full_user_text_for_this_turn = "\n".join(current_turn_user_text_parts)
            if full_user_text_for_this_turn: # Should always be true due to initial_context_block
                current_user_turn_llm_formatted_content.append({"type": "text", "text": full_user_text_for_this_turn})
        
            # Add the fully constructed current user turn to the messages
            messages_for_processor_final.append({"role": "user", "content": current_user_turn_llm_formatted_content})
            logger.debug(f"{log_prefix} Final user turn content created. Total messages for processor: {len(messages_for_processor_final)}")

            # --- Prepare inputs and generate ---
            inputs = self.processor.apply_chat_template( 
                messages_for_processor_final, 
                add_generation_prompt=True, # Important for instruct models
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
            ).to(self.device)

            full_decoded_prompt_for_log = self.processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
            logger.debug(f"{log_prefix} [GEMMA] FULL DECODED PROMPT (tokens: {inputs['input_ids'].shape[1]}):\n{full_decoded_prompt_for_log[:3000]}...") # Log a larger chunk

            gen_config_to_use = GenerationConfig.from_dict(self.generation_config.to_dict())
            # Temperature and other settings from the shared config
            # You might adjust these based on whether it's a primary diagnostic call vs. simple clarification later
            gen_config_to_use.temperature = 0.5 # Adjusted from 0.4
            gen_config_to_use.top_p = 0.9
            logger.info(f"{log_prefix} [GEMMA_CONFIG_ADJUST] Using Temp: {gen_config_to_use.temperature}, Top_p: {gen_config_to_use.top_p}")
        
            current_stopping_criteria = self.stopping_criteria
            generation_args = {
                "input_ids": inputs["input_ids"], 
                "attention_mask": inputs.get("attention_mask"), 
                "generation_config": gen_config_to_use, 
                "stopping_criteria": current_stopping_criteria
            }
            # Add pixel_values if they were processed by apply_chat_template (which they should be if image was in content)
            if "pixel_values" in inputs and inputs["pixel_values"] is not None: 
                generation_args["pixel_values"] = inputs.get("pixel_values")
            elif current_image_pil_for_main_gen and ("pixel_values" not in inputs or inputs["pixel_values"] is None):
                logger.warning(f"{log_prefix} PIL image was present but pixel_values not found in processor output. Multimodal input might be incorrect.")


            generated_text_output = "[Error: AI generation failed internally]" 
            async with GLOBAL_LLM_INFERENCE_LOCK:
                logger.debug(f"{log_prefix} Acquired LLM inference lock for main generation.")
            try:
                t_before_model_generate = time.time()
                logger.info(f"{log_prefix} [GEMMA] Calling model.generate() for main response.")
                loop = asyncio.get_event_loop(); GENERATION_TIMEOUT_SECONDS = 500
                
                generated_ids_result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self.model.generate(**generation_args)), 
                    timeout=GENERATION_TIMEOUT_SECONDS 
                )
                
                logger.info(f"{log_prefix} [GEMMA] model.generate() completed in {time.time() - t_before_model_generate:.4f}s")
                
                if generated_ids_result is not None:
                    input_ids_length = inputs["input_ids"].shape[1]
                    newly_generated_ids_tensor = torch.tensor([], device=generated_ids_result.device, dtype=generated_ids_result.dtype) # Ensure tensor type
                    if generated_ids_result.shape[1] > input_ids_length: 
                        newly_generated_ids_tensor = generated_ids_result[0][input_ids_length:]
                    
                    generated_text_output = self.processor.tokenizer.decode(newly_generated_ids_tensor, skip_special_tokens=True).strip()
                else: 
                    generated_text_output = "[Error: Generation failed, no output IDs from model]"
                
                logger.info(f"{log_prefix} [GEMMA] FINAL Decoded generated_text (len {len(generated_text_output)}): '{generated_text_output[:200]}...'")

            except asyncio.TimeoutError: 
                logger.error(f"{log_prefix} [GEMMA] model.generate() TIMED OUT!"); 
                generated_text_output = "[Error: AI response generation timed out]"
            except Exception as e_gen: 
                logger.exception(f"{log_prefix} [GEMMA] model.generate() failed: {e_gen}"); 
                generated_text_output = f"[Error: AI generation failed - {type(e_gen).__name__}]"
            finally: 
                logger.debug(f"{log_prefix} Released LLM inference lock after main generation.")
        
            logger.info(f"{log_prefix} [GEMMA] Total main generate_streaming time: {time.time() - t_start_generate_streaming:.4f}s")
            return self.EmptyStreamer(), generated_text_output
    
        except Exception as e:
        # Ensure log_prefix is defined even if error happens early
            log_prefix_fallback = f"GemmaProc Inst:{self.instance_id_log} GenerateStream Call (Error) SESS:{client_session_id}:"
            logger.exception(f"{log_prefix_fallback} Outer critical error in main generate_streaming: {e}")
            return self.EmptyStreamer(), f"[Critical outer error during Gemma processing: {type(e).__name__} - {str(e)}]"


# The rest of the script: process_user_input_and_respond, handle_client, main
# should be taken from the previous response where they were fully defined,
# making sure to integrate the latest GemmaMultimodalProcessor changes.

# For brevity, I will re-paste process_user_input_and_respond and update_session_summary.
# handle_client and main should be used from the previous full script.
# The key is that process_user_input_and_respond no longer uses classification_result for branching.


    

async def update_session_summary(session_id: str, text_chunk_for_smol: str, current_running_summary: str, gemma_processor_ref: GemmaMultimodalProcessor): # gemma_processor_ref to access session_state
    if not GLOBAL_SUMMARIZER:
        logger.warning(f"Summarizer (SmolLM) not available for session {session_id}.")
        return

    logger.info(f"Starting background summarization for session {session_id}...")
    # Pass current_running_summary to the summarizer
    new_full_summary = await GLOBAL_SUMMARIZER.summarize_text(
        text_to_summarize=text_chunk_for_smol, # This should be just the NEW turns
        current_summary=current_running_summary # Pass existing summary
    )

    if new_full_summary:
        session_state = gemma_processor_ref._get_or_create_session_state(session_id) # Ensures we have the state object
        session_state["current_running_summary"] = new_full_summary
        
        # Store updated summary in SQLite
        db_execute("INSERT OR REPLACE INTO session_summaries (session_id, summary_text, last_updated) VALUES (?, ?, ?)",
                   (session_id, new_full_summary, int(time.time())), commit=True)
        logger.info(f"SESS:{session_id} Updated running summary in state and DB (full): '{new_full_summary}'")
        
        # Optionally, update ChromaDB with the new full summary (could replace or add as new version)
        if GLOBAL_CHROMA_CLIENT and GLOBAL_EMBEDDING_MODEL:
            try:
                summary_id_in_chroma = f"summary_{session_id}" 
                summary_doc_for_chroma = f"Conversation Summary (updated {datetime.now().isoformat()}): {new_full_summary}"
                embedding_array = await asyncio.get_event_loop().run_in_executor(None, GLOBAL_EMBEDDING_MODEL.encode, summary_doc_for_chroma)
                embedding = embedding_array.tolist()
                collection = GLOBAL_CHROMA_CLIENT.get_or_create_collection(name=session_id)
                
                func_to_run = functools.partial(
                    collection.upsert,
                    ids=[summary_id_in_chroma],
                    embeddings=[embedding],
                    documents=[summary_doc_for_chroma],
                    metadatas=[{"type": "running_summary", "timestamp": int(time.time())}]
                )
                await asyncio.get_event_loop().run_in_executor(None, func_to_run)
                logger.info(f"ChromaDB: Upserted running summary for session {session_id}")
            except Exception as e_chroma_sum:
                logger.error(f"ChromaDB: Error upserting summary for {session_id}: {e_chroma_sum}", exc_info=True)
    else:
        logger.warning(f"Background summarization for session {session_id} produced empty result.")


async def process_user_input_and_respond(
    websocket, detector, 
    gemma_processor: GemmaMultimodalProcessor, # Default Gemma processor
    tts_processor,
    user_text: str, client_session_id: str, image_data_b64: str = None, generate_tts: bool = False,
    gemini_processor: GeminiAPIProcessor = None # Optional Gemini processor instance
):
    global GLOBAL_USE_GEMINI_MODEL, GLOBAL_EVAL_MODE_ACTIVE # Access global toggle states

    current_timestamp_ms = int(time.time() * 1000)
    user_message_id = f"um_{uuid.uuid4().hex}"
    
    processed_image_filename_for_db = None
    current_image_pil_for_turn = None # For Gemma
    current_image_bytes_for_gemini = None # For Gemini API
    image_provided_in_this_specific_call = bool(image_data_b64)
    
    ai_final_response_text_for_client = None
    ai_final_response_msg_id = f"am_main_resp_{user_message_id[:10]}"
    active_llm_service = "Gemma" # Default

    logger.info(f"SESS:{client_session_id} PROC_INPUT UserMsgID:{user_message_id} Text:'{user_text[:70]}' ImgProvided:{image_provided_in_this_specific_call} TTS_Req:{generate_tts} GeminiToggle:{GLOBAL_USE_GEMINI_MODEL} EvalToggle:{GLOBAL_EVAL_MODE_ACTIVE}")

    user_text_for_db = user_text if user_text else ("[User sent an image]" if image_provided_in_this_specific_call else "[Empty User Input]")
    
    if image_provided_in_this_specific_call and image_data_b64:
        try:
            decoded_image_data = base64.b64decode(image_data_b64)
            current_image_bytes_for_gemini = decoded_image_data # Store bytes for Gemini
            current_image_pil_for_turn = Image.open(io.BytesIO(decoded_image_data)) # For Gemma & saving
            
            session_media_path = os.path.join(MEDIA_STORAGE_DIR, client_session_id)
            if not os.path.exists(session_media_path): os.makedirs(session_media_path)
            processed_image_filename_for_db = f"img_{user_message_id}.jpg"
            image_filepath = os.path.join(session_media_path, processed_image_filename_for_db)
            with open(image_filepath, "wb") as f: f.write(decoded_image_data)
            logger.info(f"SESS:{client_session_id} Image saved to {image_filepath}")
            
            # For Gemma flow (if it might be used), buffer PIL and query context
            if not GLOBAL_USE_GEMINI_MODEL or not gemini_processor: 
                 await gemma_processor.set_image(decoded_image_data, client_session_id, user_text_for_db)
        except Exception as e:
            logger.error(f"SESS:{client_session_id} Error handling uploaded image: {e}", exc_info=True)
            await websocket.send(json.dumps({"error": "Error processing your uploaded image.", "sender": "System", "session_id": client_session_id}))
            await save_message_db(msg_id=user_message_id, session_id=client_session_id, sender="User", timestamp=current_timestamp_ms, text_content="[User image upload error]", image_filename=None, data_type="user_turn_image_error")
            return

    # Save user message to DB (this includes the user_text_for_db and processed_image_filename_for_db)
    await save_message_db(msg_id=user_message_id, session_id=client_session_id, sender="User", timestamp=current_timestamp_ms, text_content=user_text_for_db, image_filename=processed_image_filename_for_db, data_type="user_turn")
    
    # Update Gemma's short-term history state with the user's turn, especially if Gemma will be used.
    # If Gemini is primary, this history still helps populate the RAG context that Gemini might use.
    gemma_processor._update_history_with_complete_response(
        client_session_id, 
        user_text_for_db, # The actual text user sent this turn
        "",               # No AI response yet for this specific history entry
        image_provided_in_this_specific_call, 
        is_assistant_image_description=False, 
        is_assistant_clarification_turn=False
    )

    image_description_for_gemma_context = None 

    try:
        # --- Determine which LLM to use for the main response ---
        use_gemini_for_main_response = GLOBAL_USE_GEMINI_MODEL and gemini_processor is not None
        
        if use_gemini_for_main_response:
            active_llm_service = "Gemini"
            logger.info(f"SESS:{client_session_id} Using Gemini API for main response.")
            logger.debug(f"SESS:{client_session_id} Type of gemini_processor: {type(gemini_processor)}")
            logger.debug(f"SESS:{client_session_id} Attributes of gemini_processor: {dir(gemini_processor)}") # Will list all methods/attrs
            
            # For Gemini, RAG context will be combined with the user query.
            # The system prompt for Gemini is long and detailed, provided in GeminiAPIProcessor.
            rag_context_for_llm = await gemma_processor._retrieve_rag_context(user_text_for_db, client_session_id) 
            
            # Use the long system prompt you defined for the "Gemini" toggle
            gemini_chat_system_prompt = GEMINI_CHAT_SYSTEM_PROMPT_FOR_MAIN_RESPONSE

            ai_final_response_text_for_client = await gemini_processor.generate_response_for_chat(
                system_prompt_text=gemini_chat_system_prompt,
                user_query_text=user_text_for_db,
                image_bytes=current_image_bytes_for_gemini, 
                rag_context=rag_context_for_llm
            )
        else: # Use Gemma (local model)
            active_llm_service = "Gemma"
            logger.info(f"SESS:{client_session_id} Using local Gemma model for main response.")
            
            # Gemma Flow: Potentially Image Description -> Main Diagnostic
            if image_provided_in_this_specific_call and current_image_pil_for_turn:
                logger.info(f"SESS:{client_session_id} (Gemma) Generating image description.")
                session_state_gemma = gemma_processor._get_or_create_session_state(client_session_id)
                pil_for_desc = session_state_gemma.get("last_image_pil_for_description")
                query_context_for_desc = session_state_gemma.get("last_image_user_query_context")

                if pil_for_desc:
                    image_description_for_gemma_context = await gemma_processor.describe_image(
                        pil_for_desc, client_session_id, query_context_for_desc
                    )
                    if image_description_for_gemma_context and not image_description_for_gemma_context.lower().startswith("[error:"):
                        desc_msg_id = f"ai_img_desc_{user_message_id}"
                        desc_ts = int(time.time() * 1000)
                        await websocket.send(json.dumps({"text_response": f"Image Description: {image_description_for_gemma_context}", "sender": "AI", "session_id": client_session_id, "id": desc_msg_id, "timestamp": desc_ts, "is_image_description": True, "llm_model_used": "gemma_desc"}))
                        await save_message_db(msg_id=desc_msg_id, session_id=client_session_id, sender="AI", timestamp=desc_ts, text_content=image_description_for_gemma_context, data_type="ai_image_description", llm_model_used="gemma_desc")
                        gemma_processor._update_history_with_complete_response(client_session_id, "[Image Description Context]", image_description_for_gemma_context, True, True, False)
                        await gemma_processor.add_image_summary_to_vector_db(client_session_id, user_message_id, image_description_for_gemma_context)
                    else:
                        logger.warning(f"SESS:{client_session_id} (Gemma) Image description failed/empty: '{image_description_for_gemma_context}'")
                        await websocket.send(json.dumps({"text_response": "[System: Could not generate image description via Gemma. Proceeding with text.]", "sender": "AI", "session_id": client_session_id, "id": f"ai_img_desc_fail_{user_message_id}", "timestamp": int(time.time()*1000)}))
                        image_description_for_gemma_context = None 
                    
                    session_state_gemma["last_image_pil_for_description"] = None 
                    session_state_gemma["last_image_user_query_context"] = None
                else: image_description_for_gemma_context = None
            
            # Gemma Main Diagnostic call
            _streamer, raw_gemma_response = await gemma_processor.generate_streaming(
                user_text_for_db, client_session_id, 
                image_description=image_description_for_gemma_context 
            )
            # Sanitization
            if raw_gemma_response and raw_gemma_response.strip() and not raw_gemma_response.lower().startswith(("[error:", "error:")) and "critical error" not in raw_gemma_response.lower():
                text_to_sanitize = raw_gemma_response; text_to_sanitize = re.sub(r'(\r\n|\r|\n){3,}', '\n\n', text_to_sanitize); lines = text_to_sanitize.split('\n'); stripped_lines = [line.strip() for line in lines]; final_lines = [line for line in stripped_lines if line or (len(stripped_lines) == 1 and not line)]; sanitized_gemma_response_text = '\n'.join(final_lines).strip()
                if not sanitized_gemma_response_text.strip() and raw_gemma_response.strip(): sanitized_gemma_response_text = re.sub(r'\n{2,}', '\n', raw_gemma_response.strip())
                ai_final_response_text_for_client = sanitized_gemma_response_text if sanitized_gemma_response_text.strip() else "I seem to have generated an empty response. Could you try rephrasing?"
            else:
                ai_final_response_text_for_client = raw_gemma_response if raw_gemma_response else "[Gemma Error: Failed to generate response]"
            logger.info(f"SESS:{client_session_id} (Gemma) Main Sanitized Out: '{ai_final_response_text_for_client[:100]}...'")

        # --- Send Final Main AI Response, Save, Update History, TTS ---
        if ai_final_response_text_for_client is None:
            ai_final_response_text_for_client = "[Error: AI failed to produce a response for this turn]"
            
        final_response_timestamp = int(time.time()*1000)
        await websocket.send(json.dumps({
            "text_response": ai_final_response_text_for_client, "sender": "AI", 
            "session_id": client_session_id, "id": ai_final_response_msg_id, 
            "timestamp": final_response_timestamp, "llm_model_used": active_llm_service
        }))
        
        await save_message_db(
            msg_id=ai_final_response_msg_id, session_id=client_session_id, sender="AI", 
            timestamp=final_response_timestamp, text_content=ai_final_response_text_for_client, 
            data_type="ai_response", llm_model_used=active_llm_service
        )

        if active_llm_service == "Gemma": # Only update Gemma's specific history format if Gemma was used.
            gemma_processor._update_history_with_complete_response(
                client_session_id, "", ai_final_response_text_for_client, 
                image_provided_in_this_specific_call, 
                is_assistant_image_description=False, # This is the main response, not the initial desc
                is_assistant_clarification_turn=False 
            )

        if generate_tts and ai_final_response_text_for_client and not ai_final_response_text_for_client.lower().startswith("[error:"):
            text_for_tts_input = strip_markdown_for_tts(ai_final_response_text_for_client)
            if text_for_tts_input:
                initial_audio = await tts_processor.synthesize_initial_speech(text_for_tts_input)
                if initial_audio is not None:
                    tts_processor_sr = tts_processor.target_sr; audio_duration_sec = len(initial_audio) / tts_processor_sr
                    required_until_time = time.time() + audio_duration_sec + detector.vad_process_cooldown 
                    await detector.extend_vad_cooldown(required_until_time); 
                    audio_bytes = (initial_audio * 32767).astype(np.int16).tobytes(); base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                    await websocket.send(json.dumps({"audio": base64_audio, "sender": "AI", "session_id": client_session_id, "id": f"audio_{ai_final_response_msg_id}"})); 
                    logger.info(f"SESS:{client_session_id} Sent AI audio ({audio_duration_sec:.2f}s).")
                else: logger.warning(f"SESS:{client_session_id} TTS synthesis failed.")
            else: logger.warning(f"SESS:{client_session_id} Text for TTS became empty after stripping. Skipping TTS.")
        
        # --- EVALUATION MODE ---
        if GLOBAL_EVAL_MODE_ACTIVE and gemini_processor: # Check if Gemini processor is available
            logger.info(f"SESS:{client_session_id} Eval Mode is ON. Generating Gemini Eval response.")
            session_state_for_eval = gemma_processor._get_or_create_session_state(client_session_id)
            latest_summary_for_eval = session_state_for_eval.get("current_running_summary", "No summary available.")
            
            # Pass the user text that prompted the main AI response for eval context
            eval_user_query_context = user_text_for_db 
            
            eval_response_text = await gemini_processor.generate_eval_response(
                newest_summary=latest_summary_for_eval,
                latest_user_query=eval_user_query_context, 
                latest_image_bytes=current_image_bytes_for_gemini # Image from current turn
            )
            if eval_response_text and not eval_response_text.lower().startswith("[error:"):
                eval_msg_id = f"ai_eval_{user_message_id[:10]}"
                eval_ts = int(time.time()*1000)
                await websocket.send(json.dumps({
                    "type": "eval_response", "text_response": eval_response_text, 
                    "sender": "AI_Evaluator", "session_id": client_session_id, 
                    "id": eval_msg_id, "timestamp": eval_ts, "llm_model_used": "gemini_eval"
                }))
                await save_message_db(msg_id=eval_msg_id, session_id=client_session_id, sender="AI_Evaluator", timestamp=eval_ts, text_content=eval_response_text, data_type="ai_eval_response", llm_model_used="gemini_eval")
            else:
                logger.warning(f"SESS:{client_session_id} Gemini Eval response was empty or error: {eval_response_text}")

        # --- Summarization Trigger ---
        if GLOBAL_SUMMARIZER and client_session_id:
            # ... (Summarization logic as in your provided script, using text_chunk_for_summarizer etc.)
            session_state_summary_trigger = gemma_processor._get_or_create_session_state(client_session_id)
            session_state_summary_trigger["messages_since_last_summary"] = session_state_summary_trigger.get("messages_since_last_summary", 0) + 2 
            if session_state_summary_trigger["messages_since_last_summary"] >= SUMMARY_TRIGGER_THRESHOLD:
                logger.info(f"SESS:{client_session_id} Summary threshold reached. Triggering summarization.")
                query_limit = SUMMARY_TRIGGER_THRESHOLD + 4 
                query = "SELECT sender, text_content FROM messages WHERE session_id = ? AND data_type IN ('user_turn', 'ai_response', 'ai_image_description', 'ai_eval_response') ORDER BY timestamp DESC LIMIT ?"
                messages_for_summary_raw = db_execute(query, (client_session_id, query_limit), fetchall=True)
                text_chunk_for_summarizer = ""
                if messages_for_summary_raw: text_chunk_for_summarizer = "\n".join([f"{row['sender']}: {row['text_content']}" for row in reversed(messages_for_summary_raw)])
                if text_chunk_for_summarizer:
                    asyncio.create_task(update_session_summary(client_session_id, text_chunk_for_summarizer, session_state_summary_trigger.get("current_running_summary", ""), gemma_processor ))
                    session_state_summary_trigger["messages_since_last_summary"] = 0 
                else: logger.warning(f"SESS:{client_session_id} No new text found for summary despite threshold.")
            else: logger.debug(f"SESS:{client_session_id} Messages since last summary: {session_state_summary_trigger['messages_since_last_summary']}")

    except asyncio.TimeoutError: 
        logger.error(f"SESS:{client_session_id} A stage in process_user_input_and_respond timed out.")
        fallback_timeout_msg = "[Error: Processing your request took too long. Please try again.]"
        timeout_msg_id = f"am_timeout_{user_message_id[:10]}"
        timeout_ts = int(time.time()*1000)
        try: await websocket.send(json.dumps({"text_response": fallback_timeout_msg, "sender": "AI", "session_id": client_session_id, "id": timeout_msg_id, "timestamp": timeout_ts, "llm_model_used": active_llm_service}))
        except: pass
        await save_message_db(msg_id=timeout_msg_id, session_id=client_session_id, sender="AI", timestamp=timeout_ts, text_content=fallback_timeout_msg, data_type="ai_error", llm_model_used=active_llm_service)
        if active_llm_service == "Gemma": gemma_processor._update_history_with_complete_response(client_session_id, "", fallback_timeout_msg, image_provided_in_this_specific_call, False, False)

    except Exception as e_main_process:
        logger.exception(f"SESS:{client_session_id} Critical error in process_user_input_and_respond: {e_main_process}")
        fallback_error_msg = "[Error: An unexpected server error occurred. Please try again.]"
        error_msg_id = f"am_error_{user_message_id[:10]}"
        error_ts = int(time.time()*1000)
        try: await websocket.send(json.dumps({ "text_response": fallback_error_msg, "sender": "AI", "session_id": client_session_id, "id": error_msg_id, "timestamp": error_ts, "llm_model_used": active_llm_service}))
        except: pass
        await save_message_db(msg_id=error_msg_id, session_id=client_session_id, sender="AI", timestamp=error_ts, text_content=fallback_error_msg, data_type="ai_error", llm_model_used=active_llm_service)
        if active_llm_service == "Gemma": gemma_processor._update_history_with_complete_response(client_session_id, "", fallback_error_msg, image_provided_in_this_specific_call, False, False)

    finally:
        if generate_tts and detector: 
            await detector.clear_vad_cooldown(); await detector.set_tts_playing(False)
        
        # Clear Gemma's buffered image state if Gemma was involved with an image this turn
        if image_provided_in_this_specific_call and (not GLOBAL_USE_GEMINI_MODEL or not gemini_processor): 
            session_state_for_clear = gemma_processor._get_or_create_session_state(client_session_id)
            if session_state_for_clear.get("last_image_pil_for_description") is not None:
                logger.info(f"SESS:{client_session_id} Clearing last_image_pil_for_description from Gemma session_state (Gemma path).")
                session_state_for_clear["last_image_pil_for_description"] = None 
                session_state_for_clear["last_image_user_query_context"] = None

# handle_client and main functions as provided in the previous full script, with SmolLM loading in main().
async def handle_client(websocket):
    client_addr = websocket.remote_address
    handler_instance_id = f"handler_{uuid.uuid4().hex[:6]}" 
    logger.info(f"SESS_HANDLER ({handler_instance_id}) for {client_addr}: New connection. Initializing...")
    
    active_client_session_id = f"init_{uuid.uuid4().hex[:8]}" 
    logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Initial handler active_client_session_id set to {active_client_session_id}")

    detector = None
    gemma_processor_instance = None
    gemini_processor_instance = None # Will be initialized if API key is set
    handler_tasks = []

    try: # This is the try block that Pylance was referring to (line 1563 in your error)
        detector = AudioSegmentDetector()
        gemma_processor_instance = GemmaMultimodalProcessor(
            shared_model=GLOBAL_GEMMA_MODEL,
            shared_processor=GLOBAL_GEMMA_PROCESSOR,
            shared_hf_tokenizer=GLOBAL_GEMMA_PROCESSOR.tokenizer
        )
        logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Detector and GemmaProcessor instance created.")

        transcriber = GLOBAL_WHISPER_TRANSCRIBER
        tts_processor = GLOBAL_KOKORO_TTS_PROCESSOR
        
        if not all([GLOBAL_GEMMA_MODEL, GLOBAL_GEMMA_PROCESSOR, transcriber, tts_processor, detector, gemma_processor_instance, GLOBAL_EMBEDDING_MODEL, GLOBAL_CHROMA_CLIENT]):
            logger.error(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Core components missing. Closing."); 
            await websocket.close(code=1011, reason="Server models not fully ready"); return
        if not GLOBAL_SUMMARIZER:
            logger.warning(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Summarizer model is not available.")

        # --- Nested async helper functions ---
        async def send_keepalive():
            nonlocal active_client_session_id 
            task_name_ka = f"KeepaliveTask_{handler_instance_id}"
            logger.info(f"{task_name_ka} for {client_addr} (InitialSess: {active_client_session_id}): Starting.")
            try:
                while websocket.state == WebSocketConnectionState.OPEN: 
                    await websocket.ping()
                    logger.debug(f"{task_name_ka} sent keepalive ping.")
                    await asyncio.sleep(15) 
            except asyncio.CancelledError: logger.info(f"{task_name_ka} ({client_addr}, Sess: {active_client_session_id}): Cancelled."); raise
            except websockets.exceptions.ConnectionClosed: logger.info(f"{task_name_ka} ({client_addr}, Sess: {active_client_session_id}): Connection closed.")
            except Exception as e_ka: logger.error(f"{task_name_ka} ({client_addr}, Sess: {active_client_session_id}): Error: {e_ka}", exc_info=True)
            finally: logger.info(f"{task_name_ka} ({client_addr}, Sess: {active_client_session_id}): Exiting.")

        async def detect_speech_segments():
            nonlocal active_client_session_id 
            task_name_ds = f"SpeechDetectTask_{handler_instance_id}"
            logger.info(f"{task_name_ds} for {client_addr} (InitialSess: {active_client_session_id}): Starting.")
            MIN_WORD_COUNT_FOR_ASR_TO_LLM = 4
            try:
                while websocket.state == WebSocketConnectionState.OPEN:
                    current_vad_processing_session_id = active_client_session_id 
                    if not current_vad_processing_session_id or str(current_vad_processing_session_id).startswith("init_"): 
                        await asyncio.sleep(0.1); continue 

                    speech_segment_bytes = await detector.get_next_segment()
                    if speech_segment_bytes:
                        transcription = await transcriber.transcribe(speech_segment_bytes)
                        transcription_stripped = transcription.strip() if transcription else ""
                        
                        is_valid_for_processing = False
                        if transcription_stripped:
                            if not any(c.isalnum() for c in transcription_stripped): logger.info(f"{task_name_ds} ({current_vad_processing_session_id}): VAD Filter: Skip punctuation-only: '{transcription_stripped}'")
                            else:
                                words = [w for w in transcription_stripped.split() if any(c.isalnum() for c in w)]
                                if len(words) < MIN_WORD_COUNT_FOR_ASR_TO_LLM: logger.info(f"{task_name_ds} ({current_vad_processing_session_id}): VAD Filter: Skip short text (<{MIN_WORD_COUNT_FOR_ASR_TO_LLM} words): '{transcription_stripped}'")
                                else:
                                    filler_patterns = [r'^(um+|uh+|ah+|oh+|hm+|mhm+|hmm+|er+|erm+)$', r'^(okay|ok|yes|no|yeah|nah|right|alright|got it|i see)$', r'^(bye+|goodbye|see ya|later)$', r'^(thank(s| you)?( very much)?)$']
                                    if any(re.fullmatch(pattern, transcription_stripped.lower()) for pattern in filler_patterns): logger.info(f"{task_name_ds} ({current_vad_processing_session_id}): VAD Filter: Skip common filler/ack: '{transcription_stripped}'")
                                    else: is_valid_for_processing = True
                        
                        if is_valid_for_processing:
                            try: 
                                await websocket.send(json.dumps({"user_transcription": transcription_stripped, "sender": "User", "session_id": current_vad_processing_session_id, "id": f"ut_{uuid.uuid4().hex}", "timestamp": int(time.time()*1000)}))
                            except websockets.exceptions.ConnectionClosed: logger.warning(f"{task_name_ds} VAD: WebSocket closed before sending transcription."); break 
                            
                            await detector.set_vad_cooldown()
                            asyncio.create_task(process_user_input_and_respond(
                                websocket, detector, gemma_processor_instance, tts_processor, 
                                user_text=transcription_stripped, client_session_id=current_vad_processing_session_id, 
                                image_data_b64=None, generate_tts=True,
                                gemini_processor=gemini_processor_instance 
                            ))
            except asyncio.CancelledError: logger.info(f"{task_name_ds} ({client_addr}, Sess: {active_client_session_id}): Cancelled."); raise
            except websockets.exceptions.ConnectionClosed: logger.info(f"{task_name_ds} ({client_addr}, Sess: {active_client_session_id}): Connection closed.")
            except Exception as e_ds: logger.exception(f"{task_name_ds} ({client_addr}, Sess: {active_client_session_id}): Unexpected error: {e_ds}")
            finally: logger.info(f"{task_name_ds} ({client_addr}, Sess: {active_client_session_id}): Exiting.")

        # THIS IS WHERE THE INDENTATION WAS WRONG BEFORE
        async def receive_data_from_client(): # Correctly indented under handle_client
            nonlocal active_client_session_id, gemini_processor_instance 
            global GEMINI_API_KEY_STORE, GLOBAL_USE_GEMINI_MODEL, GLOBAL_EVAL_MODE_ACTIVE, GLOBAL_GROUNDING_ACTIVE

            task_name_rd = f"ReceiveDataTask_{handler_instance_id}"
            logger.info(f"{task_name_rd} for {client_addr} (InitialSess: {active_client_session_id}): Starting.")
            try:
                async for message_str in websocket:
                    current_op_session_id = None 
                    client_intended_session_id_for_msg = None 
                    msg_type = "unknown_type"
                    try:
                        data = json.loads(message_str)
                        client_intended_session_id_for_msg = data.get("sessionId")
                        msg_type = data.get("type", "unknown_type") 
                        
                        if "realtime_input" in data and msg_type == "unknown_type":
                            msg_type = "realtime_input_message"
                        
                        logger.debug(f"{task_name_rd} RAW_MSG_IN - Type='{msg_type}', MsgSess='{client_intended_session_id_for_msg}', HandlerActiveSess='{active_client_session_id}'")
                        
                        if client_intended_session_id_for_msg and \
                           not str(client_intended_session_id_for_msg).startswith("client_temp_") and \
                           not str(client_intended_session_id_for_msg).startswith("init_") and \
                           active_client_session_id != client_intended_session_id_for_msg:
                            active_client_session_id = client_intended_session_id_for_msg
                            logger.info(f"{task_name_rd} Switched handler's active_client_session_id to: {active_client_session_id} from incoming message.")

                        current_op_session_id = active_client_session_id

                        if current_op_session_id is None or str(current_op_session_id).startswith("init_"):
                            if client_intended_session_id_for_msg and \
                               not str(client_intended_session_id_for_msg).startswith("client_temp_") and \
                               not str(client_intended_session_id_for_msg).startswith("init_"):
                                current_op_session_id = client_intended_session_id_for_msg
                                active_client_session_id = client_intended_session_id_for_msg 
                                logger.info(f"{task_name_rd} Adopted session_id '{active_client_session_id}' from message for current operation and handler.")
                        
                        logger.debug(f"{task_name_rd} OpSessID='{current_op_session_id}' for MsgType='{msg_type}'")

                        if msg_type == "config":
                            cfg_data = data.get('data', {}); initial_id_from_config = cfg_data.get('initialSessionId')
                            if cfg_data.get('clientReady') and initial_id_from_config and \
                               not str(initial_id_from_config).startswith("client_temp_") and \
                               not str(initial_id_from_config).startswith("init_"):
                                if active_client_session_id != initial_id_from_config:
                                    active_client_session_id = initial_id_from_config
                                    logger.info(f"{task_name_rd} Handler active_client_session_id updated by CONFIG message to '{active_client_session_id}'.")
                                current_op_session_id = active_client_session_id 
                            continue 
                        
                        if current_op_session_id is None or str(current_op_session_id).startswith("init_"):
                            if msg_type not in ["load_sessions_request", "create_new_session_backend", "set_api_key", "update_toggle_state"]:
                                logger.warning(f"{task_name_rd} Operation '{msg_type}' requires an active session, but current_op_session_id is '{current_op_session_id}'. Sending error.")
                                await websocket.send(json.dumps({"type": "error", "message": f"Operation '{msg_type}' requires an active session.", "session_id": client_intended_session_id_for_msg or "unknown"}))
                                continue
                        
                        elif msg_type == "set_api_key":
                            api_data = data.get("data", {})
                            service = api_data.get("service")
                            api_key_value = api_data.get("apiKey")
                            ack_session_id = current_op_session_id if (current_op_session_id and not str(current_op_session_id).startswith("init_")) else "global_setting"

                            if service == "gemini" and isinstance(api_key_value, str) and api_key_value.strip():
                                GEMINI_API_KEY_STORE = api_key_value.strip() 
                                save_server_settings() # <<<< SAVE THE SETTINGS
                                logger.info(f"{task_name_rd} Gemini API Key Stored globally (length: {len(GEMINI_API_KEY_STORE)}).")
                                try:
                                    # Re-initialize GeminiAPIProcessor instance for THIS client handler's scope
                                    gemini_processor_instance = GeminiAPIProcessor(api_key=GEMINI_API_KEY_STORE)
                                    logger.info(f"{task_name_rd} GeminiAPIProcessor (instance for handler {handler_instance_id}) initialized/re-initialized with API key.")
                                    await websocket.send(json.dumps({"type": "api_key_set_ack", "service": "gemini", "status": "success", "session_id": ack_session_id}))
                                except Exception as e_gem_proc:
                                    logger.error(f"{task_name_rd} Failed to initialize GeminiAPIProcessor for handler {handler_instance_id} after setting key: {e_gem_proc}")
                                    # GEMINI_API_KEY_STORE = None # Don't clear global if just this handler fails init
                                    gemini_processor_instance = None 
                                    await websocket.send(json.dumps({"type": "api_key_set_ack", "service": "gemini", "status": "error", "message": "Failed to initialize with API key, but key was saved.", "session_id": ack_session_id}))
                            # ... (rest of set_api_key else block) ...
                            continue 

                        elif msg_type == "update_toggle_state":
                            toggle_data = data.get("data", {})
                            toggle_name = toggle_data.get("toggleName")
                            is_enabled = toggle_data.get("isEnabled")
                            ack_session_id_toggle = current_op_session_id if (current_op_session_id and not str(current_op_session_id).startswith("init_")) else "global_setting"

                            if isinstance(is_enabled, bool):
                                if toggle_name == "gemini":
                                    if is_enabled and not GEMINI_API_KEY_STORE:
                                        logger.warning(f"{task_name_rd} Cannot enable 'Gemini' toggle: API key not set.")
                                        await websocket.send(json.dumps({"type": "error", "message": "Gemini API key is not set. Please set it first to use this feature.", "action_required": "set_gemini_api_key", "session_id": ack_session_id_toggle}))
                                        await websocket.send(json.dumps({"type": "toggle_state_update_ack", "toggleName": "gemini", "isEnabled": GLOBAL_USE_GEMINI_MODEL, "status": "error", "message": "API key missing", "session_id": ack_session_id_toggle}))
                                    else:
                                        GLOBAL_USE_GEMINI_MODEL = is_enabled
                                        logger.info(f"{task_name_rd} Global 'Use Gemini' toggle set to: {GLOBAL_USE_GEMINI_MODEL}")
                                        await websocket.send(json.dumps({"type": "toggle_state_update_ack", "toggleName": "gemini", "isEnabled": GLOBAL_USE_GEMINI_MODEL, "status": "success", "session_id": ack_session_id_toggle}))
                                elif toggle_name == "eval":
                                    if is_enabled and not GEMINI_API_KEY_STORE:
                                        logger.warning(f"{task_name_rd} Cannot enable 'Eval' toggle: Gemini API key not set.")
                                        await websocket.send(json.dumps({"type": "error", "message": "Gemini API key is not set. Eval mode requires it. Please set it first.", "action_required": "set_gemini_api_key", "session_id": ack_session_id_toggle}))
                                        await websocket.send(json.dumps({"type": "toggle_state_update_ack", "toggleName": "eval", "isEnabled": GLOBAL_EVAL_MODE_ACTIVE, "status": "error", "message": "API key missing for Eval", "session_id": ack_session_id_toggle}))
                                    else:
                                        GLOBAL_EVAL_MODE_ACTIVE = is_enabled
                                        logger.info(f"{task_name_rd} Global 'Eval Mode' toggle set to: {GLOBAL_EVAL_MODE_ACTIVE}")
                                        await websocket.send(json.dumps({"type": "toggle_state_update_ack", "toggleName": "eval", "isEnabled": GLOBAL_EVAL_MODE_ACTIVE, "status": "success", "session_id": ack_session_id_toggle}))
                                elif toggle_name == "grounding": 
                                    GLOBAL_GROUNDING_ACTIVE = is_enabled
                                    logger.info(f"{task_name_rd} Global 'Grounding' toggle set to: {GLOBAL_GROUNDING_ACTIVE}")
                                    await websocket.send(json.dumps({"type": "toggle_state_update_ack", "toggleName": "grounding", "isEnabled": GLOBAL_GROUNDING_ACTIVE, "status": "success", "session_id": ack_session_id_toggle}))
                                else:
                                    logger.warning(f"{task_name_rd} Unknown toggle name received: {toggle_name}")
                                    await websocket.send(json.dumps({"type": "toggle_state_update_ack", "toggleName": toggle_name, "isEnabled": "unknown", "status": "error", "message": "Unknown toggle name.", "session_id": ack_session_id_toggle}))
                            else:
                                logger.warning(f"{task_name_rd} Invalid 'update_toggle_state' message payload: {toggle_data}")
                                await websocket.send(json.dumps({"type": "toggle_state_update_ack", "toggleName": toggle_name or "unknown", "isEnabled": "unknown", "status": "error", "message": "Invalid toggle data format.", "session_id": ack_session_id_toggle}))
                            continue 
                        
                        elif msg_type == "load_sessions_request":
                            ack_id_lsr = current_op_session_id if (current_op_session_id and not str(current_op_session_id).startswith("init_")) else (client_intended_session_id_for_msg if (client_intended_session_id_for_msg and not str(client_intended_session_id_for_msg).startswith("init_")) else f"temp_ack_lsr_{uuid.uuid4().hex[:4]}")
                            sessions_list = get_all_sessions_db(); await websocket.send(json.dumps({"type": "sessions_list", "data": sessions_list, "session_id": ack_id_lsr}))
                        
                        elif msg_type == "create_new_session_backend":
                            session_payload = data.get("data", {}); s_id = session_payload.get("id"); s_name = session_payload.get("name"); s_ts = session_payload.get("timestamp", int(time.time() * 1000))
                            if s_id and s_name:
                                if create_new_session_db(s_id, s_name, s_ts): active_client_session_id = s_id; current_op_session_id = s_id; logger.info(f"{task_name_rd} New session created and activated: {s_id}"); sessions_list = get_all_sessions_db(); await websocket.send(json.dumps({"type": "sessions_list", "data": sessions_list, "newly_created_id": s_id, "session_id": current_op_session_id}))
                                else: await websocket.send(json.dumps({"type": "session_create_error", "sessionId": s_id, "error": "Failed to create on server.", "session_id": current_op_session_id or client_intended_session_id_for_msg}))
                        
                        elif msg_type == "load_session_messages_request":
                            req_session_id_to_load = data.get("sessionId_to_load"); op_context_id_lsmr = current_op_session_id if (current_op_session_id and not str(current_op_session_id).startswith("init_")) else client_intended_session_id_for_msg
                            if not req_session_id_to_load or str(req_session_id_to_load).startswith("init_"): await websocket.send(json.dumps({"type": "error", "message": "Invalid session ID for loading messages.", "session_id": op_context_id_lsmr or "unknown"}))
                            else: messages_data = get_messages_for_session_db(req_session_id_to_load); await websocket.send(json.dumps({"type": "session_messages_data", "sessionId_loaded": req_session_id_to_load, "messages": messages_data, "session_id": op_context_id_lsmr }))
                        
                        elif msg_type == "rename_session_request":
                            s_id_to_rename = data.get("sessionIdToRename"); new_name = data.get("newName")
                            if s_id_to_rename and new_name and current_op_session_id and not str(current_op_session_id).startswith("init_"):
                                if rename_session_db(s_id_to_rename, new_name, int(time.time() * 1000)): await websocket.send(json.dumps({"type": "session_renamed_ack", "sessionId": s_id_to_rename, "newName": new_name, "lastUpdatedAt": int(time.time()*1000), "session_id": current_op_session_id}))
                                else: await websocket.send(json.dumps({"type": "session_rename_error", "sessionId": s_id_to_rename, "error": "Failed to rename session on server.", "session_id": current_op_session_id}))
                        
                        elif msg_type == "delete_session_request":
                             s_id_to_delete = data.get("sessionIdToDelete")
                             if s_id_to_delete : 
                                 logger.info(f"{task_name_rd} Attempting to delete session: {s_id_to_delete} (current handler active session: {active_client_session_id})")
                                 if delete_session_db(s_id_to_delete):
                                     await websocket.send(json.dumps({"type": "session_deleted_ack", "sessionId": s_id_to_delete, "session_id": active_client_session_id})) 
                                     if active_client_session_id == s_id_to_delete: logger.info(f"{task_name_rd} Active session {s_id_to_delete} was deleted. Resetting handler's active_client_session_id to an initial state."); active_client_session_id = f"init_after_delete_{uuid.uuid4().hex[:8]}" 
                                 else: await websocket.send(json.dumps({"type": "session_delete_error", "sessionId": s_id_to_delete, "error": "Failed to delete session on server.", "session_id": active_client_session_id}))
                             else: logger.warning(f"{task_name_rd} Invalid parameters for delete_session_request: sessionIdToDelete missing. Context session: {active_client_session_id}"); await websocket.send(json.dumps({"type": "session_delete_error", "sessionId": "MISSING_ID", "error": "Session ID to delete was not provided.", "session_id": active_client_session_id}))
                        
                        elif msg_type == "text_input":
                            if not current_op_session_id or str(current_op_session_id).startswith("init_"): 
                                await websocket.send(json.dumps({"type": "error", "message": "No active session for text input.", "session_id": client_intended_session_id_for_msg or "unknown"}))
                            else:
                                text_payload = data.get("data", {}); user_text = text_payload.get("text", "").strip(); image_b64 = text_payload.get("image_data"); should_generate_tts = text_payload.get("request_tts", False)
                                if user_text or image_b64: 
                                    asyncio.create_task(process_user_input_and_respond(
                                        websocket, detector, gemma_processor_instance, tts_processor, 
                                        user_text=user_text, client_session_id=current_op_session_id, 
                                        image_data_b64=image_b64, generate_tts=should_generate_tts,
                                        gemini_processor=gemini_processor_instance 
                                    ))
                                else: logger.warning(f"{task_name_rd} Received 'text_input' with no text and no image for session {current_op_session_id}.")
                        
                        elif msg_type == "realtime_input_message":
                            if not current_op_session_id or str(current_op_session_id).startswith("init_"): logger.error(f"{task_name_rd} OpSessID invalid for realtime_input: {current_op_session_id}")
                            else:
                                realtime_payload = data.get("realtime_input")
                                if isinstance(realtime_payload, dict) and "media_chunks" in realtime_payload:
                                    audio_data_to_decode = None
                                    for chunk_info in realtime_payload.get("media_chunks", []):
                                        logger.debug(f"{task_name_rd} Processing media_chunk item: {chunk_info}")
                                        chunk_mime_type = chunk_info.get("mime_type", "").lower(); b64_data_candidate = chunk_info.get("data")
                                        if isinstance(b64_data_candidate, str):
                                            if "audio/pcm" in chunk_mime_type or "audio/wav" in chunk_mime_type or "audio/webm" in chunk_mime_type or not chunk_mime_type:
                                                if ',' in b64_data_candidate: b64_data_candidate = b64_data_candidate.split(',', 1)[1]
                                                audio_data_to_decode = b64_data_candidate; logger.debug(f"{task_name_rd} Identified audio data (mime: '{chunk_mime_type}') for VAD."); break 
                                            else: logger.warning(f"{task_name_rd} Received media_chunk with non-audio mime_type '{chunk_mime_type}', skipping for VAD.")
                                        else: logger.warning(f"{task_name_rd} media_chunk 'data' is not a string: {type(b64_data_candidate)}")
                                    if audio_data_to_decode and detector:
                                        try: audio_bytes = base64.b64decode(audio_data_to_decode); await detector.add_audio(audio_bytes)
                                        except Exception as e_audio: logger.error(f"{task_name_rd} Error processing realtime_input audio: {e_audio} (Data: {str(audio_data_to_decode)[:50]}...)")
                                    elif not audio_data_to_decode: logger.warning(f"{task_name_rd} No valid audio data extracted from realtime_input payload for VAD.")
                                    elif not detector: logger.warning(f"{task_name_rd} Detector unavailable for realtime_input.")
                                else: logger.error(f"{task_name_rd} 'realtime_input' payload was not a dict with 'media_chunks': {type(realtime_payload)}")
                        
                        else: 
                            logger.warning(f"{task_name_rd} ({client_addr}, OpSess:{current_op_session_id}): Unhandled msg_type '{msg_type}': {str(data)[:200]}")
                    
                    except json.JSONDecodeError: logger.error(f"{task_name_rd} Non-JSON message received: {message_str[:100]}...")
                    except websockets.exceptions.ConnectionClosed: logger.warning(f"{task_name_rd} Connection closed during message processing loop."); raise 
                    except Exception as e_inner_loop: logger.exception(f"{task_name_rd} Error processing message (type '{msg_type}'): {e_inner_loop}");
            
            except asyncio.CancelledError: logger.info(f"{task_name_rd} Task cancelled."); raise
            except websockets.exceptions.ConnectionClosed as e_conn_closed: logger.info(f"{task_name_rd} Connection closed by client (Code: {e_conn_closed.code}, Reason: '{e_conn_closed.reason}')."); raise 
            except Exception as e_outer_loop: logger.exception(f"{task_name_rd} Unhandled error in main receive loop: {e_outer_loop}")
            finally: logger.info(f"{task_name_rd} Exiting.")

        # Create and manage tasks
        receive_task = asyncio.create_task(receive_data_from_client(), name=f"ReceiveDataTask_{handler_instance_id}")
        speech_task = asyncio.create_task(detect_speech_segments(), name=f"SpeechDetectTask_{handler_instance_id}")
        keepalive_task = asyncio.create_task(send_keepalive(), name=f"KeepaliveTask_{handler_instance_id}")
        handler_tasks.extend([receive_task, speech_task, keepalive_task])
        
        logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): All main tasks started. Waiting for first completion.")
        done, pending = await asyncio.wait(handler_tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # ... (Rest of the task cancellation and cleanup logic from your provided handle_client) ...
        completed_task_names = [t.get_name() for t in done if hasattr(t, 'get_name') and t.get_name()]
        logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): First task(s) completed: {completed_task_names}. Cancelling {len(pending)} pending tasks.")
        for task in pending:
            if not task.done(): task_name_cancel = task.get_name() if hasattr(task, 'get_name') else "UnknownTask"; logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Cancelling pending task: {task_name_cancel}"); task.cancel()
        
        logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Gathering results of all handler tasks.")
        gathered_results = await asyncio.gather(*handler_tasks, return_exceptions=True) 
        for idx, result in enumerate(gathered_results):
            task_name_from_list = handler_tasks[idx].get_name() if hasattr(handler_tasks[idx], 'get_name') else f"Task_{idx}"
            if isinstance(result, asyncio.CancelledError): logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Task '{task_name_from_list}' confirmed cancelled (expected).")
            elif isinstance(result, Exception):
                if not isinstance(result, websockets.exceptions.ConnectionClosed): logger.error(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Task '{task_name_from_list}' exited with exception: {result}", exc_info=result)
                else: logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Task '{task_name_from_list}' ended with ConnectionClosed: {getattr(result, 'reason', '')} (Code: {getattr(result, 'code', 'N/A')})")

    # This is the except and finally for the try block at the START of handle_client
    except Exception as top_level_handler_error: 
        logger.exception(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Top-level exception for session '{active_client_session_id}': {top_level_handler_error}")
    finally: 
        logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Entering FINALLY block for session '{active_client_session_id}'.")
        if detector: 
            logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Cleaning up AudioSegmentDetector tasks for VAD instance {detector.id_for_log}.")
            await detector.cancel_current_tasks()
        
        for task in handler_tasks: # Ensure all tasks created by this handler are cancelled
            if not task.done():
                task_name_final_cancel = task.get_name() if hasattr(task, 'get_name') else "UnknownTaskInFinally"
                logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Ensuring final cancellation of task in handler's finally: {task_name_final_cancel}")
                task.cancel()
        # Await them one last time
        if handler_tasks: # Only gather if tasks were actually created
            await asyncio.gather(*handler_tasks, return_exceptions=True)

        if websocket and websocket.state == WebSocketConnectionState.OPEN: 
            try:
                logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Websocket state OPEN. Attempting to close gracefully.")
                await websocket.close(code=1000, reason="Server handler: graceful shutdown")
            except websockets.exceptions.ConnectionClosed: 
                logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Websocket already closed by the time of final explicit close.")
            except RuntimeError as e_runtime_close: 
                logger.warning(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Runtime error during final explicit websocket close: {e_runtime_close}")
            except Exception as e_final_close: 
                logger.error(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Unexpected error during final explicit websocket close: {e_final_close}", exc_info=True)
        
        if gemma_processor_instance: 
            logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Releasing Gemma processor instance {gemma_processor_instance.instance_id_log}.")
            # gemma_processor_instance = None # Python GC will handle it
        if gemini_processor_instance: # Check if it was ever initialized
            logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Releasing Gemini processor instance {gemini_processor_instance.instance_id_log}.")
            # gemini_processor_instance = None
        if detector: 
            logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Releasing AudioSegmentDetector instance {detector.id_for_log}.")
            # detector = None

        logger.info(f"SESS_HANDLER ({handler_instance_id}, {client_addr}): Disconnected & handler fully cleaned up for session '{active_client_session_id}'.")


async def main():
    global GLOBAL_WHISPER_TRANSCRIBER, GLOBAL_KOKORO_TTS_PROCESSOR
    global GLOBAL_GEMMA_MODEL, GLOBAL_GEMMA_PROCESSOR, GLOBAL_EMBEDDING_MODEL, GLOBAL_CHROMA_CLIENT
    global GLOBAL_SMOL_MODEL, GLOBAL_SMOL_TOKENIZER, GLOBAL_SUMMARIZER
    global GEMINI_API_KEY_STORE
    try:
        init_db()
        load_server_settings()
        logger.info("Pre-loading ALL shared models...")

        logger.info("Loading SentenceTransformer model globally for RAG...")
        try:
            GLOBAL_EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"GLOBAL SentenceTransformer model ('all-MiniLM-L6-v2') loaded on device: {GLOBAL_EMBEDDING_MODEL.device}. Max Seq Length: {GLOBAL_EMBEDDING_MODEL.max_seq_length}")
        except Exception as e:
            logger.exception("FATAL: Global SentenceTransformer model initialization failed.")
            return

        logger.info("Loading Whisper model globally...")
        _whisper_model_obj = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small", quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16 if (torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()) else torch.float16 if torch.cuda.is_available() else torch.float32), low_cpu_mem_usage=True, use_safetensors=True)
        _whisper_processor_obj = AutoProcessor.from_pretrained("openai/whisper-small")
        GLOBAL_WHISPER_TRANSCRIBER = WhisperTranscriber(model=_whisper_model_obj, processor=_whisper_processor_obj, load_new=False)
        logger.info("GLOBAL Whisper model and processor loaded.")

        logger.info("Loading Kokoro TTS model globally...")
        try: _kokoro_pipeline_obj = KPipeline(lang_code='a'); GLOBAL_KOKORO_TTS_PROCESSOR = KokoroTTSProcessor(pipeline_instance=_kokoro_pipeline_obj, load_new=False); logger.info("GLOBAL Kokoro TTS pipeline loaded.")
        except Exception as e: logger.exception("FATAL: Global Kokoro KPipeline initialization failed."); return

        logger.info("Loading Gemma model and processor globally...")
        _gemma_model_id = "unsloth/gemma-3-4b-it-qat-bnb-4bit"
        _gemma_load_dtype = torch.bfloat16 if (torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()) else (torch.float16 if torch.cuda.is_available() else torch.float32)
        try:
            GLOBAL_GEMMA_PROCESSOR = AutoProcessor.from_pretrained(_gemma_model_id, trust_remote_code=True)
            GLOBAL_GEMMA_MODEL = AutoModelForImageTextToText.from_pretrained(
                _gemma_model_id,
                torch_dtype=_gemma_load_dtype,
                device_map="auto", 
                trust_remote_code=True
            ).eval()
            logger.info(f"Gemma model loaded. Device: {GLOBAL_GEMMA_MODEL.device if hasattr(GLOBAL_GEMMA_MODEL, 'device') else 'N/A'}")
        except ValueError as e:
            if "Some modules are dispatched on the CPU or the disk" in str(e):
                logger.warning("Failed to load Gemma with device_map='auto' due to potential offloading restrictions with BNB. "
                             "Trying with explicit BNB config for offload.")
                try:
                    bnb_gemma_config = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4", 
                        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=_gemma_load_dtype,
                        llm_int8_enable_fp32_cpu_offload=True 
                    )
                    GLOBAL_GEMMA_MODEL = AutoModelForImageTextToText.from_pretrained(
                        _gemma_model_id, quantization_config=bnb_gemma_config,
                        torch_dtype=_gemma_load_dtype, device_map="auto", trust_remote_code=True
                    ).eval()
                    logger.info(f"Gemma model loaded WITH EXPLICIT BNB OFFLOAD CONFIG. Device: {GLOBAL_GEMMA_MODEL.device if hasattr(GLOBAL_GEMMA_MODEL, 'device') else 'N/A'}")
                except Exception as e2:
                    logger.exception(f"FATAL: Failed to load Gemma model even with explicit BNB offload config: {e2}")
                    return 
            else:
                logger.exception(f"FATAL: ValueError loading Gemma model: {e}")
                return
        except Exception as e: 
            logger.exception(f"FATAL: An unexpected error occurred loading Gemma model: {e}")
            return

        if not hasattr(GLOBAL_GEMMA_MODEL, 'generation_config'): GLOBAL_GEMMA_MODEL.generation_config = GenerationConfig()
        gen_cfg = GLOBAL_GEMMA_MODEL.generation_config; tokenizer = GLOBAL_GEMMA_PROCESSOR.tokenizer
        gen_cfg.max_new_tokens = 768; gen_cfg.no_repeat_ngram_size = 3; gen_cfg.repetition_penalty = 1.2
        gen_cfg.temperature = 0.3; gen_cfg.top_p = 0.8; gen_cfg.top_k = 40; gen_cfg.do_sample = True; gen_cfg.num_beams = 1
        if tokenizer.pad_token_id is None:
            eos_token_value = tokenizer.eos_token_id
            pad_id_to_set = eos_token_value[0] if isinstance(eos_token_value, list) and eos_token_value else (eos_token_value if isinstance(eos_token_value, int) else 0)
            tokenizer.pad_token_id = pad_id_to_set
        gen_cfg.pad_token_id = tokenizer.pad_token_id
        tokenizer_eos_val = tokenizer.eos_token_id; final_eos_ids_list = []
        if isinstance(tokenizer_eos_val, int): final_eos_ids_list = [tokenizer_eos_val]
        elif isinstance(tokenizer_eos_val, list) and all(isinstance(eid, int) for eid in tokenizer_eos_val) and tokenizer_eos_val: final_eos_ids_list = tokenizer_eos_val
        else: model_config_eos_id = getattr(GLOBAL_GEMMA_MODEL.config, "eos_token_id", None); final_eos_ids_list = [model_config_eos_id] if isinstance(model_config_eos_id, int) else (model_config_eos_id if isinstance(model_config_eos_id, list) else [])
        try: eot_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>");_ = isinstance(eot_token_id, int) and eot_token_id not in final_eos_ids_list and final_eos_ids_list.append(eot_token_id)
        except: pass
        if not final_eos_ids_list and tokenizer.eos_token:
             try: eos_id_from_string = tokenizer.convert_tokens_to_ids(tokenizer.eos_token); isinstance(eos_id_from_string, int) and final_eos_ids_list.append(eos_id_from_string)
             except: pass
        if not final_eos_ids_list: final_eos_ids_list = [tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 106] 
        processed_eos_ids = []; [processed_eos_ids.extend(i for i in item if isinstance(i, int)) if isinstance(item, list) else (processed_eos_ids.append(item) if isinstance(item,int) else None) for item in final_eos_ids_list]
        gen_cfg.eos_token_id = sorted(list(set(processed_eos_ids)))
        if not gen_cfg.eos_token_id: logger.warning("EOS token ID list became empty, defaulting to [106]"); gen_cfg.eos_token_id = [106]
        if tokenizer.bos_token_id is not None: gen_cfg.bos_token_id = tokenizer.bos_token_id
        else: gen_cfg.bos_token_id = getattr(GLOBAL_GEMMA_MODEL.config, "bos_token_id", 2)
        logger.info(f"GLOBAL Gemma model G.CFG: {str(gen_cfg).replace(chr(10),' ')}")

        logger.info("Loading Qwen2.5-0.5B-Instruct model globally for summarization...")
        try:
            smol_model_id = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
            GLOBAL_QWEN_TOKENIZER = AutoTokenizer.from_pretrained(smol_model_id)
            qwen_compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()) else torch.float16
            GLOBAL_QWEN_MODEL = AutoModelForCausalLM.from_pretrained(
                smol_model_id,
                torch_dtype=qwen_compute_dtype, 
                device_map="auto", 
                trust_remote_code=True 
            )
            logger.info(f"GLOBAL Qwen2.5-0.5B-Instruct model and tokenizer loaded. Device: {GLOBAL_SMOL_MODEL.device if hasattr(GLOBAL_SMOL_MODEL, 'device') else 'N/A'}")
            GLOBAL_SUMMARIZER = SummarizationProcessor(GLOBAL_QWEN_MODEL, GLOBAL_QWEN_TOKENIZER)
            logger.info("GLOBAL SummarizationProcessor initialized with Qwen2.5-0.5B-Instruct.")
        except Exception as e:
            logger.exception("FATAL: Global Qwen2.5-0.5B-Instruct model initialization failed.")
            GLOBAL_QWEN_MODEL = None; GLOBAL_QWEN_TOKENIZER = None; GLOBAL_SUMMARIZER = None
        
        logger.info("--- All shared models pre-loaded successfully (Summarizer might be None if Qwen failed) ---")
        addr, port = "0.0.0.0", 9073

        # Define allowed origins
        allowed_origins = [
    "http://localhost:5173",       
    "http://127.0.0.1:5173",     
    "https://zw70f854-5173.asse.devtunnels.ms",
    "http://zw70f854-5173.asse.devtunnels.ms",  # Add non-HTTPS variant
    "https://zw70f854-9073.asse.devtunnels.ms",
    "http://zw70f854-9073.asse.devtunnels.ms"   # Add non-HTTPS variant
]
        logger.info(f"Allowed WebSocket origins: {allowed_origins}")
        logger.info(f"Starting WebSocket server on {addr}:{port}")
        server = await websockets.serve(
            handle_client, 
            addr, 
            port,
            ping_interval=20,
            ping_timeout=60,
            origins=allowed_origins,
            max_size=2**20,
            max_queue=2**5,
            process_request=None,
            compression=None,
            subprotocols=None,
            server_header=None,
            ssl=None  # Add this to explicitly handle non-SSL connections
        )
            
        
        logger.info(f"WebSocket server RUNNING on {addr}:{port}")
        try: await asyncio.Future()
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError): logger.info("Server shutdown signal received...")
        finally:
            logger.info("Closing server...");
            if server: server.close(); await server.wait_closed()
            logger.info("Server closed.")
    except Exception as e:
        logger.exception(f"CRITICAL SERVER ERROR during startup or main run: {e}")

# --- TEST BLOCK ---
logger.info("--- RUNNING PRE-FLIGHT CLASS DEFINITION TEST ---")
if GEMINI_API_KEY_STORE: # Only test if API key is loaded/set
    try:
        test_processor = GeminiAPIProcessor(api_key=GEMINI_API_KEY_STORE)
        logger.info(f"Test_processor type: {type(test_processor)}")
        logger.info(f"Test_processor dir: {dir(test_processor)}")
        if hasattr(test_processor, 'generate_response_for_chat'):
            logger.info("SUCCESS: Test_processor HAS 'generate_response_for_chat' method.")
        else:
            logger.error("FAILURE: Test_processor MISSING 'generate_response_for_chat' method.")
    except Exception as e_test:
        logger.error(f"Test block error: {e_test}")
else:
    logger.warning("Skipping pre-flight class test: GEMINI_API_KEY_STORE is not set.")
logger.info("--- END PRE-FLIGHT CLASS DEFINITION TEST ---")
# --- END TEST BLOCK ---

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: logger.info("Application shutdown requested (KeyboardInterrupt).")
