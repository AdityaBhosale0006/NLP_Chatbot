import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from googletrans import Translator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import asyncio
from asyncio import Semaphore
import time
import psutil
from datetime import datetime
import csv
import pandas as pd
from pathlib import Path
from fastapi.staticfiles import StaticFiles

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Metrics CSV file path
METRICS_CSV = DATA_DIR / "metrics_data.csv"

# Initialize CSV file with headers if it doesn't exist
if not METRICS_CSV.exists():
    with open(METRICS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp',
            'user_id',
            'uptime_seconds',
            'total_requests',
            'error_count',
            'error_rate',
            'avg_response_time',
            'memory_usage_mb',
            'cpu_percent',
            'active_connections'
        ])

# Performance metrics
class Metrics:
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        self.response_times = []
        self.active_connections = 0
    
    def add_request(self, response_time: float):
        self.request_count += 1
        self.response_times.append(response_time)
        # Keep only last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times.pop(0)
    
    def add_error(self):
        self.error_count += 1
    
    def increment_connections(self):
        self.active_connections += 1
    
    def decrement_connections(self):
        self.active_connections = max(0, self.active_connections - 1)
    
    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": (self.error_count / self.request_count * 100) if self.request_count > 0 else 0,
            "avg_response_time": avg_response_time,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "cpu_percent": psutil.cpu_percent(),
            "active_connections": self.active_connections
        }
    
    def save_to_csv(self, user_id=None):
        stats = self.get_stats()
        stats['user_id'] = user_id
        # Ensure user_id is the second column
        fieldnames = ['timestamp', 'user_id', 'uptime_seconds', 'total_requests', 'error_count', 'error_rate', 'avg_response_time', 'memory_usage_mb', 'cpu_percent', 'active_connections']
        with open(METRICS_CSV, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(stats)

metrics = Metrics()

# Load environment variables
load_dotenv()

# Initialize translator with connection pooling
translator = Translator()

# Rate limiting settings
MAX_REQUESTS_PER_MINUTE = 60
REQUEST_TIMEOUT = 30  # seconds

# Semaphore for rate limiting
request_semaphore = Semaphore(MAX_REQUESTS_PER_MINUTE)

# Request tracking
request_timestamps = []

# Supported languages
SUPPORTED_LANGUAGES = {
    # Indian Languages
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi',
    
    # International Languages
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'ar': 'Arabic',
    'tr': 'Turkish',
    'nl': 'Dutch',
    'pl': 'Polish',
    'vi': 'Vietnamese',
    'th': 'Thai',
    'id': 'Indonesian',
    'ms': 'Malay',
    'fa': 'Persian',
    'he': 'Hebrew',
    'da': 'Danish',
    'fi': 'Finnish',
    'sv': 'Swedish',
    'no': 'Norwegian',
    'cs': 'Czech',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'el': 'Greek',
    'uk': 'Ukrainian',
    'bg': 'Bulgarian',
    'hr': 'Croatian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'et': 'Estonian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'sr': 'Serbian',
    'bs': 'Bosnian',
    'mk': 'Macedonian',
    'ca': 'Catalan',
    'eu': 'Basque',
    'gl': 'Galician',
    'is': 'Icelandic',
    'ga': 'Irish',
    'cy': 'Welsh',
    'mt': 'Maltese',
    'sq': 'Albanian',
    'hy': 'Armenian',
    'ka': 'Georgian',
    'az': 'Azerbaijani',
    'uz': 'Uzbek',
    'kk': 'Kazakh',
    'ky': 'Kyrgyz',
    'tg': 'Tajik',
    'mn': 'Mongolian',
    'ne': 'Nepali',
    'si': 'Sinhala',
    'km': 'Khmer',
    'lo': 'Lao',
    'my': 'Burmese',
    'jw': 'Javanese',
    'su': 'Sundanese',
    'ceb': 'Cebuano',
    'hmn': 'Hmong',
    'haw': 'Hawaiian',
    'sm': 'Samoan',
    'st': 'Sesotho',
    'sn': 'Shona',
    'xh': 'Xhosa',
    'zu': 'Zulu',
    'af': 'Afrikaans',
    'sw': 'Swahili',
    'am': 'Amharic',
    'ha': 'Hausa',
    'yo': 'Yoruba',
    'ig': 'Igbo',
    'so': 'Somali',
    'mg': 'Malagasy',
    'ny': 'Chichewa',
    'co': 'Corsican',
    'fy': 'Frisian',
    'gd': 'Scots Gaelic',
    'ht': 'Haitian Creole',
    'ku': 'Kurdish',
    'lb': 'Luxembourgish',
    'mi': 'Maori',
    'ps': 'Pashto',
    'sd': 'Sindhi',
    'tt': 'Tatar',
    'ug': 'Uyghur',
    'yi': 'Yiddish'
}

# Get the base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to knowledge base JSON
DOCUMENT_PATH = os.path.join(BASE_DIR, "pccoe_knowledge.json")

# Image mapping for specific topics (just filenames)
IMAGE_MAPPING = {
    "campus": "campus.jpeg",
    "library": "library.jpeg",
    "admission": "admission_process.png",
    # Add more mappings as needed
}

# Greeting messages (in English)
GREETING_MESSAGES = [
    "Hello! I'm your AI assistant. How can I help you today?",
    "Hi there! I'm here to help. What would you like to know?",
    "Welcome! I'm your friendly AI chatbot. How may I assist you?",
    "Greetings! I'm ready to help. What's on your mind?",
]

# Follow-up messages (in English)
FOLLOW_UP_MESSAGES = [
    "Is there anything specific you'd like to know about?",
    "Feel free to ask me anything!",
    "I'm here to help with any questions you have.",
    "What would you like to learn more about?",
]

# Thank you messages (in English)
THANK_YOU_MESSAGES = [
    "You're welcome! Feel free to ask if you need anything else.",
    "Happy to help! Let me know if you have more questions.",
    "Glad I could assist! Don't hesitate to ask if you need anything else.",
    "Anytime! I'm here if you need more information.",
]

# No questions phrases (in English)
NO_QUESTIONS_PHRASES = [
    "no thanks",
    "no thank you",
    "that's all",
    "nothing else",
    "no more questions",
    "that's it",
    "nothing more",
    "no further questions",
    "no more",
    "all good",
    "that's all i need",
    "nothing else to ask",
]

class APIKeyManager:
    def __init__(self):
        self.encryption_key = os.getenv('ENCRYPTION_KEY')
        if not self.encryption_key:
            raise ValueError("ENCRYPTION_KEY environment variable is not set")
        # Ensure encryption key is 44 characters (base64-encoded 32 bytes)
        if len(self.encryption_key) != 44:
            raise ValueError("ENCRYPTION_KEY must be 44 characters (base64-encoded 32 bytes) for Fernet")
        self.fernet = Fernet(self.encryption_key.encode())
        
    def get_api_key(self) -> str:
        encrypted_api_key = os.getenv('API_KEY')
        if not encrypted_api_key:
            raise ValueError("API_KEY environment variable is not set")
        
        try:
            # Decrypt the API key
            decrypted_key = self.fernet.decrypt(encrypted_api_key.encode())
            return decrypted_key.decode()
        except Exception as e:
            logger.error(f"Error decrypting API key: {e}")
            raise ValueError("Invalid API key format")

# Initialize API key manager
try:
    api_key_manager = APIKeyManager()
except Exception as e:
    logger.error(f"Failed to initialize API key manager: {e}")
    raise

def verify_api_key(x_api_key: str = Header(...)):
    try:
        valid_api_key = api_key_manager.get_api_key()
        if x_api_key != valid_api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        return x_api_key
    except Exception as e:
        logger.error(f"API key verification failed: {e}")
        raise HTTPException(
            status_code=401,
            detail="Authentication failed"
        )

# Efficient NLTK data download

def safe_nltk_download(resource: str, download_name: str):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(download_name)

safe_nltk_download("tokenizers/punkt", "punkt")
safe_nltk_download("corpora/stopwords", "stopwords")

app = FastAPI()

# Mount static files for images
static_images_path = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_images_path), name="static")

# Load and preprocess documents
try:
    raw_text = Path(DOCUMENT_PATH).read_text()
    logger.info(f"Loaded document from {DOCUMENT_PATH}")
except FileNotFoundError:
    logger.error(f"Document file '{DOCUMENT_PATH}' not found.")
    raise Exception(f"Document file '{DOCUMENT_PATH}' not found. Please ensure the file exists.")
except Exception as e:
    logger.error(f"Error loading document: {e}")
    raise Exception(f"Error loading document: {e}")

# Chunking by paragraph, fallback to sentences if needed
def chunk_text(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    # fallback to sentences
    return sent_tokenize(text)

chunks: List[str] = chunk_text(raw_text)

# Load sentence transformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Loaded SentenceTransformer model.")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer: {e}")
    raise Exception(f"Error loading SentenceTransformer: {e}")

# Create embeddings for chunks
try:
    embeddings = model.encode(chunks)
    logger.info(f"Created embeddings for {len(chunks)} chunks.")
except Exception as e:
    logger.error(f"Error creating embeddings: {e}")
    raise Exception(f"Error creating embeddings: {e}")

# Setup FAISS index
try:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    logger.info("FAISS index created and embeddings added.")
except Exception as e:
    logger.error(f"Error setting up FAISS index: {e}")
    raise Exception(f"Error setting up FAISS index: {e}")

# Setup stopwords for text cleanup
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Clean and format text for better readability."""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Capitalize first letter of each sentence
    sentences = sent_tokenize(text)
    sentences = [s[0].upper() + s[1:] for s in sentences]
    return ' '.join(sentences)

class QueryRequest(BaseModel):
    query: str
    is_first_message: bool = False
    last_greeting: Optional[str] = None
    language: str = 'en'  # Default to English
    user_id: Optional[str] = None  # <-- Added user_id

def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target language."""
    try:
        if target_lang == 'en':
            return text
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text

def get_random_message(messages: List[str], exclude: Optional[str] = None, target_lang: str = 'en') -> str:
    """Get a random message from the provided list and translate it."""
    available_messages = [msg for msg in messages if msg != exclude]
    message = np.random.choice(available_messages)
    return translate_text(message, target_lang)

def get_relevant_image(query: str) -> Optional[str]:
    """Determine if an image is relevant to the query based on keywords."""
    query_lower = query.lower()
    for keyword, filename in IMAGE_MAPPING.items():
        if keyword in query_lower:
            return f"/static/{filename}"
    return None

def is_no_questions(query: str, target_lang: str = 'en') -> bool:
    """Check if the query indicates no further questions."""
    # Translate the query to English for checking
    if target_lang != 'en':
        try:
            query = translator.translate(query, dest='en').text
        except Exception as e:
            logger.error(f"Translation error in no_questions check: {e}")
    
    query_lower = query.lower()
    return any(phrase in query_lower for phrase in NO_QUESTIONS_PHRASES)

def get_top_k_matches(query: str, k: int = 3, target_lang: str = 'en') -> Dict[str, Any]:
    """Embed the query, search for top k similar chunks, and return results."""
    # Translate query to English for searching
    if target_lang != 'en':
        try:
            query = translator.translate(query, dest='en').text
        except Exception as e:
            logger.error(f"Translation error in query: {e}")
    
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    matches = [chunks[i] for i in I[0]]
    
    # Combine and clean the response
    combined = " ".join(matches)
    cleaned_response = clean_text(combined)
    
    # Translate response to target language
    if target_lang != 'en':
        cleaned_response = translate_text(cleaned_response, target_lang)
    
    # Check if an image is relevant
    relevant_image = get_relevant_image(query)
    
    return {
        "message": cleaned_response,
        "image": relevant_image
    }

async def rate_limit():
    """Rate limiting middleware"""
    current_time = time.time()
    # Remove timestamps older than 1 minute
    global request_timestamps
    request_timestamps = [ts for ts in request_timestamps if current_time - ts < 60]
    
    if len(request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    
    request_timestamps.append(current_time)
    return True

@app.get("/metrics")
async def get_metrics(user_id: Optional[str] = None):
    """Get current performance metrics and save to CSV"""
    metrics.save_to_csv(user_id=user_id)
    return metrics.get_stats()

@app.get("/metrics/analysis")
async def get_metrics_analysis():
    """Get basic analysis of collected metrics"""
    try:
        df = pd.read_csv(METRICS_CSV)
        analysis = {
            "total_requests": int(df["total_requests"].iloc[-1]),
            "total_errors": int(df["error_count"].iloc[-1]),
            "avg_error_rate": float(df["error_rate"].mean()),
            "avg_response_time": float(df["avg_response_time"].mean()),
            "max_memory_usage": float(df["memory_usage_mb"].max()),
            "peak_cpu_usage": float(df["cpu_percent"].max()),
            "peak_connections": int(df["active_connections"].max()),
            "data_points": len(df),
            "time_span_hours": (pd.to_datetime(df["timestamp"].iloc[-1]) - pd.to_datetime(df["timestamp"].iloc[0])).total_seconds() / 3600
        }
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing metrics: {e}")
        raise HTTPException(status_code=500, detail="Error analyzing metrics data")

@app.post("/query")
async def query_document(
    body: QueryRequest,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Query the document with connection tracking"""
    metrics.increment_connections()
    start_time = time.time()
    try:
        await rate_limit()
        async with request_semaphore:
            try:
                result = await asyncio.wait_for(
                    process_query(body),
                    timeout=REQUEST_TIMEOUT
                )
                metrics.add_request(time.time() - start_time)
                metrics.save_to_csv(user_id=body.user_id)  # <-- Save with user_id
                return result
            except asyncio.TimeoutError:
                metrics.add_error()
                metrics.save_to_csv(user_id=body.user_id)
                raise HTTPException(status_code=504, detail="Request timed out")
            except Exception as e:
                metrics.add_error()
                metrics.save_to_csv(user_id=body.user_id)
                logger.error(f"Error processing query: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        metrics.add_error()
        metrics.save_to_csv(user_id=body.user_id)
        raise e
    finally:
        metrics.decrement_connections()

async def process_query(body: QueryRequest) -> Dict[str, Any]:
    """Process the query with all the existing logic"""
    query = body.query.strip()
    target_lang = body.language.lower()
    
    # Validate language
    if target_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Supported languages: {', '.join(SUPPORTED_LANGUAGES.values())}"
        )
    
    # Handle first message
    if body.is_first_message:
        greeting = get_random_message(GREETING_MESSAGES, target_lang=target_lang)
        return {
            "message": greeting,
            "image": None
        }
    
    if not query:
        logger.warning("Received empty query.")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Check if user indicates no further questions
        if is_no_questions(query, target_lang):
            return {
                "message": get_random_message(THANK_YOU_MESSAGES, target_lang=target_lang),
                "image": None
            }
        
        result = get_top_k_matches(query, target_lang=target_lang)
        
        # Add follow-up message if it's a very short query
        if len(query.split()) <= 3:
            follow_up = get_random_message(FOLLOW_UP_MESSAGES, body.last_greeting, target_lang)
            result["message"] = f"{result['message']}\n\n{follow_up}"
        
        logger.info(f"Query processed: {query}")
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add CORS and security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "yourdomain.com"]  # Restrict trusted hosts
)

# ---
# For future: To handle large datasets, consider saving/loading FAISS index and embeddings to disk.
# Example (not implemented here):
# faiss.write_index(index, "faiss.index")
# np.save("embeddings.npy", embeddings)
# ---

# Add periodic logging of metrics
async def log_metrics():
    while True:
        metrics.save_to_csv(user_id=None)
        stats = metrics.get_stats()
        logger.info(f"Performance Metrics: {stats}")
        await asyncio.sleep(300)  # Log every 5 minutes

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(log_metrics())

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("chatbot.chatbot:app", host="0.0.0.0", port=port, reload=True)

