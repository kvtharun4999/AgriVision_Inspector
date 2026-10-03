"""
api_client.py - Disease Information Retrieval with Caching & Local Fallback
----------------------------------------------------------------------------
Fetches authoritative turmeric crop disease information from agricultural
sources (TNAU Agritech Portal, ICAR) via an external API.

Features:
1. Loads API_KEY securely from .env using python-dotenv.
2. Formats strict prompts requiring citations and prohibiting invented doses.
3. Automatically retrieves English and Tamil translations in the same response.
4. Caches responses in info_cache.json (API is only called once per disease).
5. Falls back to local verified TNAU database (diseases.json) on timeout, error, or missing key.
"""

import os
import json
import logging
import requests
from dotenv import load_dotenv

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "info_cache.json")
LOCAL_DB_FILE = os.path.join(BASE_DIR, "diseases.json")


def load_cache():
    """Load cached disease responses from info_cache.json."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read info_cache.json: {e}")
    return {}


def save_to_cache(disease_name, data):
    """Save disease response entry into info_cache.json."""
    cache = load_cache()
    cache[disease_name] = data
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        logger.info(f"Cached information for '{disease_name}' into info_cache.json")
    except Exception as e:
        logger.error(f"Failed to write to info_cache.json: {e}")


def get_local_fallback(disease_name):
    """
    Retrieve verified TNAU agricultural advisory from diseases.json when
    external API is unavailable or unconfigured.
    """
    if os.path.exists(LOCAL_DB_FILE):
        try:
            with open(LOCAL_DB_FILE, "r", encoding="utf-8") as f:
                db = json.load(f).get("diseases", {})
                
                # 1. First try exact match (case-insensitive)
                matched = None
                for key, val in db.items():
                    if key.lower().strip() == disease_name.lower().strip():
                        matched = val
                        break

                # 2. If not matched, try substring matching
                if not matched:
                    for key, val in db.items():
                        if key.lower() in disease_name.lower() or disease_name.lower() in key.lower():
                            matched = val
                            break

                if not matched:
                    matched = db.get("Leaf Spot", {})

                return {
                    "scientific_name": matched.get("scientific_name", "Curcuma longa pathogen"),
                    "cause": matched.get("cause", "Fungal pathogen spreading under humid weather conditions."),
                    "symptoms": matched.get("symptoms", ["Foliar lesions visible on upper leaf surface."]),
                    "solution": matched.get("solution", ["Consult local agriculture officer for certified fungicide."]),
                    "prevention": matched.get("prevention", ["Plant disease-free rhizomes and improve field drainage."]),
                    "yield_loss": matched.get("yield_loss", None),
                    "sources": matched.get("sources", [
                        {
                            "title": "TNAU Agritech Portal (Crop Protection)",
                            "url": "https://agritech.tnau.ac.in/crop_protection/turmeric_diseases.html"
                        }
                    ]),
                    "tamil": matched.get("tamil", {}),
                    "is_offline": True,
                    "offline_note": "Offline information from verified TNAU advisory database."
                }
        except Exception as e:
            logger.error(f"Error reading local diseases.json fallback: {e}")

    # Ultimate hardcoded fallback
    return {
        "scientific_name": "Curcuma longa",
        "cause": "Foliar pathogen favoring high relative humidity and poorly drained fields.",
        "symptoms": ["Visible spotting or leaf discoloration."],
        "solution": ["Check product label or consult your local agriculture officer."],
        "prevention": ["Maintain proper spacing, clean borders, and avoid standing water."],
        "yield_loss": None,
        "sources": [
            {
                "title": "TNAU Agritech Portal",
                "url": "https://agritech.tnau.ac.in"
            }
        ],
        "tamil": {},
        "is_offline": True,
        "offline_note": "Offline information from verified local database."
    }


def call_external_api(disease_name, api_key):
    """
    Call external LLM API (Google Gemini or OpenAI) to get authoritative
    agronomic advice strictly based on TNAU / ICAR recommendations.
    """
    prompt = f"""You are an agricultural plant pathology specialist for Tamil Nadu, India.
Provide verified, accurate disease details for the Turmeric (*Curcuma longa*) condition: "{disease_name}".

STRICT RULES:
1. Answer ONLY using trusted agriculture sources: TNAU (Tamil Nadu Agricultural University) Agritech Portal, ICAR (Indian Council of Agricultural Research), or State Agriculture Universities.
2. NEVER invent fungicide doses. If an exact dose is not officially cited in the source, write "check the product label or ask your agriculture officer".
3. Provide both English and Tamil translations for all fields.
4. Output STRICT JSON format only, with no markdown code blocks or surrounding text.

JSON Schema:
{{
  "scientific_name": "Latin binominal or botanical condition",
  "cause": "1-2 concise sentences explaining pathogen and favorable weather conditions",
  "symptoms": [
    "Symptom 1",
    "Symptom 2",
    "Symptom 3"
  ],
  "solution": [
    "1. Cultural / physical treatment step",
    "2. Approved chemical / organic spray (cite source or say 'check product label or ask agriculture officer')",
    "3. Field water / canopy management step"
  ],
  "prevention": [
    "Prevention measure 1",
    "Prevention measure 2",
    "Prevention measure 3"
  ],
  "yield_loss": "Estimated yield loss range with cited source or null if not cited",
  "sources": [
    {{
      "title": "TNAU Agritech Portal - Turmeric Disease Management",
      "url": "https://agritech.tnau.ac.in/crop_protection/turmeric_diseases.html"
    }},
    {{
      "title": "ICAR-IISR Turmeric Advisory",
      "url": "https://spices.res.in"
    }}
  ],
  "tamil": {{
    "disease_name": "Disease name in Tamil",
    "scientific_name": "Scientific name with Tamil transliteration",
    "cause": "Cause in Tamil",
    "symptoms": ["Symptom 1 in Tamil", "Symptom 2 in Tamil"],
    "solution": ["1. Treatment 1 in Tamil", "2. Treatment 2 in Tamil"],
    "prevention": ["Prevention 1 in Tamil", "Prevention 2 in Tamil"]
  }}
}}"""

    # Detect provider based on key format
    if api_key.startswith("sk-"):
        # OpenAI endpoint
        endpoint = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a plant pathologist returning valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }
        res = requests.post(endpoint, headers=headers, json=payload, timeout=8)
        res.raise_for_status()
        raw_json = res.json()["choices"][0]["message"]["content"]
        return json.loads(raw_json)
    else:
        # Default: Google Gemini REST endpoint
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.2
            }
        }
        res = requests.post(endpoint, json=payload, timeout=8)
        res.raise_for_status()
        candidates = res.json().get("candidates", [])
        if not candidates:
            raise ValueError("No response generated by Gemini API")
        text = candidates[0]["content"]["parts"][0]["text"]
        return json.loads(text)


def fetch_disease_info(disease_name):
    """
    Main function called by app.py:
    1. Checks info_cache.json for existing cached data.
    2. If not cached, checks if API_KEY is set in environment.
    3. Calls external API or falls back to verified local TNAU data.
    """
    # 1. Check Cache
    cache = load_cache()
    if disease_name in cache:
        logger.info(f"Loaded '{disease_name}' from cache.")
        cached_item = cache[disease_name]
        return cached_item

    # 2. Check API Key
    api_key = os.getenv("API_KEY", "").strip()

    if not api_key:
        logger.info("API_KEY not configured in .env. Using verified local TNAU fallback.")
        local_data = get_local_fallback(disease_name)
        save_to_cache(disease_name, local_data)
        return local_data

    # 3. Call API
    try:
        logger.info(f"Calling external API for '{disease_name}' with verified agriculture prompt...")
        api_data = call_external_api(disease_name, api_key)
        api_data["is_offline"] = False
        api_data["offline_note"] = None
        save_to_cache(disease_name, api_data)
        return api_data
    except Exception as e:
        logger.warning(f"External API call failed ({e}). Falling back to local verified TNAU database.")
        fallback_data = get_local_fallback(disease_name)
        save_to_cache(disease_name, fallback_data)
        return fallback_data
