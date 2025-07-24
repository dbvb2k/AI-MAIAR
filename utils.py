import re
import numpy as np
import pandas as pd
from fuzzywuzzy import process as fuzzy_process
from sentence_transformers import SentenceTransformer
import joblib
import config
import os
import time
from datetime import datetime
from loguru import logger
import json
from typing import Any, Dict, Optional

# --- Logging Setup ---
def setup_logging():
    """Setup comprehensive logging for the application"""
    # Remove default handler
    logger.remove()
    
    # Create logs directory if it doesn't exist
    os.makedirs(config.log_folder, exist_ok=True)
    
    # Console handler with timestamp
    def console_sink(message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message.record['message']}")
    
    logger.add(
        console_sink,
        level=config.log_level,
        format="{message}",
        colorize=True,
        enqueue=False
    )
    
    # File handler for all logs
    logger.add(
        os.path.join(config.log_folder, "app.log"),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=False
    )
    
    # Separate error log file
    logger.add(
        os.path.join(config.log_folder, "errors.log"),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=False
    )
    
    # Performance log file
    logger.add(
        os.path.join(config.log_folder, "performance.log"),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        filter=lambda record: "PERFORMANCE" in record["message"],
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=False
    )

def log_api_request(endpoint: str, method: str, request_data: Any, start_time: float):
    """Log API request details"""
    duration = time.time() - start_time
    logger.info(f"API_REQUEST | {method} {endpoint} | Duration: {duration:.3f}s")
    logger.debug(f"API_REQUEST_DATA | {method} {endpoint} | Request: {json.dumps(request_data, default=str, indent=2)}")

def log_api_response(endpoint: str, method: str, response_data: Any, status_code: int, start_time: float):
    """Log API response details"""
    duration = time.time() - start_time
    logger.info(f"API_RESPONSE | {method} {endpoint} | Status: {status_code} | Duration: {duration:.3f}s")
    logger.debug(f"API_RESPONSE_DATA | {method} {endpoint} | Response: {json.dumps(response_data, default=str, indent=2)}")

def log_llm_request(query: str, classifier_prediction: str, top_n_results: list, start_time: float):
    """Log LLM API request details"""
    duration = time.time() - start_time
    logger.info(f"LLM_REQUEST | Query: {query[:100]}... | Prediction: {classifier_prediction} | Duration: {duration:.3f}s")
    logger.debug(f"LLM_REQUEST_FULL | Query: {query} | Prediction: {classifier_prediction} | Top-N: {json.dumps(top_n_results, default=str, indent=2)}")

def log_llm_response(explanation: str, start_time: float):
    """Log LLM API response details"""
    duration = time.time() - start_time
    logger.info(f"LLM_RESPONSE | Explanation length: {len(explanation)} chars | Duration: {duration:.3f}s")
    logger.debug(f"LLM_RESPONSE_FULL | Explanation: {explanation}")

def log_error(error: Exception, context: str = "", include_traceback: bool = True):
    """Log error with context and optional traceback"""
    error_msg = f"ERROR | {context} | {type(error).__name__}: {str(error)}"
    if include_traceback:
        import traceback
        error_msg += f"\nTraceback:\n{traceback.format_exc()}"
    logger.error(error_msg)

def log_performance(operation: str, duration: float, details: Optional[Dict] = None):
    """Log performance metrics"""
    perf_msg = f"PERFORMANCE | {operation} | Duration: {duration:.3f}s"
    if details:
        perf_msg += f" | Details: {json.dumps(details, default=str)}"
    logger.info(perf_msg)

def log_model_loading(model_name: str, success: bool, duration: float = 0.0, error: Optional[str] = None):
    """Log model loading operations"""
    if success:
        logger.info(f"MODEL_LOADED | {model_name} | Duration: {duration:.3f}s")
    else:
        logger.error(f"MODEL_LOAD_FAILED | {model_name} | Error: {error}")

def log_vector_search(query: str, top_n: int, results_count: int, duration: float):
    """Log vector search operations"""
    logger.info(f"VECTOR_SEARCH | Query: {query[:100]}... | Top-N: {top_n} | Results: {results_count} | Duration: {duration:.3f}s")

def log_classifier_prediction(query: str, prediction: str, probabilities: Optional[Dict] = None, duration: float = 0.0):
    """Log classifier predictions"""
    logger.info(f"CLASSIFIER_PREDICTION | Query: {query[:100]}... | Prediction: {prediction} | Duration: {duration:.3f}s")
    if probabilities:
        logger.debug(f"CLASSIFIER_PROBABILITIES | {json.dumps(probabilities, indent=2)}")

# --- Field Matching Utilities ---
def normalize_field_name(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())

def find_field(meta_keys, candidates):
    norm_keys = {normalize_field_name(k): k for k in meta_keys}
    for cand in candidates:
        norm_cand = normalize_field_name(cand)
        if norm_cand in norm_keys:
            return norm_keys[norm_cand]
    # Fallback to fuzzy
    match, score = fuzzy_process.extractOne(candidates[0], meta_keys)
    return match if score > 70 else None

def get_first_nonblank(meta, candidates):
    for field in candidates:
        val = meta.get(field, '')
        if pd.notna(val) and str(val).strip():
            return val
    return ''

def dedup_slash(val):
    parts = [v for v in str(val).split('/') if v]
    seen = set()
    deduped = []
    for p in parts:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    return '/'.join(deduped)

# --- Model Loading ---
def load_embedding_model():
    start_time = time.time()
    try:
        logger.info(f"Loading embedding model: {config.embedding_model}")
        model = SentenceTransformer(config.embedding_model)
        duration = time.time() - start_time
        log_model_loading("Embedding Model", True, duration)
        return model
    except Exception as e:
        duration = time.time() - start_time
        log_model_loading("Embedding Model", False, duration, str(e))
        log_error(e, "Failed to load embedding model")
        return None

def load_classifier_and_vectorizer():
    start_time = time.time()
    try:
        logger.info("Loading classifier and vectorizer models")
        clf = joblib.load('models/rf_classifier.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        duration = time.time() - start_time
        log_model_loading("Classifier & Vectorizer", True, duration)
        return clf, vectorizer
    except Exception as e:
        duration = time.time() - start_time
        log_model_loading("Classifier & Vectorizer", False, duration, str(e))
        log_error(e, "Failed to load classifier and vectorizer")
        return None, None

# --- Vector Search Logic ---
def vector_search(query, embedder, embeddings, metadatas, config, top_n=3, batch_size=2000):
    start_time = time.time()
    logger.info(f"Starting vector search for query: {query[:100]}...")
    
    # Embed the query
    embed_start = time.time()
    query_emb = embedder.encode([query], normalize_embeddings=True)[0]
    embed_duration = time.time() - embed_start
    logger.debug(f"Query embedding completed in {embed_duration:.3f}s")
    
    # Compute cosine similarity in batches
    sim_start = time.time()
    sims = []
    for i in range(0, len(embeddings), batch_size):
        batch_emb = embeddings[i:i+batch_size]
        batch_sims = np.dot(batch_emb, query_emb) / (np.linalg.norm(batch_emb, axis=1) * np.linalg.norm(query_emb) + 1e-10)
        sims.extend(batch_sims)
    sims = np.array(sims)
    sim_duration = time.time() - sim_start
    logger.debug(f"Similarity computation completed in {sim_duration:.3f}s")
    # Get top-N unique (Ticket ID, Application) results
    ticketid_candidates = [
        'IncidentID*+', 'Incident ID*+', 'ID', 'TicketID', 'Incident ID', 'Ref'
    ]
    app_candidates = [
        'Service*+', 'Service Category', 'Service', 'Application', 'Classification', 'Group'
    ]
    summary_candidates = ['Summary', 'Title', 'Description']
    meta_keys = metadatas[0].keys()
    ticketid_field = find_field(meta_keys, ticketid_candidates)
    app_field = find_field(meta_keys, app_candidates)
    summary_field = find_field(meta_keys, summary_candidates)
    seen = set()
    unique_results = []
    top_idx = np.argsort(sims)[-top_n*5:][::-1]  # search deeper for uniqueness
    for idx in top_idx:
        meta = metadatas[idx]
        def safe_val(field):
            val = meta.get(field, '') if field else ''
            if pd.isna(val):
                return ''
            return val
        summary = safe_val(summary_field)
        ticketid = safe_val(ticketid_field)
        if not ticketid:
            ticketid = get_first_nonblank(meta, ticketid_candidates)
        ticketid = dedup_slash(ticketid)
        app = safe_val(app_field)
        if not app:
            app = get_first_nonblank(meta, app_candidates)
        app = dedup_slash(app)
        key = (ticketid, app)
        if key not in seen:
            seen.add(key)
            unique_results.append({
                'ticketid': ticketid,
                'application': app,
                'summary': summary,
                'similarity': float(sims[idx]),
                'meta': meta
            })
        if len(unique_results) == top_n:
            break
    
    total_duration = time.time() - start_time
    log_vector_search(query, top_n, len(unique_results), total_duration)
    logger.debug(f"Vector search completed. Found {len(unique_results)} unique results in {total_duration:.3f}s")
    
    return unique_results 

# --- ChromaDB Loading Utility ---
def load_chroma_embeddings(collection, batch_size=2000):
    start_time = time.time()
    n = collection.count()
    logger.info(f"Loading ChromaDB embeddings: {n} total documents, batch_size={batch_size}")
    
    all_embeddings = []
    all_metas = []
    for i in range(0, n, batch_size):
        batch_start = time.time()
        batch = collection.get(
            include=["embeddings", "metadatas"],
            offset=i,
            limit=min(batch_size, n - i)
        )
        all_embeddings.extend(batch["embeddings"])
        all_metas.extend(batch["metadatas"])
        batch_duration = time.time() - batch_start
        logger.debug(f"Loaded batch {i//batch_size + 1}/{(n-1)//batch_size + 1} ({len(batch['embeddings'])} docs) in {batch_duration:.3f}s")
    
    total_duration = time.time() - start_time
    logger.info(f"ChromaDB embeddings loaded: {len(all_embeddings)} embeddings, {len(all_metas)} metadata in {total_duration:.3f}s")
    log_performance("ChromaDB Loading", total_duration, {"total_docs": n, "batch_size": batch_size})
    
    return np.array(all_embeddings), all_metas 