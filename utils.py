import re
import numpy as np
import pandas as pd
from fuzzywuzzy import process as fuzzy_process
from sentence_transformers import SentenceTransformer
import joblib
import config

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
    return SentenceTransformer(config.embedding_model)

def load_classifier_and_vectorizer():
    try:
        clf = joblib.load('models/rf_classifier.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        return clf, vectorizer
    except Exception:
        return None, None

# --- Vector Search Logic ---
def vector_search(query, embedder, embeddings, metadatas, config, top_n=3, batch_size=2000):
    # Embed the query
    query_emb = embedder.encode([query], normalize_embeddings=True)[0]
    # Compute cosine similarity in batches
    sims = []
    for i in range(0, len(embeddings), batch_size):
        batch_emb = embeddings[i:i+batch_size]
        batch_sims = np.dot(batch_emb, query_emb) / (np.linalg.norm(batch_emb, axis=1) * np.linalg.norm(query_emb) + 1e-10)
        sims.extend(batch_sims)
    sims = np.array(sims)
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
    return unique_results 

# --- ChromaDB Loading Utility ---
def load_chroma_embeddings(collection, batch_size=2000):
    n = collection.count()
    all_embeddings = []
    all_metas = []
    for i in range(0, n, batch_size):
        batch = collection.get(
            include=["embeddings", "metadatas"],
            offset=i,
            limit=min(batch_size, n - i)
        )
        all_embeddings.extend(batch["embeddings"])
        all_metas.extend(batch["metadatas"])
    return np.array(all_embeddings), all_metas 