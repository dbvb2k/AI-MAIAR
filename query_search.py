import sys
import os
import numpy as np
from tqdm import tqdm
from loguru import logger
from fuzzywuzzy import process as fuzzy_process
from sentence_transformers import SentenceTransformer
import chromadb
import config
import re
import pandas as pd
from colorama import Fore, Style, init  # Add at the top
init(autoreset=True)  # Initialize colorama
import joblib

# Fields to try for application detection
DEBUG_MODE = False
APP_FIELDS = ['Service', 'Service*+', 'Application', 'Classification', 'Group']

def normalize_field_name(name):
    # Lowercase, remove spaces and special characters
    return re.sub(r'[^a-z0-9]', '', name.lower())

# Improved field matching: try exact normalized match first, then fuzzy
def find_field(meta_keys, candidates):
    norm_keys = {normalize_field_name(k): k for k in meta_keys}
    for cand in candidates:
        norm_cand = normalize_field_name(cand)
        if norm_cand in norm_keys:
            return norm_keys[norm_cand]
    # Fallback to fuzzy
    match, score = fuzzy_process.extractOne(candidates[0], meta_keys)
    return match if score > 70 else None

# Helper: Cosine similarity
def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load embeddings and metadata from ChromaDB in batches
def load_chroma_embeddings(collection, batch_size=2000):
    n = collection.count()
    all_embeddings = []
    all_metas = []
    for i in tqdm(range(0, n, batch_size), desc="Loading embeddings", unit="batch"):
        batch = collection.get(
            include=["embeddings", "metadatas"],
            offset=i,
            limit=min(batch_size, n - i)
        )
        all_embeddings.extend(batch["embeddings"])
        all_metas.extend(batch["metadatas"])
    return np.array(all_embeddings), all_metas

def get_first_nonblank(meta, candidates):
    for field in candidates:
        val = meta.get(field, '')
        if pd.notna(val) and str(val).strip():
            return val
    return ''

def dedup_slash(val):
    # Split by '/', remove duplicates, and join back
    parts = [v for v in str(val).split('/') if v]
    seen = set()
    deduped = []
    for p in parts:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    return '/'.join(deduped)

# Main search function
def main():
    import argparse
    parser = argparse.ArgumentParser(description="ITSM Ticket Similarity Search")
    parser.add_argument('--top_n', type=int, default=3, help='Number of top results to show (default: 3)')
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size for loading embeddings (default: 2000)')
    args = parser.parse_args()

    # Load embedding model
    logger.info(f"Loading embedding model: {config.embedding_model}")
    embedder = SentenceTransformer(config.embedding_model)

    # Optionally load classifier ensemble
    clf = None
    vectorizer = None
    if getattr(config, 'enable_classifier_ensemble', False):
        try:
            clf = joblib.load('models/rf_classifier.pkl')
            vectorizer = joblib.load('models/vectorizer.pkl')
            logger.info("Loaded classifier ensemble model and vectorizer.")
        except Exception as e:
            print("Warning: Could not load classifier ensemble model. Skipping classifier prediction.")
            logger.warning(f"Could not load classifier ensemble: {e}")
            clf = None
            vectorizer = None

    # Load ChromaDB collection
    logger.info(f"Loading ChromaDB vector store from: {config.vector_store_path}")
    client = chromadb.PersistentClient(path=config.vector_store_path)
    collection = client.get_or_create_collection("itsm_tickets")
    n = collection.count()
    logger.info(f"Collection contains {n} tickets.")
    if n == 0:
        print("No tickets found in vector store.")
        return

    # Load all embeddings and metadata in batches
    embeddings, metadatas = load_chroma_embeddings(collection, batch_size=args.batch_size)

    # Improved field detection
    meta_keys = metadatas[0].keys()
    ticketid_candidates = [
        'IncidentID*+', 'Incident ID*+', 'ID', 'TicketID', 'Incident ID', 'Ref'
    ]
    app_candidates = [
        'Service*+', 'Service Category', 'Service', 'Application', 'Classification', 'Group'
    ]
    summary_candidates = ['Summary', 'Title', 'Description']

    ticketid_field = find_field(meta_keys, ticketid_candidates)
    app_field = find_field(meta_keys, app_candidates)
    summary_field = find_field(meta_keys, summary_candidates)

    if not ticketid_field:
        print("Warning: Could not find Ticket ID field in metadata.")
    if not app_field:
        print("Warning: Could not find Application field in metadata.")
    if not summary_field:
        print("Warning: Could not find Summary/Title/Description field in metadata.")

    # Query loop
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        if not query:
            print("No query entered. Please try again.")
            continue
        # Embed the query
        print(f"\n{Fore.YELLOW}Input Query: {query}{Style.RESET_ALL}\n")
        query_emb = embedder.encode([query], normalize_embeddings=True)[0]

        # Compute cosine similarity in batches
        logger.info("Computing cosine similarities...")
        sims = []
        for i in tqdm(range(0, len(embeddings), args.batch_size), desc="Similarity", unit="batch"):
            batch_emb = embeddings[i:i+args.batch_size]
            batch_sims = np.dot(batch_emb, query_emb) / (np.linalg.norm(batch_emb, axis=1) * np.linalg.norm(query_emb) + 1e-10)
            sims.extend(batch_sims)
        sims = np.array(sims)
        # Get top-N results
        top_n = min(args.top_n, len(sims))
        top_idx = np.argsort(sims)[-top_n:][::-1]
        # Get top-N unique (Ticket ID, Application) results
        seen = set()
        unique_results = []
        for rank, idx in enumerate(top_idx, 1):
            meta = metadatas[idx]
            if rank == 1 and DEBUG_MODE:
                print(f"DEBUG META: {meta}")  # Debug print for the top result
            
            # Robust value extraction
            def safe_val(field):
                val = meta.get(field, '') if field else ''
                if pd.isna(val):
                    return ''
                return val

            summary = safe_val(summary_field)

            # Ticket ID: show first non-blank from all candidates
            ticketid = safe_val(ticketid_field)
            if not ticketid:
                ticketid = get_first_nonblank(meta, ticketid_candidates)
            ticketid = dedup_slash(ticketid)

            # Application: show first non-blank from all candidates
            app = safe_val(app_field)
            if not app:
                app = get_first_nonblank(meta, app_candidates)
            app = dedup_slash(app)
            key = (ticketid, app)
            if key not in seen:
                seen.add(key)
                unique_results.append((idx, meta, ticketid, app, summary))
            if len(unique_results) == top_n:
                break

        # Classifier ensemble prediction
        if clf is not None and vectorizer is not None:
            # Use the same feature extraction as in training
            # (combine fields as in config.fields_to_embed)
            query_parts = [query]
            query_text = " ".join(query_parts).lower()
            X_query = vectorizer.transform([query_text])
            pred = clf.predict(X_query)[0]
            proba = max(clf.predict_proba(X_query)[0])
            print(f"\n{Fore.CYAN}Classifier prediction (Random Forest): {pred}{Style.RESET_ALL} (confidence: {proba:.2f})")

        print(f"\nTop {top_n} most similar tickets:")
        app_tally = {}
        for rank, (idx, meta, ticketid, app, summary) in enumerate(unique_results, 1):
            score = sims[idx]
            print(f"{rank}. Ticket ID: {ticketid}\n   Application: {app}\n   Summary: {summary}\n   Similarity: {score:.4f}\n")
            app_tally[app] = app_tally.get(app, 0) + 1

        # Determine most likely application
        if app_tally:
            likely_app = max(app_tally.items(), key=lambda x: x[1])[0]
            print(f"Most likely application (by top-{top_n} tally): {likely_app}")
        else:
            print("Could not determine application from top results.")

if __name__ == "__main__":
    main() 