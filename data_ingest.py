import os
import sys
import importlib
import pandas as pd
from tqdm import tqdm
from loguru import logger
from datetime import datetime
from fuzzywuzzy import process as fuzzy_process
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import config
import torch  # Add this import at the top
import re

# Optional: FAISS support
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Ensure log and vector store folders exist
os.makedirs(config.log_folder, exist_ok=True)
os.makedirs(config.vector_store_path, exist_ok=True)

# Setup logging
logfile = os.path.join(
    config.log_folder,
    f"itsm_embed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logger.remove()
logger.add(sys.stderr, level=config.log_level)
logger.add(logfile, level=config.log_level, format="{time} | {level} | {message}")

# Helper: Fuzzy column matching
def find_best_column(columns, target):
    match, score = fuzzy_process.extractOne(target, columns)
    return match if score > 70 else None

def normalize_field_name(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())

def extract_metadata_fields(row, col_map, config):
    # Candidate fields for TicketID and Application
    ticketid_candidates = [
        'IncidentID*+', 'Incident ID*+', 'ID', 'TicketID', 'Incident ID', 'Ref'
    ]
    app_candidates = [
        'Service*+', 'Service Category', 'Service', 'Application', 'Classification', 'Group'
    ]
    # Helper to get all non-empty values for candidates
    def get_concat_value(candidates):
        vals = []
        for cand in candidates:
            col = col_map.get(cand)
            val = str(row[col]).strip() if col and pd.notnull(row[col]) else ''
            if val:
                vals.append(val)
        return '/'.join(vals)
    # TicketID
    ticketid_val = get_concat_value(ticketid_candidates)
    # Application
    app_val = get_concat_value(app_candidates)
    # Handle config for empty/omit
    meta = {}
    if ticketid_val or config.store_empty_metadata:
        meta['TicketID'] = ticketid_val if ticketid_val else ''
    if app_val or config.store_empty_metadata:
        meta['Application'] = app_val if app_val else ''
    return meta

def load_and_preprocess(file_path, fields_to_embed):
    # Detect file type and load accordingly
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.csv':
            logger.info(f"Loading CSV file: {file_path}")
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1')
        elif ext in ['.xls', '.xlsx']:
            logger.info(f"Loading Excel file: {file_path}")
            df = pd.read_excel(file_path, engine=None)
        else:
            logger.error(f"Unsupported file type: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None
    # Fuzzy match columns
    col_map = {}
    for field in fields_to_embed:
        best_col = find_best_column(df.columns, field)
        col_map[field] = best_col
    # Also map all candidate fields for TicketID and Application
    for cand in ['IncidentID*+', 'Incident ID*+', 'ID', 'TicketID', 'Incident ID', 'Ref',
                 'Service*+', 'Service Category', 'Service', 'Application', 'Classification', 'Group']:
        if cand not in col_map:
            best_col = find_best_column(df.columns, cand)
            col_map[cand] = best_col
    # Build text for embedding and standardized metadata
    texts = []
    metadatas = []
    for idx, row in df.iterrows():
        parts = []
        # Standardized metadata
        meta = extract_metadata_fields(row, col_map, config)
        meta['file'] = os.path.basename(file_path)
        meta['row'] = int(idx)
        for field in fields_to_embed:
            col = col_map[field]
            val = str(row[col]).strip() if col and pd.notnull(row[col]) else ''
            parts.append(val)
            # Optionally, add embedded fields to metadata as well
            meta[field] = val
        # Preprocess: join, lower, strip, remove extra whitespace
        text = " ".join(parts)
        text = " ".join(text.split()).lower()
        texts.append(text)
        metadatas.append(meta)
    return texts, metadatas

# Embedding backend (decoupled for future flexibility)
def get_embedder(model_name):
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device for embedding: {device}")
    return SentenceTransformer(model_name, device=device)

def embed_batches(embedder, texts, batch_size):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[i:i+batch_size]
        emb = embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.extend(emb)
    return embeddings

# Vector store backend (ChromaDB default, FAISS optional)
def store_embeddings_chroma(embeddings, metadatas, persist_dir, batch_size=None):
    if batch_size is None:
        batch_size = getattr(config, 'chroma_batch_size', 2000)
    import chromadb
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("itsm_tickets")
    ids = [f"ticket_{i}" for i in range(len(embeddings))]
    total = len(embeddings)
    for i in range(0, total, batch_size):
        logger.info(f"Adding batch {i} to {min(i+batch_size, total)} to ChromaDB...")
        collection.add(
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
    # No need to call persist() in ChromaDB 1.x
    return len(ids)

def store_embeddings_faiss(embeddings, metadatas, persist_dir):
    import pickle
    import numpy as np
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype('float32'))
    # Save index and metadata
    faiss.write_index(index, os.path.join(persist_dir, "faiss.index"))
    with open(os.path.join(persist_dir, "faiss_meta.pkl"), "wb") as f:
        pickle.dump(metadatas, f)
    return len(metadatas)

def main():
    logger.info("Starting ITSM embedding pipeline")
    total_tickets = 0
    all_texts = []
    all_metas = []

    num_files = len(config.file_list)
    for idx, fname in enumerate(config.file_list, 1):
        file_path = os.path.join(config.data_folder, fname)
        progress_msg = f"Processing file {idx}/{num_files}: {file_path}"
        print(progress_msg)
        logger.info(progress_msg)
        result = load_and_preprocess(file_path, config.fields_to_embed)
        if result is None:
            logger.warning(f"Skipping file: {file_path}")
            continue
        texts, metas = result
        logger.info(f"Loaded {len(texts)} tickets from {fname}")
        all_texts.extend(texts)
        all_metas.extend(metas)
        total_tickets += len(texts)

    if not all_texts:
        logger.error("No tickets found. Exiting.")
        return

    logger.info(f"Total tickets to embed: {total_tickets}")

    # Embedding
    embedder = get_embedder(config.embedding_model)
    embeddings = embed_batches(embedder, all_texts, config.batch_size)
    logger.info(f"Generated {len(embeddings)} embeddings.")

    # Store in vector store
    if config.vector_store_type == 'chroma':
        n = store_embeddings_chroma(embeddings, all_metas, config.vector_store_path)
        logger.info(f"Stored {n} embeddings in ChromaDB at {config.vector_store_path}")
    elif config.vector_store_type == 'faiss':
        if not FAISS_AVAILABLE:
            logger.error("FAISS is not installed. Please install faiss-cpu.")
            return
        n = store_embeddings_faiss(embeddings, all_metas, config.vector_store_path)
        logger.info(f"Stored {n} embeddings in FAISS at {config.vector_store_path}")
    else:
        logger.error(f"Unknown vector store type: {config.vector_store_type}")
        return
    logger.info(f"Pipeline complete. Files processed: {len(config.file_list)}. Tickets embedded: {total_tickets}. Vector store: {config.vector_store_path}")
    print(f"\nSummary:\n  Files processed: {len(config.file_list)}\n  Tickets embedded: {total_tickets}\n  Vector store: {config.vector_store_path}\n  Log file: {logfile}")

if __name__ == "__main__":
    main() 