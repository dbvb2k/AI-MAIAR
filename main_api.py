import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import chromadb
import config
import utils
from loguru import logger
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import JSONResponse

# --- Pydantic Schemas ---
class VectorSearchRequest(BaseModel):
    query: str
    top_n: int = 3

class VectorSearchResult(BaseModel):
    ticketid: str
    application: str
    summary: str
    similarity: float
    meta: dict

class ClassifierRequest(BaseModel):
    query: str

class ClassifierResponse(BaseModel):
    prediction: str
    probabilities: Optional[dict] = None

class HealthResponse(BaseModel):
    embedding_model: bool
    vector_store: bool
    classifier: bool
    message: str

# --- FastAPI App ---
app = FastAPI(title="ITSM Ticket Search API")

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
# Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/frontend_config")
def frontend_config():
    llm_url = getattr(config, 'llm_api_url', 'http://localhost:8080/llm_explanation')
    if not llm_url.rstrip('/').endswith('/llm_explanation'):
        llm_url = llm_url.rstrip('/') + '/llm_explanation'

    # print(f"LLM URL: {llm_url}")
    return JSONResponse({
        "llm_api_url": llm_url
    })

# --- Global Objects ---
EMBEDDER = None
CHROMA_CLIENT = None
COLLECTION = None
EMBEDDINGS = None
METADATAS = None
CLF = None
VECTORIZER = None

# --- Startup Event ---
@app.on_event("startup")
def load_models():
    global EMBEDDER, CHROMA_CLIENT, COLLECTION, EMBEDDINGS, METADATAS, CLF, VECTORIZER
    try:
        logger.info("Loading embedding model...")
        EMBEDDER = utils.load_embedding_model()
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        EMBEDDER = None
    try:
        logger.info(f"Connecting to ChromaDB at {config.vector_store_path}...")
        CHROMA_CLIENT = chromadb.PersistentClient(path=config.vector_store_path)
        COLLECTION = CHROMA_CLIENT.get_or_create_collection("itsm_tickets")
        if COLLECTION.count() > 0:
            EMBEDDINGS, METADATAS = utils.load_chroma_embeddings(COLLECTION, batch_size=config.chroma_batch_size)
        else:
            EMBEDDINGS, METADATAS = None, None
    except Exception as e:
        logger.error(f"Failed to load ChromaDB vector store: {e}")
        COLLECTION = None
        EMBEDDINGS, METADATAS = None, None
    # Only load classifier if enabled in config
    if getattr(config, 'enable_classifier_ensemble', False):
        try:
            CLF, VECTORIZER = utils.load_classifier_and_vectorizer()
        except Exception as e:
            logger.error(f"Failed to load classifier/vectorizer: {e}")
            CLF, VECTORIZER = None, None
    else:
        CLF, VECTORIZER = None, None

# --- Health Endpoint ---
@app.get("/health", response_model=HealthResponse)
def health():
    status = HealthResponse(
        embedding_model=EMBEDDER is not None,
        vector_store=COLLECTION is not None and EMBEDDINGS is not None,
        classifier=(CLF is not None and VECTORIZER is not None) if getattr(config, 'enable_classifier_ensemble', False) else False,
        message="OK"
    )
    if not status.embedding_model:
        status.message = "Embedding model not loaded"
    elif not status.vector_store:
        status.message = "Vector store not loaded"
    elif getattr(config, 'enable_classifier_ensemble', False) and not status.classifier:
        status.message = "Classifier not loaded"
    elif not getattr(config, 'enable_classifier_ensemble', False):
        status.message = "Classifier ensemble disabled in config"
    return status

# --- Vector Search Endpoint ---
@app.post("/vector_search", response_model=List[VectorSearchResult])
def vector_search(req: VectorSearchRequest):
    if EMBEDDER is None or EMBEDDINGS is None or METADATAS is None:
        raise HTTPException(status_code=503, detail="Model or vector store not loaded.")
    try:
        results = utils.vector_search(
            req.query,
            EMBEDDER,
            EMBEDDINGS,
            METADATAS,
            config,
            top_n=req.top_n
        )
        return [VectorSearchResult(**r) for r in results]
    except Exception as e:
        logger.error(f"Vector search error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Vector search failed.")

# --- Classifier Endpoint ---
@app.post("/classifier", response_model=ClassifierResponse)
def classifier(req: ClassifierRequest):
    # Only allow if classifier ensemble is enabled
    if not getattr(config, 'enable_classifier_ensemble', False):
        raise HTTPException(status_code=404, detail="Classifier ensemble is disabled in config.")
    if CLF is None or VECTORIZER is None:
        raise HTTPException(status_code=503, detail="Classifier not loaded.")
    try:
        query_text = req.query.lower()
        X_query = VECTORIZER.transform([query_text])
        pred = CLF.predict(X_query)[0]
        if hasattr(CLF, "predict_proba"):
            proba = CLF.predict_proba(X_query)[0]
            classes = CLF.classes_
            prob_dict = {c: float(p) for c, p in zip(classes, proba)}
        else:
            prob_dict = None
        return ClassifierResponse(prediction=pred, probabilities=prob_dict)
    except Exception as e:
        logger.error(f"Classifier error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Classifier prediction failed.") 