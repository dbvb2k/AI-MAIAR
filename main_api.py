import os
import traceback
import time
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
    start_time = time.time()
    utils.log_api_request("/frontend_config", "GET", {}, start_time)
    
    llm_url = getattr(config, 'llm_api_url', 'http://localhost:8080/llm_explanation')
    if not llm_url.rstrip('/').endswith('/llm_explanation'):
        llm_url = llm_url.rstrip('/') + '/llm_explanation'

    response_data = {"llm_api_url": llm_url}
    utils.log_api_response("/frontend_config", "GET", response_data, 200, start_time)
    return JSONResponse(response_data)

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
    
    # Setup logging
    utils.setup_logging()
    logger.info("Starting AI MAIAR application...")
    
    try:
        logger.info("Loading embedding model...")
        EMBEDDER = utils.load_embedding_model()
    except Exception as e:
        utils.log_error(e, "Failed to load embedding model")
        EMBEDDER = None
    
    try:
        logger.info(f"Connecting to ChromaDB at {config.vector_store_path}...")
        start_time = time.time()
        CHROMA_CLIENT = chromadb.PersistentClient(path=config.vector_store_path)
        COLLECTION = CHROMA_CLIENT.get_or_create_collection("itsm_tickets")
        duration = time.time() - start_time
        utils.log_performance("ChromaDB Connection", duration)
        
        if COLLECTION.count() > 0:
            EMBEDDINGS, METADATAS = utils.load_chroma_embeddings(COLLECTION, batch_size=config.chroma_batch_size)
        else:
            logger.warning("ChromaDB collection is empty")
            EMBEDDINGS, METADATAS = None, None
    except Exception as e:
        utils.log_error(e, "Failed to load ChromaDB vector store")
        COLLECTION = None
        EMBEDDINGS, METADATAS = None, None
    
    # Only load classifier if enabled in config
    if getattr(config, 'enable_classifier_ensemble', False):
        try:
            CLF, VECTORIZER = utils.load_classifier_and_vectorizer()
        except Exception as e:
            utils.log_error(e, "Failed to load classifier/vectorizer")
            CLF, VECTORIZER = None, None
    else:
        logger.info("Classifier ensemble disabled in config")
        CLF, VECTORIZER = None, None
    
    logger.info("Application startup completed")

# --- Health Endpoint ---
@app.get("/health", response_model=HealthResponse)
def health():
    start_time = time.time()
    utils.log_api_request("/health", "GET", {}, start_time)
    
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
    
    utils.log_api_response("/health", "GET", status.dict(), 200, start_time)
    return status

# --- Vector Search Endpoint ---
@app.post("/vector_search", response_model=List[VectorSearchResult])
def vector_search(req: VectorSearchRequest):
    start_time = time.time()
    utils.log_api_request("/vector_search", "POST", req.dict(), start_time)
    
    if EMBEDDER is None or EMBEDDINGS is None or METADATAS is None:
        error_msg = "Model or vector store not loaded."
        utils.log_error(Exception(error_msg), "Vector search endpoint")
        utils.log_api_response("/vector_search", "POST", {"error": error_msg}, 503, start_time)
        raise HTTPException(status_code=503, detail=error_msg)
    
    try:
        results = utils.vector_search(
            req.query,
            EMBEDDER,
            EMBEDDINGS,
            METADATAS,
            config,
            top_n=req.top_n
        )
        
        response_data = [VectorSearchResult(**r) for r in results]
        utils.log_api_response("/vector_search", "POST", {"results_count": len(response_data)}, 200, start_time)
        return response_data
        
    except Exception as e:
        utils.log_error(e, "Vector search endpoint", include_traceback=True)
        utils.log_api_response("/vector_search", "POST", {"error": str(e)}, 500, start_time)
        raise HTTPException(status_code=500, detail="Vector search failed.")

# --- Classifier Endpoint ---
@app.post("/classifier", response_model=ClassifierResponse)
def classifier(req: ClassifierRequest):
    start_time = time.time()
    utils.log_api_request("/classifier", "POST", req.dict(), start_time)
    
    # Only allow if classifier ensemble is enabled
    if not getattr(config, 'enable_classifier_ensemble', False):
        error_msg = "Classifier ensemble is disabled in config."
        utils.log_error(Exception(error_msg), "Classifier endpoint")
        utils.log_api_response("/classifier", "POST", {"error": error_msg}, 404, start_time)
        raise HTTPException(status_code=404, detail=error_msg)
    
    if CLF is None or VECTORIZER is None:
        error_msg = "Classifier not loaded."
        utils.log_error(Exception(error_msg), "Classifier endpoint")
        utils.log_api_response("/classifier", "POST", {"error": error_msg}, 503, start_time)
        raise HTTPException(status_code=503, detail=error_msg)
    
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
        
        response_data = ClassifierResponse(prediction=pred, probabilities=prob_dict)
        utils.log_classifier_prediction(req.query, pred, prob_dict, time.time() - start_time)
        utils.log_api_response("/classifier", "POST", {"prediction": pred}, 200, start_time)
        return response_data
        
    except Exception as e:
        utils.log_error(e, "Classifier endpoint", include_traceback=True)
        utils.log_api_response("/classifier", "POST", {"error": str(e)}, 500, start_time)
        raise HTTPException(status_code=500, detail="Classifier prediction failed.") 