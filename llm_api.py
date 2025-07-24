import os
import time
import json
import config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx
from loguru import logger
from fastapi.middleware.cors import CORSMiddleware
import utils
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    utils.setup_logging()
    logger.info("LLM API starting up...")
    yield

app = FastAPI(title="LLM Explanation API", lifespan=lifespan)

# Add CORS middleware for local frontend use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging on startup
# @app.on_event("startup")
# def startup_event():
#     utils.setup_logging()
#     logger.info("LLM API starting up...")

class LLMExplanationRequest(BaseModel):
    query: str
    classifier_prediction: str
    top_n_results: List[dict]
    tone: Optional[str] = None

class LLMExplanationResponse(BaseModel):
    explanation: str

# Helper: Build prompt for LLM
def build_prompt(query, classifier_prediction, top_n_results, tone):
    prompt = f"""
You are an expert ITSM assistant. Given the following user query, classifier prediction, and top-N similar tickets, explain in {tone or config.llm_tone.value} language why the predicted application is the most relevant.

User Query:
{query}

Classifier Prediction:
{classifier_prediction}

Top-N Similar Tickets:
"""
    for i, r in enumerate(top_n_results, 1):
        summary = r.get('summary', '')
        app = r.get('application', '')
        ticketid = r.get('ticketid', '')
        prompt += f"\n{i}. Ticket ID: {ticketid}\n   Application: {app}\n   Summary: {summary}"
    prompt += f"\n\nIn 5-6 sentences, explain why the predicted application is most relevant for this query."
    return prompt

# Helper: Call LLM (Ollama or API)
async def call_llm(prompt):
    start_time = time.time()
    provider = getattr(config, 'llm_provider', 'ollama')
    endpoint = getattr(config, 'llm_endpoint', 'http://localhost:11434/api/generate')
    api_key = getattr(config, 'llm_api_key', '')
    
    logger.info(f"Calling LLM provider: {provider} at {endpoint}")
    
    if provider == 'ollama':
        # Ollama API expects {model, prompt, stream: false}
        payload = {
            "model": config.llm_model_name,
            "prompt": prompt,
            "stream": False
        }
        
        logger.debug(f"Ollama payload: {json.dumps(payload, default=str)}")
        
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                resp = await client.post(endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()
                response_text = data.get('response', '').strip()
                
                duration = time.time() - start_time
                logger.info(f"Ollama LLM response received in {duration:.3f}s")
                logger.debug(f"Ollama response data: {json.dumps(data, default=str)}")
                
                return response_text
                
            except Exception as e:
                duration = time.time() - start_time
                utils.log_error(e, f"Ollama LLM call failed after {duration:.3f}s")
                raise HTTPException(status_code=502, detail=f"Ollama LLM error: {e}")
                
    elif provider == 'openai':
        # Placeholder for OpenAI API
        utils.log_error(Exception("OpenAI provider not implemented"), "LLM provider")
        raise HTTPException(status_code=501, detail="OpenAI provider not implemented yet.")
        
    elif provider == 'together':
        # Placeholder for Together API
        utils.log_error(Exception("Together provider not implemented"), "LLM provider")
        raise HTTPException(status_code=501, detail="Together provider not implemented yet.")
        
    else:
        utils.log_error(Exception(f"Unknown provider: {provider}"), "LLM provider")
        raise HTTPException(status_code=400, detail=f"Unknown LLM provider: {provider}")

@app.post("/llm_explanation", response_model=LLMExplanationResponse)
async def llm_explanation(req: LLMExplanationRequest):
    start_time = time.time()
    utils.log_api_request("/llm_explanation", "POST", req.dict(), start_time)
    
    try:
        prompt = build_prompt(req.query, req.classifier_prediction, req.top_n_results, req.tone)
        logger.debug(f"Generated prompt for LLM: {prompt[:500]}...")
        
        utils.log_llm_request(req.query, req.classifier_prediction, req.top_n_results, start_time)
        explanation = await call_llm(prompt)
        utils.log_llm_response(explanation, start_time)
        
        response_data = LLMExplanationResponse(explanation=explanation)
        utils.log_api_response("/llm_explanation", "POST", {"explanation_length": len(explanation)}, 200, start_time)
        return response_data
        
    except Exception as e:
        utils.log_error(e, "LLM explanation endpoint", include_traceback=True)
        utils.log_api_response("/llm_explanation", "POST", {"error": str(e)}, 500, start_time)
        raise HTTPException(status_code=500, detail=f"LLM explanation failed: {str(e)}") 

@app.get("/llm_health")
def llm_health():
    """Health check for LLM API: checks if Ollama is reachable and responsive."""
    import requests
    provider = getattr(config, 'llm_provider', 'ollama')
    # Use /api/tags for GET
    # endpoint = 'http://localhost:11434/api/tags'
    endpoint = getattr(config, 'llm_health_check_endpoint', 'http://localhost:11434/api/tags')
    if provider != 'ollama':
        return {"ok": False, "error": "Only Ollama health check implemented"}
    try:
        resp = requests.get(endpoint, timeout=3)
        resp.raise_for_status()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)} 