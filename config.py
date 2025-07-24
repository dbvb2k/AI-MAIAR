from enum import Enum

class LLMTone(str, Enum):
    CONCISE = 'concise'
    TECHNICAL = 'technical'
    USER_FRIENDLY = 'user-friendly'

# Configuration for ITSM Embedding Pipeline
# This file is used to configure the AI MAIAR application.
# It is used to configure the data folder, log folder, file list, vector store type, vector store path, embedding model, chroma batch size, batch size for embedding, fields to embed, logging level, whether to store empty metadata fields, whether to enable classifier ensemble, and LLM configuration. 


# Path to the folder containing the Excel files
data_folder = 'data'

# Path to the log folder
log_folder = 'logs'

# Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
log_level = 'INFO' 

# List of Excel filenames to process (relative to data_folder)
file_list = [
    'GMSCRFDump.csv',
    'itsm_data_junk_8k.xlsx',
    'Itsm_full_data_1.xlsx',
    'OnsiteDCDDump.csv'
    # Add more files as needed
]

# Vector store type: 'chroma' or 'faiss'
vector_store_type = 'chroma'

# Path to the vector store folder
vector_store_path = 'datastore'

# Embedding model (HuggingFace model name)
embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'

# Chroma batch size
chroma_batch_size = 2000  # or any value <= 5461

# Batch size for embedding
batch_size = 64

# Fields to embed (auto-detect these fields in each file, case-insensitive, fuzzy match)
fields_to_embed = [
    'SLM',
    'Title',
    'Description',
    'Resolution Comments',
    'Classification',
    'Summary',
]

# Whether to store empty metadata fields (True) or omit them (False)
store_empty_metadata = True 

# Enable or disable classifier ensemble in query_search.py
enable_classifier_ensemble = True 

# LLM configuration
llm_provider = 'ollama'  # Options: 'ollama', 'openai', 'together', etc.
llm_endpoint = 'http://localhost:11434/api/generate'  # Ollama default endpoint
llm_api_key = ''  # For OpenAI, Together, etc. (not used for local Ollama)
llm_tone = LLMTone.CONCISE  # Default tone. Options: LLMTone.CONCISE, LLMTone.TECHNICAL, LLMTone.USER_FRIENDLY

# LLM API Related
llm_api_url = 'http://localhost:8080/llm_explanation' # Our LLM API URL

# LLM Model name
llm_model_name = 'llama3:8b'

# LLM health check config
llm_health_check_interval = 30  # seconds
llm_health_check_timeout = 3    # seconds
llm_health_check_endpoint = 'http://localhost:11434/api/tags'  # Ollama default health check endpoint