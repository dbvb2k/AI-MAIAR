# Configuration for ITSM Embedding Pipeline

# Path to the folder containing the Excel files
data_folder = 'data'

# Path to the log folder
log_folder = 'logs'

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

# Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
log_level = 'INFO' 

# Whether to store empty metadata fields (True) or omit them (False)
store_empty_metadata = True 

# Enable or disable classifier ensemble in query_search.py
enable_classifier_ensemble = True 

# LLM configuration
llm_provider = 'ollama'  # Options: 'ollama', 'openai', 'together', etc.
llm_endpoint = 'http://localhost:11434/api/generate'  # Ollama default endpoint
llm_api_key = ''  # For OpenAI, Together, etc. (not used for local Ollama)
llm_tone = 'concise'  # Options: 'concise', 'technical', 'user-friendly'

