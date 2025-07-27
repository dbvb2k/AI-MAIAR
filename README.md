# AI-Driven Multi-Application Incident Assignment & Resolution (AI MAIAR)

A sophisticated AI-powered system for intelligent incident ticket classification, search, and resolution using advanced natural language processing, vector embeddings, and large language models.

## 🚀 Features

- **Intelligent Ticket Classification**: Machine learning-based classification of ITSM tickets using ensemble methods
- **Semantic Search**: Advanced vector-based similarity search using ChromaDB and sentence transformers
- **LLM-Powered Explanations**: AI-generated explanations for classification decisions using configurable tone options
- **Real-time Health Monitoring**: Live status monitoring of all system components (embedding model, vector store, classifier, LLM API)
- **Comprehensive Logging**: Detailed logging with multiple levels, file rotation, and performance tracking
- **Modern Web Interface**: Responsive web UI with real-time updates and interactive features
- **Configurable Architecture**: Flexible configuration for different deployment scenarios

## 📋 Prerequisites

- Python 3.8+
- Ollama (for local LLM inference)
- Sufficient RAM for embedding models (recommended: 32GB+)
- Disk space for vector store and models
- GPU for faster inference

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI-Driven-MAIAR
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and configure Ollama**
   ```bash
   # Install Ollama (follow instructions at https://ollama.ai)
   # Pull the required model
   ollama pull llama3:8b
   ```

4. **Prepare your data**
   - Place your ITSM ticket data files in the `data/` directory
   - Supported formats: CSV, Excel (.xlsx)
   - Update `config.py` with your file list

## ⚙️ Configuration

### Main Configuration (`config.py`)

```python
# Data configuration
data_folder = 'data'
file_list = ['your_ticket_data.csv']

# Vector store configuration
vector_store_type = 'chroma'
vector_store_path = 'datastore'
embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'

# LLM configuration
llm_provider = 'ollama'
llm_model_name = 'llama3:8b'
llm_tone = LLMTone.CONCISE  # Options: CONCISE, TECHNICAL, USER_FRIENDLY

# Health check configuration
llm_health_check_interval = 30  # seconds
llm_health_check_timeout = 3    # seconds
```

### Environment Variables

Create a `.env` file in the project root:

```env
# API Keys (if using cloud LLM providers)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
PERPLEXITY_API_KEY=your_perplexity_key

# Ollama configuration (optional)
OLLAMA_BASE_URL=http://localhost:11434/api
```

## 🚀 Usage

### Starting the Application

1. **Start the main API server**
   ```bash
   python main_api.py
   ```
   The main server runs on `http://localhost:8000`

2. **Start the LLM API server** (in a separate terminal)
   ```bash
   python llm_api.py
   ```
   The LLM server runs on `http://localhost:8080`

3. **Access the web interface**
   Open your browser and navigate to `http://localhost:8000`

### Using the Web Interface

1. **Enter your query** in the text area
2. **Set the number of results** you want to see (Top-N)
3. **Click "Search"** to find similar tickets
4. **Select a tone** for the LLM explanation (Concise, Technical, or User-friendly)
5. **Click "Explain with LLM"** to get AI-generated explanations

## 📚 API Documentation

### Main API Endpoints

#### `GET /`
- **Description**: Web interface
- **Response**: HTML page

#### `GET /health`
- **Description**: System health status
- **Response**:
  ```json
  {
    "embedding_model": true,
    "vector_store": true,
    "classifier": true,
    "llm_api": true,
    "message": "OK"
  }
  ```

#### `POST /vector_search`
- **Description**: Semantic search for similar tickets
- **Request**:
  ```json
  {
    "query": "User query text",
    "top_n": 3
  }
  ```
- **Response**: Array of similar tickets with metadata

#### `POST /classifier`
- **Description**: Classify ticket into application categories
- **Request**:
  ```json
  {
    "query": "Ticket description"
  }
  ```
- **Response**:
  ```json
  {
    "prediction": "Application Name",
    "probabilities": {
      "App1": 0.8,
      "App2": 0.2
    }
  }
  ```

### LLM API Endpoints

#### `GET /llm_health`
- **Description**: LLM service health check
- **Response**:
  ```json
  {
    "ok": true
  }
  ```

#### `POST /llm_explanation`
- **Description**: Generate AI explanation for classification
- **Request**:
  ```json
  {
    "query": "User query",
    "classifier_prediction": "Predicted application",
    "top_n_results": [...],
    "tone": "concise"
  }
  ```
- **Response**:
  ```json
  {
    "explanation": "AI-generated explanation text"
  }
  ```

## 🏗️ Architecture

### System Components

1. **Frontend** (`templates/index.html`, `static/`)
   - React-like web interface
   - Real-time status monitoring
   - Interactive search and explanation features

2. **Main API** (`main_api.py`)
   - FastAPI-based REST API
   - Vector search functionality
   - Classifier integration
   - Health monitoring

3. **LLM API** (`llm_api.py`)
   - Dedicated LLM service
   - Ollama integration
   - Explanation generation

4. **Core Utilities** (`utils.py`)
   - Model loading and management
   - Vector search algorithms
   - Comprehensive logging system

5. **Configuration** (`config.py`)
   - Centralized configuration management
   - Environment-specific settings

### Data Flow

1. User submits query through web interface
2. Main API performs vector search and classification
3. Results are displayed to user
4. User requests LLM explanation
5. LLM API generates contextual explanation
6. Explanation is displayed with selected tone

## 📊 Logging

The application uses a comprehensive logging system with multiple levels:

- **Console Output**: Real-time logs with timestamps
- **File Logs**: Rotated log files with different levels
  - `app.log`: General application logs
  - `errors.log`: Error logs with stack traces
  - `performance.log`: Performance metrics

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General application information
- **WARNING**: Warning messages
- **ERROR**: Error messages with stack traces

### Performance Tracking

The system tracks:
- API request/response times
- LLM call durations
- Model loading times
- Vector search performance

## 🔧 Troubleshooting

### Common Issues

1. **LLM API Status Red**
   - Ensure Ollama is running: `ollama serve`
   - Check if model is downloaded: `ollama list`
   - Verify Ollama endpoint in config

2. **Vector Store Not Loading**
   - Check if `datastore/` directory exists
   - Verify data files are in `data/` directory
   - Check ChromaDB logs for errors

3. **Classifier Not Working**
   - Ensure `enable_classifier_ensemble = True` in config
   - Check if model files exist in `models/` directory
   - Verify training data format

4. **Embedding Model Issues**
   - Check internet connection for model download
   - Verify sufficient disk space
   - Check sentence-transformers installation

### Performance Optimization

1. **Reduce batch sizes** in config for memory-constrained environments
2. **Use smaller embedding models** for faster inference
3. **Adjust ChromaDB settings** for your data size
4. **Monitor memory usage** during large data processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black .
isort .

# Lint code
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Ollama](https://ollama.ai/) for local LLM inference
- [Loguru](https://loguru.readthedocs.io/) for logging

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the logs for error details

## 🔄 Version History

- **v1.0.0**: Initial release with core functionality
- **v1.1.0**: Added LLM explanations and health monitoring
- **v1.2.0**: Enhanced logging and UI improvements
- **v1.3.0**: Added tone selection and performance optimizations

---

**Note**: This is an AI-powered system designed for ITSM environments. Ensure proper data handling and security measures are in place for production deployments. 