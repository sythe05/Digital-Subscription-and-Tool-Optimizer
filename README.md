# Financial Optimization Dashboard: Local RAG Pipeline

An intelligent, privacy-first financial optimization engine designed to help users identify, analyze, and mitigate digital subscription bloat. The system ingest expense data, evaluates usage-based Return on Investment (ROI), detects functional redundancies, and automatically maps budget constraints against live-harvested open-source software alternatives using local semantic search and Large Language Model (LLM) synthesis.

---

## Architecture Overview

The system operates entirely on local infrastructure to ensure complete data privacy and zero external API runtime costs:

* **Dynamic Data Scraper (`Python`):** Connects to the GitHub API to dynamically harvest a matrix of open-source repositories tagged under production developer tools, self-hosted alternatives, databases, and collaboration platforms.
* **Vector Indexing Engine (`Weaviate`):** A containerized vector database that stores structural metadata (descriptions, tags, URLs, star counts) and executes semantic vector queries using deep text context matches.
* **Local Inference Synthesis (`Ollama` & `Llama 3`):** Processes retrieved software alternatives and shapes them into clean, high-density, 3-sentence cost-mitigation paths.
* **Interactive Terminal Shell:** Provides an active runtime loop allowing continuous, back-to-back problem inputs without database re-initialization.

---

## Prerequisites

Before setting up the project, ensure your local machine meets the following requirements:

* **Operating System:** Windows 10/11 (with WSL2 enabled), macOS, or Linux.
* **Python:** Version `3.10` or higher.
* **Docker:** Docker Desktop installed and running.

---

## Installation & Setup Guide

### Step 1: Install and Configure Docker Desktop

1. Download and install **Docker Desktop** for your operating system from the [Official Docker Website](https://www.docker.com/products/docker-desktop/).
2. Launch Docker Desktop and ensure the Docker engine is completely initialized (the status indicator in the bottom corner should turn green).
3. *(Windows Only)* Ensure that Docker is configured to use the WSL2 backend for optimal performance.

### Step 2: Clone and Configure the Project

Create your project directory and set up your configuration file:

```bash
mkdir financial-optimizer-rag
cd financial-optimizer-rag

```

Create a `docker-compose.yml` file in the root directory to manage your local infrastructure stack:

```yaml
version: '3.8'

services:
  weaviate:
    command:
    - --host
    - 0.0.0.0
    - --port
    - '8080'
    - --scheme
    - http
    image: cr.weaviate.io/semitechnologies/weaviate:1.24.1
    ports:
    - 8080:8080
    - 50051:50051
    volumes:
    - weaviate_data:/var/lib/weaviate
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: ''

  ollama:
    image: ollama/ollama:latest
    ports:
    - 11434:11434
    volumes:
    - ollama_data:/root/.ollama

volumes:
  weaviate_data:
  ollama_data:

```

### Step 3: Spin Up Infrastructure Containers

Run the following command to download the service images and spin up the databases in detached background mode:

```bash
docker compose up -d

```

Verify that both containers are active and healthy:

```bash
docker compose ps

```

### Step 4: Download the Local LLM Weights

Ollama requires explicit local ingestion of model weights before running inference. Pull the Llama 3 model into your container's registry:

```bash
docker compose exec ollama ollama pull llama3

```

To verify the model downloaded successfully, test the local endpoint:

```bash
curl http://localhost:11434/api/tags

```

*(You should see `"name":"llama3:latest"` in the JSON payload returned).*

### Step 5: Set Up the Python Environment

1. Initialize a localized virtual environment to isolate dependency packages:
```bash
python -m venv venv

```


2. Activate the virtual environment:
* **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
* **Linux/macOS:** `source venv/bin/activate`


3. Install the required external client packages:
```bash
pip install weaviate-client requests

```



---

## Running the Application

Launch the interactive terminal shell using the following command:

```bash
python optimizer_rag.py

```

### Expected Behavior:

1. On the first run, the script detects that the Weaviate collection is uninitialized. It automatically hits the GitHub API, harvests the software alternative matrices, generates structural vector metrics, and seeds the collection.
2. Once initialization completes, the shell presents an open interactive query line:
```text
============================================================
LOCAL SYSTEM READY: INTERACTIVE ANALYSIS MODE
============================================================
Type your software cost or subscription issue below.
Type 'exit' or 'quit' to close the program safely.

Enter your query: 

```



---

## Sample Queries for Testing

Test the vector mapping accuracy by passing constraints covering student deployment barriers, design software pricing overhead, or workspace coordination tooling:

* **Database & Hosting Optimization:**
> *We are looking to scale our mobile application backend, but the recurring monthly costs for Firebase authentication, Firestore reads/writes, and cloud functions are becoming unsustainable for our student budget.*


* **UI/UX Software Footprint Reduction:**
> *Our frontend design team is expanding, and the enterprise licensing fees for Figma are taking up too much of our annual toolkit budget. We need a collaborative, web-standards alternative we can self-host.*


* **Local Project Coordination Workspace:**
> *We are trying to organize code repositories and track team tasks for our upcoming B.Tech assignments. GitHub Pro and Jira are overkill. What are some self-hosted version control and Kanban board alternatives we can run locally?*



---

## Database Management & Maintenance

If you modify the search queries inside `fetch_software_alternatives_dataset()` to target different technology domains, you must clear out the old vector storage cache to force a fresh data build.

Wipe the internal docker volumes completely and reseed by executing:

```bash
# Bring down containers and destroy old volume blocks
docker compose down -v

# Restart fresh containers
docker compose up -d

# Execute runtime script to trigger new data ingestion
python optimizer_rag.py

```

---
