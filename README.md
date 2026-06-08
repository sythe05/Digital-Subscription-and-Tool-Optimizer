# Financial Optimization Dashboard: Local RAG Pipeline

An intelligent, privacy-first financial optimization engine designed to help users identify, analyze, and mitigate digital subscription bloat. The system ingests expense data, evaluates usage-based Return on Investment (ROI), detects functional redundancies, and automatically maps budget constraints against live-harvested open-source software alternatives using local semantic search and Large Language Model (LLM) synthesis.

## 🏗️ Architecture Overview

The system operates entirely on local infrastructure to ensure complete data privacy and zero external API runtime costs:

* **Dynamic Data Scraper:** Connects to the GitHub API to dynamically harvest a matrix of open-source repositories tagged under production developer tools, self-hosted alternatives, databases, and collaboration platforms.
* **Vector Indexing Engine (`Weaviate`):** A containerized vector database running locally that stores structural metadata and executes semantic vector queries using the `BAAI/bge-m3` embedding model.
* **Local Inference Synthesis (`Ollama` & `Llama 3`):** Processes retrieved software alternatives and shapes them into detailed, step-by-step migration guides using local LLM inference.
* **Interactive Terminal Shell:** Provides an active runtime loop allowing continuous, back-to-back problem inputs without database re-initialization.

---

## ⚙️ Prerequisites

Before setting up the project, ensure your local machine meets the following requirements:
* **Operating System:** Windows 10/11 (with WSL2 enabled), macOS, or Linux.
* **Python:** Version `3.10` or higher.
* **Docker:** Docker Desktop installed and running.

---

## 🚀 Installation & Setup Guide

### Step 1: Clone the Repository
Clone this project to your local machine and navigate into the directory:
```bash
git clone [https://github.com/sythe05/Digital-Subscription-and-Tool-Optimizer.git](https://github.com/sythe05/Digital-Subscription-and-Tool-Optimizer.git)
cd Digital-Subscription-and-Tool-Optimizer

```

### Step 2: Environment Setup & Dependencies

It is highly recommended to use a virtual environment to isolate the project packages.

1. Create and activate a virtual environment:
* **Windows:** `python -m venv venv` and then `.\venv\Scripts\activate`
* **macOS/Linux:** `python3 -m venv venv` and then `source venv/bin/activate`


2. Install the strict dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt

```



### Step 3: Configure API Keys (Important)

Open the main Python script and update the environment variables with your personal access tokens. **Do not commit your actual tokens to version control.**

```python
GITHUB_TOKEN = "your_github_personal_access_token"
HF_TOKEN = "your_hugging_face_token"

```

### Step 4: Spin Up Infrastructure Containers

The `docker-compose.yml` file is pre-configured with secure API keys and specific DNS overrides to ensure smooth model downloads. Spin up the Weaviate and Ollama databases in detached mode:

```bash
docker compose up -d

```

### Step 5: Download the Local LLM Weights

Ollama requires explicit local ingestion of model weights before running inference. Pull the Llama 3 model directly into your container's registry:

```bash
docker compose exec ollama ollama pull llama3

```

*(Wait for the download to reach 100% and output "success").*

---

## 💻 Running the Application

Launch the interactive terminal shell using the following command:

```bash
python optimizer_rag.py

```

### Expected Behavior:

1. **Initial Run:** The script detects an empty Weaviate collection. It automatically connects to the GitHub API, harvests software alternatives based on targeted queries, vectorizes the descriptions, and seeds the local database.
2. **Interactive Loop:** Once initialization is complete, the shell opens for querying:
```text
============================================================
LOCAL SYSTEM READY: INTERACTIVE ANALYSIS MODE
============================================================
Type your software cost or subscription issue below.
Type 'exit' or 'quit' to close the program safely.

Enter your query: 

```



---

## 🧪 Sample Queries for Testing

Test the vector mapping accuracy by pasting these scenarios into the terminal:

* **Database & Hosting Optimization:**
> *"Our software development team is paying high recurring fees for centralized cloud-hosted databases, user authentication, and realtime syncing software tools."*


* **Local Project Coordination Workspace:**
> *"We are trying to organize code repositories and track team tasks for our upcoming B.Tech assignments. GitHub Pro and Jira are overkill. What are some self-hosted version control and Kanban board alternatives we can run locally?"*



---

## 🧹 Database Management & Maintenance

If you modify the search queries inside `fetch_software_alternatives_dataset()` to target different technology domains, you must clear out the old vector storage cache to force a fresh data build.

**⚠️ Important:** The `-v` flag in the teardown command destroys *all* attached volumes. This means it wipes the Weaviate data, but it *also* wipes the Ollama model weights. You must re-pull the Llama 3 model after running this.

Wipe the internal docker volumes completely and reseed by executing the following sequence:

```bash
# 1. Bring down containers and destroy old volume blocks (This deletes the Llama 3 model too!)
docker compose down -v

# 2. Restart fresh containers
docker compose up -d

# 3. Re-download the Llama 3 model weights
docker compose exec ollama ollama pull llama3

# 4. Run the script to trigger fresh vector data ingestion
python optimizer_rag.py

```

```

```
