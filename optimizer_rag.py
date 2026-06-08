import json
import requests
import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ==========================================
# CENTRALIZED ACCOUNT CREDENTIALS & ENDPOINTS
# ==========================================
GITHUB_TOKEN = ""  # Helps avoid GitHub API rate limits
HF_TOKEN = ""          # Required if accessing restricted/gated models
WEAVIATE_API_KEY = ""  # Must match docker-compose.yml!

# Infrastructure Endpoint Definitions
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
COLLECTION_NAME = "SubscriptionAlternative"

# ==========================================
# SYSTEM INITIALIZATION & CONNECTION
# ==========================================

print("Loading local BGE-M3 embedding model to system memory...")
# Initializes model locally. Leverages HF_TOKEN if provided for secure connections.
embedding_model = SentenceTransformer(
    'BAAI/bge-m3', 
    token=HF_TOKEN if HF_TOKEN != "" else None
)

print("Establishing authenticated hook to secured Weaviate node on localhost...")
# UPDATED: Weaviate v4 connection requires explicit grpc_port
client = weaviate.connect_to_local(
    port=8080,
    grpc_port=50051,
    auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
)

# ==========================================
# LIVE DATA HARVESTING (GITHUB API LAYER)
# ==========================================

def fetch_software_alternatives_dataset():
    """Queries public registries on GitHub for popular open-source software alternatives."""
    print("\nHarvesting live alternative software records via GitHub API...")
    url = "https://api.github.com/search/repositories"
    
    search_query = "topic:open-source-alternative OR topic:self-hosted-alternative OR alternative-to"
    params = {
        "q": search_query,
        "sort": "stars",
        "order": "desc",
        "per_page": 40
    }
    
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN and GITHUB_TOKEN != "":
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"Warning: GitHub API returned status {response.status_code}. Using emergency backup seed data.")
            return get_backup_seed_data()
            
        repo_data = response.json().get("items", [])
        parsed_knowledge_base = []
        
        for repo in repo_data:
            description = repo.get("description")
            if not description:
                continue
                
            parsed_knowledge_base.append({
                "title": repo.get("name"),
                "primary_tags": ", ".join(repo.get("topics", [])[:4]),
                "description": description,
                "source_url": repo.get("html_url"),
                "popularity_score": int(repo.get("stargazers_count", 0)),
                "is_open_source": True
            })
            
        print(f"Successfully harvested {len(parsed_knowledge_base)} verified software alternatives.")
        return parsed_knowledge_base
        
    except Exception as e:
        print(f"Network exception encountered during extraction: {e}. Falling back to backup database.")
        return get_backup_seed_data()

def get_backup_seed_data():
    """Fallback dataset to guarantee execution if external API request bounds fail."""
    return [
        {"title": "Supabase", "primary_tags": "firebase-alternative, database, auth", "description": "Open source Firebase alternative. High performance postgres database, authentication, instant APIs, edge functions and realtime subscriptions.", "source_url": "https://github.com/supabase/supabase", "popularity_score": 70000, "is_open_source": True},
        {"title": "NocoDB", "primary_tags": "airtable-alternative, spreadsheets, database", "description": "Turns any database into a smart spreadsheet. Free & open source alternative to Airtable.", "source_url": "https://github.com/nocodb/nocodb", "popularity_score": 45000, "is_open_source": True},
        {"title": "Penpot", "primary_tags": "figma-alternative, design, UI-UX", "description": "The first open source design and prototyping platform meant for cross-domain teams. Uses open web standards like SVG.", "source_url": "https://github.com/penpot/penpot", "popularity_score": 28000, "is_open_source": True}
    ]

# ==========================================
# VECTOR DATABASE SETUP & INGESTION
# ==========================================

def initialize_vector_schema_if_needed():
    """Configures internal cluster index metrics only if they do not already exist."""
    # UPDATED: Smart check to prevent wiping your database every run
    if client.collections.exists(COLLECTION_NAME):
        print(f"Verified: Existing index '{COLLECTION_NAME}' is healthy and loaded. Skipping data recreation.")
        return False
        
    print(f"Configuring new Weaviate vector schema for '{COLLECTION_NAME}'...")
    client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=None,  # Disables internal auto-vectorization; using external local BGE-M3
        properties=[
            wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="primary_tags", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="description", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="source_url", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="popularity_score", data_type=wvc.config.DataType.INT),
            wvc.config.Property(name="is_open_source", data_type=wvc.config.DataType.BOOL),
        ]
    )
    return True

def populate_vector_store(software_records):
    """Generates vectors locally and streams the metrics into the Weaviate container."""
    collection = client.collections.get(COLLECTION_NAME)
    
    print("Vectorizing data via BGE-M3 and indexing into Weaviate container...")
    for record in tqdm(software_records):
        context_string = f"Software: {record['title']}. Categorization: {record['primary_tags']}. Capabilities: {record['description']}"
        dense_vector = embedding_model.encode(context_string).tolist()
        
        collection.data.insert(
            properties=record,
            vector=dense_vector
        )
    print("Vector indices successfully constructed and populated.")

# ==========================================
# RAG PIPELINE EXECUTION ENGINE
# ==========================================

def run_optimization_query(expense_issue: str):
    """Executes a full local Retrieval-Augmented Generation loop using Docker infrastructure."""
    search_vector = embedding_model.encode(expense_issue).tolist()
    
    collection = client.collections.get(COLLECTION_NAME)
    retrieval_matches = collection.query.near_vector(
        near_vector=search_vector,
        limit=2,
        return_properties=["title", "primary_tags", "description", "source_url", "popularity_score"]
    )
    
    context_accumulator = ""
    for rank, item in enumerate(retrieval_matches.objects):
        attributes = item.properties
        context_accumulator += f"\n[Alternative Option #{rank + 1}]\n"
        context_accumulator += f"Product Name: {attributes['title']}\n"
        context_accumulator += f"Community Traction: {attributes['popularity_score']} Stars on GitHub\n"
        context_accumulator += f"Functional Utility: {attributes['description']}\n"
        context_accumulator += f"Access URL: {attributes['source_url']}\n"
        
    if not context_accumulator:
        context_accumulator = "No direct matching open-source solutions currently cataloged in the vector engine store."

    system_instruction = (
        "You are an advanced cost-optimization engineer specializing in digital asset management. "
        "Your role is to map user software spending problems to the extracted context data. "
        "Provide a concrete, highly objective 3-sentence mitigation path. Always name the alternative tool and its URL."
    )
    
    execution_prompt = f"""
Context from Verified Open-Source Registry Database:
{context_accumulator}

User Subscription Overhead/Problem Statement:
"{expense_issue}"

Analyze the parameters. Provide a specific workflow recommendation based only on the options provided. Include tracking metrics like repository star popularity to build user trust.
"""

    payload = {
        "model": "llama3",
        "prompt": f"<|system|>\n{system_instruction}\n<|user|>\n{execution_prompt}\n<|assistant|>",
        "stream": False
    }
    
    try:
        api_callback = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=150)
        
        # Check for non-200 HTTP status codes before parsing JSON
        if api_callback.status_code != 200:
            return f"Ollama returned HTTP error status {api_callback.status_code}. Raw response text: {api_callback.text}"
            
        # Safely try to parse JSON
        try:
            return api_callback.json().get("response", "Error: Received blank generation metadata.")
        except requests.exceptions.JSONDecodeError:
            return f"Failed to parse JSON from Ollama. The raw server response was: {api_callback.text}"
            
    except Exception as e:
        return f"Failed to execute local synthesis loop via Dockerized Ollama service: {e}"

# ==========================================
# PROCESS ENTRYPOINT EXECUTOR
# ==========================================

if __name__ == "__main__":
    try:
        # UPDATED: Only fetch and seed if the database is brand new
        needs_seeding = initialize_vector_schema_if_needed()
        
        if needs_seeding:
            dataset = fetch_software_alternatives_dataset()
            populate_vector_store(dataset)
        
        print("\n" + "="*60)
        print("LOCAL SYSTEM READY: RUNNING ANALYSIS")
        print("="*60)
        
        user_problem = "Our software development team is paying high recurring fees for centralized cloud-hosted databases, user authentication, and realtime syncing software tools."
        print(f"\nUser Input Query: '{user_problem}'")
        
        print("\nQuerying vector indexes and running local inference synthesis...")
        optimization_report = run_optimization_query(user_problem)
        
        print("\n" + "-"*30 + " GENERATED OPTIMIZATION REPORT " + "-"*30)
        print(optimization_report)
        print("-"*91)
        
    finally:
        client.close()
        print("\nWeaviate infrastructure sockets gracefully disconnected.")