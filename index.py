# index.py
import json
import requests
from pathlib import Path
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore


def get_documents_from_markdown_files(docs_dir="docs"):
    """Reads markdown files from docs directory and returns list of Documents"""

    documents = []
    docs_path = Path(docs_dir)

    # Find all markdown files recursively
    md_files = list(docs_path.rglob("*.md"))

    print(f"Found {len(md_files)} markdown files")

    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Get relative path for better identification
            relative_path = md_file.relative_to(docs_path)

            # Create metadata
            metadata = {
                "file_path": str(relative_path),
                "file_name": md_file.name,
                "source": "cscs-docs"
            }

            # Create Document object
            doc = Document(
                text=content,
                metadata=metadata
            )
            documents.append(doc)

        except Exception as e:
            print(f"Error processing {md_file}: {e}")
            continue

    return documents


# ElasticsearchStore is a VectorStore that
# takes care of ES Index and Data management.
# es_vector_store = ElasticsearchStore(index_name="calls",
#                                      vector_field='conversation_vector',
#                                      text_field='conversation',
#                                      es_cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
#                                      es_api_key=os.getenv("ELASTIC_API_KEY"))

es_vector_store = ElasticsearchStore(
    index_name="cscs_docs",
    vector_field='doc_vector',
    text_field='content',
    es_url="http://127.0.0.1:9200"
)


def save_documents_to_json(documents, output_file="cscs_docs.json"):
    """Save documents to JSON file for inspection/backup"""
    json_docs = []

    for idx, doc in enumerate(documents):
        # Extract title from first heading if available
        title = doc.metadata.get('file_name', '').replace('.md', '')
        lines = doc.text.split('\n')
        for line in lines:
            if line.strip().startswith('# '):
                title = line.strip().replace('# ', '')
                break

        json_docs.append({
            "doc_id": idx,
            "file_path": doc.metadata.get('file_path', ''),
            "file_name": doc.metadata.get('file_name', ''),
            "title": title,
            "content": doc.text,
            "source": doc.metadata.get('source', 'cscs-docs')
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_docs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(json_docs)} documents to {output_file}")


def clean_elasticsearch_index(index_name="cscs_docs", es_url="http://127.0.0.1:9200"):
    """Delete existing index to ensure clean state"""
    try:
        response = requests.delete(f"{es_url}/{index_name}")
        if response.status_code == 200:
            print(f"Deleted existing index: {index_name}")
        elif response.status_code == 404:
            print(f"Index {index_name} does not exist (this is fine for first run)")
        else:
            print(f"Warning: Could not delete index. Status: {response.status_code}")
    except Exception as e:
        print(f"Warning: Error cleaning index: {e}")


def main(save_json=True, clean_index=True):
    print("Starting CSCS Docs indexing pipeline (TEST MODE: 5 DOCS)...")

    # 1. Clean index
    if clean_index:
        clean_elasticsearch_index()

    # 2. Load Documents
    print("Loading markdown documents...")
    documents = get_documents_from_markdown_files(docs_dir="cscs-docs/docs")

    if not documents:
        print("No documents found!")
        return

    # --- TEST MODE: KEEP ONLY 5 DOCUMENTS ---
    print(f"Original count: {len(documents)}. Keeping only the first 5 for testing.")
    documents = documents[:5] 
    # ----------------------------------------

    if save_json:
        save_documents_to_json(documents)

    # 3. Setup Embedding Model
    print("Initializing Ollama model...")
    ollama_embedding = OllamaEmbedding("llama3.2:1b")

    # 4. Setup Pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=64),
            ollama_embedding,
        ],
        vector_store=es_vector_store
    )

    # 5. Run Pipeline
    # We still use a small loop just to see progress, even for 5 docs
    total_docs = len(documents)
    print(f"Processing {total_docs} documents...")
    
    # Run all 5 at once (it's small enough)
    pipeline.run(documents=documents)

    print("\n.....Done running pipeline.....")
    print(f"Successfully indexed {total_docs} documents to Elasticsearch")

if __name__ == "__main__":
    main()
