# query.py
import sys
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
# This imports the updated configuration from your new index.py
from index import es_vector_store

# 1. Setup LLM (Must match the model in index.py!)
# We use request_timeout=300.0 to prevent "ReadTimeout" errors
local_llm = Ollama(model="llama3.2:1b", request_timeout=300.0)
Settings.embed_model = OllamaEmbedding("llama3.2:1b")
Settings.llm = local_llm

# 2. Connect to Index
index = VectorStoreIndex.from_vector_store(es_vector_store)

# 3. Create Query Engine
query_engine = index.as_query_engine(local_llm, similarity_top_k=3, streaming=True)

def chat_loop():
    print("\n--- CSCS Admin Chat (Type 'exit' to quit) ---")
    while True:
        user_input = input("\nQuery: ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        
        try:
            print("\nResponse: ", end="", flush=True)
            
            # Run the query
            response = query_engine.query(user_input)
            
            # Stream the answer text
            response.print_response_stream()
            print("\n") 
            
            print("-" * 30)
            print("Sources used:")
            seen_files = set()
            for node in response.source_nodes:
                # Get the filename from metadata
                file_name = node.metadata.get('file_name', 'Unknown file')
                score = node.score if node.score else 0.0
                
                # Avoid printing the same file twice
                if file_name not in seen_files:
                    print(f"• {file_name} (Relevance: {score:.2f})")
                    seen_files.add(file_name)
            print("-" * 30)

        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    chat_loop()