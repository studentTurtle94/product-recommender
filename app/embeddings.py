import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv
import time  # Import time for potential polling
import re

# Load environment variables from .env file
load_dotenv()

# Import reviews module for enhanced embeddings
from .reviews import enhance_product_embedding_text, product_reviews

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global client for OpenAI
client = None

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small dimensions
VECTOR_STORE_ID = "vs_67f182cc8c0081918a1628b922d36ae9"

def init_embedding_client():
    """Initialize the OpenAI client for embeddings."""
    global client
    
    if VECTOR_STORE_ID == "[key here]":
        logger.error("VECTOR_STORE_ID is not set in app/embeddings.py.")
        logger.error("Please replace '[key here]' with your actual OpenAI Vector Store ID.")
        raise ValueError("Missing VECTOR_STORE_ID. Update app/embeddings.py.")
        
    # Check for API key in environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        logger.error("Please add your OpenAI API key to the .env file or set it as an environment variable.")
        logger.error("Example: OPENAI_API_KEY=sk-your-key-here")
        raise ValueError("Missing OPENAI_API_KEY. See logs for instructions.")
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized successfully.")

async def get_embedding(text: str) -> List[float]:
    """Get embedding for a text using OpenAI API. Returns a list."""
    if not client:
        raise ValueError("OpenAI client not initialized. Call init_embedding_client first.")
    
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIMENSIONS
        )
        
        # Extract the embedding vector as a list
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        # Return a zero vector as fallback
        return [0.0] * EMBEDDING_DIMENSIONS

async def generate_product_embeddings(products):
    """Generate embeddings for all products and upload to OpenAI vector store."""
    if not client:
        raise ValueError("OpenAI client not initialized.")
        
    from .products import get_all_products
    
    products = get_all_products()
    logger.info(f"Generating embeddings for {len(products)} products and uploading to Vector Store ID: {VECTOR_STORE_ID}...")
    
    # Process products in batches
    batch_size = 100  # Adjust as needed based on API limits and performance
    total_uploaded = 0
    
    for i in range(0, len(products), batch_size):
        batch_products = products[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} ({len(batch_products)} products)...")
        
        # Prepare batch payload for vector store upsert
        vectors_payload = []
        for product in batch_products:
            try:
                # 1. Get rich text representation
                product_text = enhance_product_embedding_text(product, product.parent_asin or product.id)
                
                # 2. Generate embedding (returns list)
                embedding_values = await get_embedding(product_text)
                
                # 3. Prepare vector object for the batch
                vectors_payload.append({
                    "id": product.id, # Use product ID as the vector ID
                    "values": embedding_values,
                    "metadata": {
                        "title": product.title or "",
                        "price": product.price or 0.0,
                        "categories": product.categories or [],
                        "main_category": product.main_category or "",
                        "has_reviews": bool(product.parent_asin in product_reviews)
                        # Ensure metadata values are JSON serializable primitive types
                    }
                })
                
            except Exception as e:
                logger.error(f"Error preparing embedding for product {product.id}: {e}")
                # Decide if you want to skip this product or halt the batch
                continue # Skip this product

        if not vectors_payload:
            logger.warning(f"Batch {i//batch_size + 1} resulted in no vectors to upload.")
            continue

        # Upload batch to vector store using the correct API
        try:
            logger.info(f"Uploading {len(vectors_payload)} vectors to store {VECTOR_STORE_ID}...")
            # Note: The SDK might evolve. This uses the structure as of recent versions.
            # This endpoint might not exist or might have changed.
            # Check official OpenAI documentation for the latest Vector Store API.
            # Assuming 'vectors.batch_upsert' is the intended function based on previous context.
            # If this fails, the API structure might differ (e.g., might need file uploads).
            
            # ATTENTION: The `client.beta.vector_stores.vectors` path might be incorrect
            # or part of an outdated beta. Let's try the file-based approach
            # which seems more standard now.

            # ---- Alternative: File-based Upload ----
            # 1. Create a temporary file with product texts + metadata
            temp_file_path = f"data/temp_batch_{i//batch_size + 1}.jsonl"
            file_ids_for_batch = []
            try:
                os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    for product in batch_products:
                         # Regenerate text or store it temporarily if needed
                         product_text = enhance_product_embedding_text(product, product.parent_asin or product.id)
                         # Write data needed for embedding by OpenAI.
                         # Include ProductID clearly in the text for later retrieval.
                         f.write(json.dumps({
                             "product_id": product.id, # Keep for reference if needed
                             "text": f"ProductID: {product.id}\nContent:\n{product_text}" # Embed ProductID in text
                         }) + "\n")

                # 2. Upload the file
                logger.info(f"Uploading temporary file: {temp_file_path}")
                uploaded_file = client.files.create(
                    file=open(temp_file_path, "rb"),
                    purpose="assistants" # Use 'assistants' purpose for vector stores
                )
                logger.info(f"File uploaded successfully: ID {uploaded_file.id}")
                file_ids_for_batch.append(uploaded_file.id)

            except Exception as e:
                 logger.error(f"Error creating or uploading temporary file for batch {i//batch_size + 1}: {e}")
                 continue # Skip this batch
            finally:
                # 5. Clean up the temporary file
                 if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                        logger.info(f"Removed temporary file: {temp_file_path}")
                    except OSError as e:
                        logger.error(f"Error removing temporary file {temp_file_path}: {e}")

            if not file_ids_for_batch:
                logger.warning(f"No files generated for batch {i//batch_size + 1}. Skipping vector store update.")
                continue

            # 3. Add the file(s) to the vector store via a batch
            try:
                 logger.info(f"Adding file batch {file_ids_for_batch} to vector store {VECTOR_STORE_ID}...")
                 file_batch = client.vector_stores.file_batches.create(
                     vector_store_id=VECTOR_STORE_ID,
                     file_ids=file_ids_for_batch
                 )
                 logger.info(f"File batch created: ID {file_batch.id}, Status: {file_batch.status}")

                 # 4. (Optional but recommended) Poll for completion
                 while file_batch.status not in ["completed", "failed", "cancelled"]:
                     time.sleep(5) # Wait 5 seconds before checking again
                     file_batch = client.vector_stores.file_batches.retrieve(
                         vector_store_id=VECTOR_STORE_ID,
                         batch_id=file_batch.id
                     )
                     logger.info(f"Polling file batch {file_batch.id}: Status {file_batch.status}")

                 if file_batch.status == "completed":
                     logger.info(f"File batch {file_batch.id} completed successfully. Files processed: {file_batch.file_counts.completed}")
                     total_uploaded += file_batch.file_counts.completed # Assuming 1 file per batch here
                 else:
                     logger.error(f"File batch {file_batch.id} failed or was cancelled. Status: {file_batch.status}")
                     # Log specific errors if available in file_batch details

            except Exception as e:
                 logger.error(f"Error adding file batch to vector store {VECTOR_STORE_ID}: {e}")
                 logger.error(f"Full error details: {str(e)}")

            # ---- End of File-based Upload ----

        except Exception as e:
            logger.error(f"General error during upload for batch {i//batch_size + 1}: {e}")
            logger.error(f"Full error: {str(e)}")
            # Consider adding more robust error handling / retries
    
    logger.info(f"Completed processing all products. Total files added to vector store batches: {total_uploaded}") # Adjusted log message
    return True

async def search_similar_products(query: str, top_k: int = 5) -> List[str]:
    """
    Search for products similar to the query using the vector store's search capability.
    This searches the content of the files added to the store.
    """
    if not client:
        raise ValueError("OpenAI client not initialized.")

    try:
        logger.info(f"Performing vector store search for query: '{query}' in store {VECTOR_STORE_ID}")
        
        # Use the documented search endpoint
        search_results = client.vector_stores.search(
            vector_store_id=VECTOR_STORE_ID,
            query=query,
            limit=top_k # Request slightly more initially if needed for filtering/parsing
        )

        product_ids = set() # Use a set to avoid duplicates
        if search_results and search_results.data:
            logger.info(f"Vector store search returned {len(search_results.data)} results.")
            for result in search_results.data:
                # Attempt to extract ProductID from the text content snippets
                if result.content:
                    for content_item in result.content:
                        if content_item.type == "text":
                            # Simple extraction assuming 'ProductID: ID\nContent...' format
                            match = re.search(r"^ProductID:\s*(\S+)", content_item.text)
                            if match:
                                product_id = match.group(1)
                                product_ids.add(product_id)
                                # Limit to top_k unique IDs
                                if len(product_ids) >= top_k:
                                    break
                if len(product_ids) >= top_k:
                    break
        else:
             logger.info("Vector store search returned no results.")

        found_ids = list(product_ids)
        logger.info(f"Extracted {len(found_ids)} unique product IDs from search results: {found_ids}")
        return found_ids

    except Exception as e:
        logger.error(f"Error during vector store search: {e}")
        logger.error(f"Full error: {str(e)}")
        return []

async def delete_all_vectors():
    """
    Delete all files associated with the vector store.
    Note: This deletes the *files* from the store, effectively removing their content.
    It does not delete the vector store itself.
    Uses the stable `client.vector_stores.files.*` endpoints.
    """
    if not client:
        raise ValueError("OpenAI client not initialized.")
    
    try:
        logger.info(f"Listing all files in vector store {VECTOR_STORE_ID} to delete...")
        # List all files currently associated with the vector store
        store_files = client.vector_stores.files.list(vector_store_id=VECTOR_STORE_ID)
        
        file_ids_to_delete = [f.id for f in store_files.data]
        
        if not file_ids_to_delete:
            logger.info(f"No files found in vector store {VECTOR_STORE_ID} to delete.")
            return True
            
        logger.info(f"Found {len(file_ids_to_delete)} files to delete from vector store {VECTOR_STORE_ID}.")

        deleted_count = 0
        failed_count = 0
        for file_id in file_ids_to_delete:
            try:
                # Delete each file association from the vector store
                delete_status = client.vector_stores.files.delete(
                    vector_store_id=VECTOR_STORE_ID,
                    file_id=file_id
                )
                if delete_status.deleted:
                    logger.info(f"Successfully deleted file {file_id} from vector store {VECTOR_STORE_ID}.")
                    deleted_count += 1
                    # Optionally, delete the file from OpenAI org storage too if no longer needed
                    # try:
                    #     client.files.delete(file_id=file_id)
                    #     logger.info(f"Successfully deleted file {file_id} from organization storage.")
                    # except Exception as file_del_err:
                    #     logger.warning(f"Could not delete file {file_id} from organization storage: {file_del_err}")
                else:
                    logger.warning(f"Failed to delete file {file_id} from vector store {VECTOR_STORE_ID}. Status: {delete_status}")
                    failed_count += 1
            except Exception as e:
                logger.error(f"Error deleting file {file_id} from vector store {VECTOR_STORE_ID}: {e}")
                failed_count += 1
        
        logger.info(f"Completed deletion attempt: {deleted_count} files deleted, {failed_count} failed.")
        return failed_count == 0 # Return True if all deletions were successful

    except Exception as e:
        logger.error(f"Error listing or deleting files from vector store {VECTOR_STORE_ID}: {e}")
        logger.error(f"Full error: {str(e)}")
        return False
