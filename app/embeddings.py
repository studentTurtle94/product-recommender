import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from pathlib import Path
from dotenv import load_dotenv
import time
import re
import uuid

# Load environment variables from .env file
load_dotenv()

# Import reviews module for enhanced embeddings
from .reviews import enhance_product_embedding_text, product_reviews

# Import product search function for fallback
from .products import search_products_by_keyword, ProductItem

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("This is a test debug message from app/embeddings.py.")

# Global clients
openai_client = None
qdrant_client = None

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "products")

# Create a namespace UUID for consistent point ID generation
NAMESPACE_UUID = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # UUID namespace for URLs

def init_embedding_client():
    """Initialize OpenAI and Qdrant clients and ensure the collection exists."""
    global openai_client, qdrant_client
    
    try:
        # Initialize OpenAI client
        openai_client = OpenAI()
        logger.info("OpenAI client initialized successfully.")
        
        # Initialize Qdrant client
        logger.info(f"Initializing Qdrant client with URL: {QDRANT_URL}")
        logger.debug(f"QDRANT_API_KEY: {'*' * len(QDRANT_API_KEY) if QDRANT_API_KEY else 'None'}")  # Mask the key
        if QDRANT_API_KEY:
            qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                timeout=10.0  # Add a timeout
                )
        else:
            qdrant_client = QdrantClient(
                url=QDRANT_URL,
                timeout=10.0  # Add a timeout
            )
        logger.info("Qdrant client initialized successfully.")
        
        # Check if the collection exists
        try:
            collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            logger.info(f"Connected to existing Qdrant collection '{QDRANT_COLLECTION_NAME}'.")

            # Verify vector params match
            if isinstance(collection_info.vectors_config, models.VectorsConfig):
                params = collection_info.vectors_config.params
            elif isinstance(collection_info.vectors_config, dict):
                if '' in collection_info.vectors_config:
                    params = collection_info.vectors_config[''].params
                else:
                    first_key = next(iter(collection_info.vectors_config))
                    params = collection_info.vectors_config[first_key].params
                    logger.warning(f"Multiple named vectors found, checking parameters for '{first_key}'. Adapt if using a different vector name.")
            else:
                logger.error(f"Unexpected vectors_config format in collection info: {collection_info.vectors_config}")
                raise ValueError("Could not parse collection's vector configuration.")

            if params.size != EMBEDDING_DIMENSIONS or params.distance != models.Distance.COSINE:
                logger.warning(f"Collection '{QDRANT_COLLECTION_NAME}' exists but has mismatching vector parameters (Size: {params.size} vs {EMBEDDING_DIMENSIONS}, Distance: {params.distance} vs {models.Distance.COSINE})!")

        except Exception as e:
            if "Not found" in str(e) or "doesn't exist" in str(e) or "status_code=404" in str(e):
                logger.warning(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' not found. Creating it...")
                try:
                    qdrant_client.create_collection(
                        collection_name=QDRANT_COLLECTION_NAME,
                        vectors_config=VectorParams(size=EMBEDDING_DIMENSIONS, distance=Distance.COSINE),
                    )
                    logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created successfully.")
                except Exception as create_err:
                    logger.error(f"Failed to create Qdrant collection '{QDRANT_COLLECTION_NAME}': {create_err}")
                    raise create_err

        # Log collection stats (point count)
        count_result = qdrant_client.count(collection_name=QDRANT_COLLECTION_NAME, exact=False)
        logger.info(f"Qdrant Collection '{QDRANT_COLLECTION_NAME}' approximate point count: {count_result.count}")
        
    except Exception as e:
        logger.error(f"Error initializing clients: {e}", exc_info=True)
        raise ValueError(f"Qdrant initialization failed: {e}")

async def get_embedding(text: str) -> List[float]:
    """Get embedding for a text using OpenAI API. Returns a list."""
    if not openai_client:
        raise ValueError("OpenAI client not initialized. Call init_embedding_client first.")
    
    try:
        # Ensure text is not empty and is a string
        text = str(text).replace("\\n", " ") # Replace newlines with spaces for embedding models
        if not text.strip():
            logger.warning("Attempted to get embedding for empty text. Returning zero vector.")
            return [0.0] * EMBEDDING_DIMENSIONS

        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text], # API expects a list of strings
            dimensions=EMBEDDING_DIMENSIONS
        )
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding for text snippet '{text[:100]}...': {e}")
        return [0.0] * EMBEDDING_DIMENSIONS

async def generate_product_embeddings(products: Optional[List[Dict[str, Any]]] = None):
    """Generate embeddings for products and upsert them into the Qdrant collection."""
    global qdrant_client
    if not openai_client or not qdrant_client:
        raise ValueError("Clients not initialized. Call init_embedding_client first.")

    if products is None:
        from .products import get_all_products # Assuming get_all_products returns list of dicts or objects
        try:
            products = get_all_products() # Fetch if not provided
            if not products:
                 logger.warning("No products found to generate embeddings for.")
                 return False
        except Exception as e:
            logger.error(f"Failed to fetch products: {e}")
            return False

    logger.info(f"Generating embeddings for {len(products)} products and upserting to Qdrant collection '{QDRANT_COLLECTION_NAME}'...")

    batch_size = 100 # Qdrant can handle large batches, adjust as needed based on performance/memory
    total_upserted = 0

    for i in range(0, len(products), batch_size):
        batch_products = products[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{len(products)//batch_size + 1} ({len(batch_products)} products)...")

        points_to_upsert = []
        texts_to_embed = []
        product_ids_in_batch = [] # Keep track of IDs for association

        for product in batch_products:
            # Handle ProductItem object access consistently
            product_id_any = product.id # Use direct attribute access
            parent_asin = product.parent_asin # Use direct attribute access

            if not product_id_any:
                logger.warning(f"Skipping product due to missing ID: {product}")
                continue

            # Generate a deterministic UUID for the point ID based on the product ID
            point_id = str(uuid.uuid5(NAMESPACE_UUID, str(product_id_any)))

            try:
                # 1. Get rich text representation
                product_text = enhance_product_embedding_text(product, parent_asin or product_id_any)
                if not product_text:
                    logger.warning(f"Skipping product {point_id} due to empty text after enhancement.")
                    continue

                texts_to_embed.append(product_text)
                product_ids_in_batch.append(point_id)

                # Prepare payload (metadata) - Use attribute access for ProductItem
                metadata = {
                    "title": product.title,
                    "price": product.price if product.price is not None else 0.0,
                    "categories": ", ".join(product.categories or []),
                    "main_category": product.main_category or "",
                    "has_reviews": product.has_reviews,
                    "rating": product.rating if product.rating is not None else 0.0,
                    "original_id": str(product_id_any)
                }
                # Clean payload: Ensure values are suitable JSON types
                cleaned_payload = {}
                for k, v in metadata.items():
                    if v is not None:
                        if isinstance(v, (str, int, float, bool, list, dict)):
                            cleaned_payload[k] = v
                        else:
                            try:
                                cleaned_payload[k] = str(v)
                                logger.debug(f"Converted payload field '{k}' of type {type(v)} to string for product {point_id}")
                            except Exception:
                                logger.warning(f"Could not serialize payload field '{k}' for product {point_id}. Skipping field.")

                points_to_upsert.append({
                    "id": point_id,
                    "payload": cleaned_payload,
                })

            except AttributeError as ae:
                logger.error(f"Attribute error processing product {product_id_any}: {ae}. Product data: {product.dict()}", exc_info=True)
                if len(texts_to_embed) > len(product_ids_in_batch): texts_to_embed.pop()
                if len(points_to_upsert) > len(product_ids_in_batch): points_to_upsert.pop()
                continue
            except Exception as e:
                logger.error(f"Error preparing data for product {product_id_any}: {e}", exc_info=True)
                if len(texts_to_embed) > len(product_ids_in_batch): texts_to_embed.pop()
                if len(points_to_upsert) > len(product_ids_in_batch): points_to_upsert.pop()
                continue

        if not texts_to_embed or not points_to_upsert:
            logger.warning(f"Batch {i//batch_size + 1} resulted in no valid products to process.")
            continue

        # 2. Generate embeddings for the batch texts
        try:
            logger.debug(f"Requesting embeddings for {len(texts_to_embed)} texts in batch...")
            response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts_to_embed,
                dimensions=EMBEDDING_DIMENSIONS
            )
            batch_embeddings = [item.embedding for item in response.data]
            logger.debug(f"Received {len(batch_embeddings)} embeddings for batch.")

            if len(batch_embeddings) != len(points_to_upsert):
                 logger.error(f"Mismatch between embeddings ({len(batch_embeddings)}) and points prepared ({len(points_to_upsert)}) for batch {i//batch_size + 1}. Skipping upsert.")
                 continue

            for idx, point_data in enumerate(points_to_upsert):
                point_data["vector"] = batch_embeddings[idx]

        except Exception as e:
            logger.error(f"Error getting embeddings for batch {i//batch_size + 1}: {e}")
            continue

        # 3. Upsert batch to Qdrant
        try:
            if points_to_upsert:
                logger.info(f"Upserting {len(points_to_upsert)} points to Qdrant collection '{QDRANT_COLLECTION_NAME}'...")

                qdrant_points = [models.PointStruct(**point) for point in points_to_upsert]

                upsert_response = qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points=qdrant_points,
                    wait=True
                )
                logger.debug(f"Qdrant upsert response status: {upsert_response.status}")

                if upsert_response.status == models.UpdateStatus.COMPLETED:
                    upserted_count_batch = len(points_to_upsert)
                    total_upserted += upserted_count_batch
                    logger.info(f"Successfully upserted batch {i//batch_size + 1} ({upserted_count_batch} points). Status: {upsert_response.status}")
                else:
                     logger.warning(f"Qdrant upsert status for batch {i//batch_size + 1} was not COMPLETED: {upsert_response.status}")

            else:
                logger.warning(f"Skipping upsert for batch {i//batch_size + 1} as no points were generated.")

        except Exception as e:
            logger.error(f"Error upserting batch {i//batch_size + 1} to Qdrant: {e}", exc_info=True)

    logger.info(f"Completed processing all products. Total points potentially upserted to Qdrant: {total_upserted}")
    final_count = qdrant_client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True)
    logger.info(f"Final exact point count in Qdrant collection '{QDRANT_COLLECTION_NAME}': {final_count.count}")
    return True

async def search_similar_products(query: str, top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Search for products similar to the query text using Qdrant, with keyword fallback.

    Returns:
        A list of dictionaries, each containing 'original_id' and 'rating'.
    """
    global qdrant_client
    if not openai_client or not qdrant_client:
        raise ValueError("Clients not initialized. Call init_embedding_client first.")

    products_found = []
    try:
        # 1. Get the embedding for the query text
        logger.info(f"Getting embedding for search query: '{query[:100]}...'")
        query_embedding = await get_embedding(query)

        if not any(query_embedding):
            logger.warning("Failed to get a valid embedding for the search query. Attempting keyword fallback.")
            # Fallback directly if embedding fails
            keyword_results = search_products_by_keyword(keyword=query, limit=top_k)
            products_found = [{"original_id": p.id, "rating": p.rating if p.rating is not None else 0.0} for p in keyword_results]
            logger.info(f"Found {len(products_found)} products via keyword fallback (embedding failure).")
            return products_found

        # 2. Build Qdrant search filter (if filter_dict is provided)
        qdrant_filter = None
        if filter_dict:
            must_conditions = []
            for key, value in filter_dict.items():
                 if isinstance(value, bool) or isinstance(value, str) or isinstance(value, int) or isinstance(value, float):
                      must_conditions.append(
                          models.FieldCondition(
                              key=key,
                              match=models.MatchValue(value=value)
                          )
                      )
                 else:
                     logger.warning(f"Unsupported filter type for key '{key}': {type(value)}. Skipping filter condition.")

            if must_conditions:
                 qdrant_filter = models.Filter(must=must_conditions)
                 logger.info(f"Constructed Qdrant filter: {qdrant_filter}")

        # 3. Perform the search in Qdrant
        logger.info(f"Searching Qdrant collection '{QDRANT_COLLECTION_NAME}' with top_k={top_k} and filter={qdrant_filter}")
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True
        )
        logger.debug(f"Qdrant search result: {search_result}")

        # 4. Extract original product IDs and ratings from the results payload
        for hit in search_result:
            if hit.payload and 'original_id' in hit.payload:
                product_info = {
                    "original_id": hit.payload['original_id'],
                    "rating": hit.payload.get('rating', 0.0) # Get rating, default to 0.0
                }
                products_found.append(product_info)
            else:
                logger.warning(f"Search hit {hit.id} missing payload or original_id. Skipping.")

        if not products_found:
            logger.info("Vector search yielded no results. Attempting keyword fallback.")
            # Fallback if vector search is empty
            keyword_results = search_products_by_keyword(keyword=query, limit=top_k)
            products_found = [{"original_id": p.id, "rating": p.rating if p.rating is not None else 0.0} for p in keyword_results]
            logger.info(f"Found {len(products_found)} products via keyword fallback (empty vector search).")
        else:
            logger.info(f"Found {len(products_found)} similar products via vector search.")

        return products_found

    except Exception as e:
        logger.error(f"Error during vector search: {e}. Attempting keyword fallback.", exc_info=True)
        try:
            # Fallback in case of any exception during vector search
            keyword_results = search_products_by_keyword(keyword=query, limit=top_k)
            products_found = [{"original_id": p.id, "rating": p.rating if p.rating is not None else 0.0} for p in keyword_results]
            logger.info(f"Found {len(products_found)} products via keyword fallback (vector search error).")
            return products_found
        except Exception as fallback_e:
            logger.error(f"Error during keyword fallback search: {fallback_e}", exc_info=True)
            return [] # Return empty if both searches fail

async def delete_vectors_by_ids(ids: List[str | int]):
    """Delete vectors from the Qdrant collection by their IDs."""
    global qdrant_client
    if not qdrant_client:
        raise ValueError("Qdrant client not initialized.")

    if not ids:
        logger.warning("No IDs provided for deletion.")
        return

    logger.info(f"Attempting to delete {len(ids)} vectors from Qdrant collection '{QDRANT_COLLECTION_NAME}'...")
    try:
        delete_result = qdrant_client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=models.PointIdsList(points=ids),
            wait=True
        )
        logger.info(f"Qdrant delete operation status: {delete_result.status}")
        if delete_result.status != models.UpdateStatus.COMPLETED:
             logger.warning(f"Qdrant delete operation did not complete successfully: {delete_result.status}")

    except Exception as e:
        logger.error(f"Error deleting vectors from Qdrant: {e}", exc_info=True)

async def delete_all_vectors():
    """Delete all vectors from the Qdrant collection."""
    global qdrant_client
    if not qdrant_client:
        raise ValueError("Qdrant client not initialized.")

    logger.warning(f"Attempting to delete ALL vectors from Qdrant collection '{QDRANT_COLLECTION_NAME}'...")
    try:
        logger.info(f"Deleting collection '{QDRANT_COLLECTION_NAME}'...")
        qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
        logger.info(f"Recreating collection '{QDRANT_COLLECTION_NAME}'...")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMENSIONS, distance=Distance.COSINE)
        )
        logger.info(f"Successfully deleted and recreated Qdrant collection '{QDRANT_COLLECTION_NAME}'.")

    except Exception as e:
        logger.error(f"Error deleting all vectors from Qdrant: {e}", exc_info=True)

# Example usage (consider moving to a script or main execution block)
# async def main():
#     init_embedding_client()
#     # await generate_product_embeddings() # Example call
#     # results = await search_similar_products("elegant summer dress")
#     # print(results)
#     # await delete_vectors_by_ids(["product_id_1", "product_id_2"])
#     # await delete_all_vectors() # Be careful!

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
