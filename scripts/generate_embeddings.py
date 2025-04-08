# scripts/generate_embeddings.py
import asyncio
import logging
import sys
import os

# Add the parent directory (project root) to the Python path
# This allows us to import modules from the 'app' directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.products import load_products, get_all_products
from app.embeddings import init_embedding_client, generate_product_embeddings, delete_all_vectors, QDRANT_COLLECTION_NAME
from app.reviews import load_reviews # Need to load reviews too for enhanced embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generate_embeddings_script")

async def main(clear_existing: bool = False):
    """
    Main function to load products, initialize clients, optionally clear existing data,
    and generate embeddings.
    """
    logger.info("Starting embedding generation process...")

    # 1. Load data
    logger.info("Loading products...")
    load_products()
    logger.info("Loading reviews...")
    load_reviews() # Load reviews needed for enhance_product_embedding_text
    products = get_all_products()
    if not products:
        logger.error("No products loaded. Exiting.")
        return
    logger.info(f"Loaded {len(products)} products.")

    # 2. Initialize clients (OpenAI and Qdrant)
    logger.info("Initializing clients (OpenAI, Qdrant)...")
    try:
        init_embedding_client() # This also ensures the collection exists
    except ValueError as e:
        logger.error(f"Failed to initialize clients: {e}")
        return
    logger.info("Clients initialized successfully.")

    # 3. Optionally clear existing vectors
    if clear_existing:
        logger.warning(f"Attempting to delete all vectors from collection: {QDRANT_COLLECTION_NAME}")
        try:
            await delete_all_vectors()
            logger.info(f"Successfully cleared existing vectors from {QDRANT_COLLECTION_NAME}.")
        except Exception as e:
            logger.error(f"Failed to clear existing vectors: {e}")
            # Decide if you want to proceed or exit if clearing fails
            # return

    # 4. Generate and upsert embeddings
    logger.info("Generating and upserting embeddings...")
    try:
        success = await generate_product_embeddings(products)
        if success:
            logger.info("Embedding generation and upsert completed successfully.")
        else:
            logger.warning("Embedding generation process finished, but may not have been fully successful. Check logs.")
    except Exception as e:
        logger.error(f"An error occurred during embedding generation: {e}", exc_info=True)

    logger.info("Embedding generation process finished.")

if __name__ == "__main__":
    # Add command-line argument parsing for --clear
    import argparse
    parser = argparse.ArgumentParser(description="Generate and upsert product embeddings to Qdrant.")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing vectors in the collection before generating new ones."
    )
    args = parser.parse_args()

    asyncio.run(main(clear_existing=args.clear)) 