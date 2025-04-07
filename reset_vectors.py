#!/usr/bin/env python3
"""
Utility script to reset the vector store and regenerate embeddings.
"""
import requests
import time
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API endpoints
BASE_URL = "http://localhost:8000"
DELETE_ENDPOINT = f"{BASE_URL}/vectors"
GENERATE_ENDPOINT = f"{BASE_URL}/generate-embeddings"

def reset_vectors():
    """
    Reset the vector store by deleting all vectors and regenerating them.
    """
    try:
        # Step 1: Delete existing vectors
        logger.info("Deleting existing vectors...")
        response = requests.delete(DELETE_ENDPOINT)
        if response.status_code != 200:
            logger.error(f"Failed to delete vectors: {response.text}")
            return False
        
        logger.info("Vector deletion initiated. Waiting 5 seconds for completion...")
        time.sleep(5)  # Wait for deletion to complete
        
        # Step 2: Generate new embeddings
        logger.info("Generating new embeddings...")
        response = requests.post(GENERATE_ENDPOINT)
        if response.status_code != 200:
            logger.error(f"Failed to generate embeddings: {response.text}")
            return False
        
        logger.info("Embedding generation initiated.")
        logger.info("The process will continue in the background.")
        return True
    
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to API at {BASE_URL}. Make sure the server is running.")
        return False
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting vector store reset process...")
    
    if not reset_vectors():
        logger.error("Failed to reset vectors.")
        sys.exit(1)
    
    logger.info("Reset process initiated successfully.")
    logger.info("Note: Embedding generation continues in the background and may take some time to complete.") 