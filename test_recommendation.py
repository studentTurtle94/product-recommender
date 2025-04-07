import os
import asyncio
import json
import logging
from dotenv import load_dotenv
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configuration
API_URL = "http://localhost:8000"
SEARCH_ENDPOINT = f"{API_URL}/search"
RECOMMEND_ENDPOINT = f"{API_URL}/recommend"
REVIEWS_ENDPOINT = f"{API_URL}/reviews"

async def test_search(query: str):
    """Test the search endpoint with a query."""
    async with httpx.AsyncClient() as client:
        params = {"query": query, "limit": 5}
        logger.info(f"Testing search with query: '{query}'")
        response = await client.get(SEARCH_ENDPOINT, params=params)
        
        if response.status_code == 200:
            results = response.json()
            logger.info(f"Found {len(results.get('products', []))} products")
            
            # Print product titles
            for i, product in enumerate(results.get('products', []), 1):
                logger.info(f"  {i}. {product.get('title')} - ${product.get('price', 0):.2f}")
            
            return results.get('products', [])
        else:
            logger.error(f"Search failed: {response.status_code} - {response.text}")
            return []

async def test_recommend(query: str):
    """Test the recommend endpoint with a query."""
    async with httpx.AsyncClient() as client:
        params = {"query": query, "limit": 5}
        logger.info(f"Testing recommendation with query: '{query}'")
        response = await client.get(RECOMMEND_ENDPOINT, params=params)
        
        if response.status_code == 200:
            results = response.json()
            logger.info(f"Found {len(results.get('products', []))} recommended products")
            
            # Print product titles with explanations
            for i, product in enumerate(results.get('products', []), 1):
                logger.info(f"  {i}. {product.get('title')} - ${product.get('price', 0):.2f}")
                logger.info(f"     Explanation: {product.get('explanation', 'No explanation')}")
            
            # Print alternative searches
            if results.get('alternative_searches'):
                logger.info(f"Alternative searches: {', '.join(results.get('alternative_searches'))}")
            
            return results
        else:
            logger.error(f"Recommendation failed: {response.status_code} - {response.text}")
            return {"products": [], "alternative_searches": []}

async def test_reviews(product_id: str):
    """Test the reviews endpoint with a product ID."""
    async with httpx.AsyncClient() as client:
        logger.info(f"Testing reviews for product: {product_id}")
        response = await client.get(f"{REVIEWS_ENDPOINT}/{product_id}")
        
        if response.status_code == 200:
            results = response.json()
            logger.info(f"Found {results.get('review_count', 0)} reviews")
            
            # Print sentiment data
            if 'overall_sentiment' in results:
                logger.info(f"Overall sentiment: {results['overall_sentiment']:.2f}")
            
            # Print key phrases
            if results.get('key_phrases'):
                logger.info(f"Key phrases: {', '.join(results.get('key_phrases', [])[:3])}")
            
            # Print usage contexts
            if results.get('usage_contexts'):
                logger.info(f"Usage contexts: {', '.join(results.get('usage_contexts', [])[:2])}")
            
            return results
        else:
            logger.error(f"Reviews failed: {response.status_code} - {response.text}")
            return {}

async def run_tests():
    """Run a series of tests for the recommendation system."""
    test_queries = [
        "A comfortable dress for summer vacation",
        "Professional shoes for work that look stylish",
        "Warm winter jacket that's also waterproof",
        "Casual outfit for weekend brunch",
        "Sports bra with good support for running"
    ]
    
    # Test each query
    for query in test_queries:
        logger.info("\n" + "-"*80)
        logger.info(f"TESTING QUERY: {query}")
        logger.info("-"*80)
        
        # Get search results first
        products = await test_search(query)
        
        # Then test recommendations
        await test_recommend(query)
        
        # Test reviews for the first product if available
        if products and len(products) > 0:
            product_id = products[0].get('id')
            if product_id:
                await test_reviews(product_id)
        
        logger.info("-"*80 + "\n")

if __name__ == "__main__":
    logger.info("Starting recommendation system tests...")
    asyncio.run(run_tests())
    logger.info("Tests completed!") 