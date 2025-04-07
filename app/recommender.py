import logging
import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
import numpy as np

from .products import get_product_by_id, ProductItem
from .embeddings import get_embedding, search_similar_products

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global OpenAI client
client = None

def init_recommender():
    """Initialize the recommender module with OpenAI client."""
    global client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        logger.error("Please add your OpenAI API key to the .env file or set it as an environment variable.")
        logger.error("Example: OPENAI_API_KEY=sk-your-key-here")
        raise ValueError("Missing OPENAI_API_KEY. See logs for instructions.")
    
    client = OpenAI(api_key=api_key)
    logger.info("Recommender initialized with OpenAI client")

async def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform semantic search on products based on natural language query."""
    if not client:
        raise ValueError("Recommender not initialized. Call init_recommender first.")
    
    try:
        logger.info(f"Performing semantic search for query: {query}")
        
        # With our new implementation, search_similar_products handles everything
        product_ids = await search_similar_products(query, top_k=top_k)
        
        # Get full product information
        products = []
        for product_id in product_ids:
            product = get_product_by_id(product_id)
            if product:
                products.append(product.dict())
        
        logger.info(f"Found {len(products)} products matching query")
        return products
    
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return []

async def refine_recommendations(query: str, products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Use LLM to refine search results and provide explanations."""
    if not client:
        raise ValueError("Recommender not initialized. Call init_recommender first.")
    
    try:
        # Create a detailed prompt with all product information
        products_text = ""
        for i, product in enumerate(products):
            price_str = f"${product['price']:.2f}" if product.get('price') else "Price not available"
            products_text += f"Product {i+1}: {product['title']} - {price_str}\n"
            products_text += f"Categories: {', '.join(product['categories'])}\n"
            products_text += f"Features: {product.get('description', 'No description available')}\n\n"
        
        prompt = f"""
        You are a fashion recommendation assistant. A customer has made the following request:
        
        "{query}"
        
        Based on this request, I found these products that might be relevant:
        
        {products_text}
        
        Please analyze these products and the customer's request to:
        1. Rank the products from most to least relevant for this specific request
        2. Provide a brief explanation for each recommendation (why it matches their request)
        3. Suggest one or two additional relevant search terms the customer might want to try
        
        Format your response as a JSON with these keys:
        - ranked_products: list of product indices (1-based) in order of relevance
        - explanations: object with product indices as keys and explanation strings as values
        - alternative_searches: list of suggested search terms
        """
        
        # Get LLM response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful fashion recommendation assistant."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Extract recommendations
        recommendations_text = response.choices[0].message.content
        recommendations = json.loads(recommendations_text)
        
        # Reorder products based on LLM ranking
        ranked_products = []
        for idx in recommendations.get("ranked_products", []):
            # Adjust for 0-based indexing
            adjusted_idx = idx - 1
            if 0 <= adjusted_idx < len(products):
                product = products[adjusted_idx]
                # Add explanation to product
                product_id = str(adjusted_idx + 1)  # Convert back to 1-based for lookup
                product["explanation"] = recommendations.get("explanations", {}).get(product_id, "")
                ranked_products.append(product)
        
        result = {
            "products": ranked_products,
            "alternative_searches": recommendations.get("alternative_searches", [])
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error refining recommendations: {e}")
        # Return original products without refinement
        return {"products": products, "alternative_searches": []}
