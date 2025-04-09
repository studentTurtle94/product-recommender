import logging
import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
import numpy as np
import base64

from .products import get_product_by_id, ProductItem
from .embeddings import get_embedding, search_similar_products

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("This is a test debug message from app/recommender.py.")

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

async def multimodal_search(query_text: str, image_data: bytes, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform multimodal search using text and image.
    1. Send text + image to GPT-4o for analysis and refined search query generation.
    2. Use the generated query for semantic search via search_similar_products.
    """
    if not client:
        raise ValueError("Recommender not initialized. Call init_recommender first.")

    try:
        logger.info(f"Performing multimodal search for text: '{query_text[:50]}...' and image (size: {len(image_data)} bytes)")

        # Encode image to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')

        # --- Call GPT-4o for analysis ---
        # This prompt asks the model to describe the item/style and suggest a search query.
        # Adapt the prompt based on desired output format/detail.
        vision_prompt = f"""
        Analyze the following image and the user's text query.
        The goal is to find similar fashion products from an e-commerce catalog.
        Describe the key visual elements (item type, style, color, pattern, fabric if discernible) and incorporate the user's text request.
        Based on this combined understanding, generate a concise and descriptive search query suitable for a vector database search focused on product descriptions.
        The query should capture the essence of the user's request, merging visual details with textual refinements.

        User Text Query: "{query_text}"

        Output ONLY the generated search query string, nothing else.
        Example output: "blue summer asymetrical beach dress"
        """
        
        # TODO: Determine appropriate image type (e.g., 'image/jpeg', 'image/png')
        # For now, assuming JPEG, but ideally, detect or get from frontend.
        image_type = "image/jpeg"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{image_type};base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=100 # Limit response length for the query
        )

        generated_query = response.choices[0].message.content.strip()
        logger.info(f"GPT-4o generated search query: {generated_query}")

        if not generated_query:
             logger.warning("GPT-4o did not return a usable query. Falling back to original text query.")
             generated_query = query_text # Fallback or could handle differently

        # --- Perform semantic search using the generated query ---
        found_products_info = await search_similar_products(query=generated_query, top_k=top_k)

        # Get full product information and add the rating
        products = []
        for product_info in found_products_info:
            product_id = product_info['original_id']
            product = get_product_by_id(product_id)
            if product:
                product_dict = product.dict()
                product_dict['rating'] = product_info.get('rating', product.rating if product.rating is not None else 0.0)
                products.append(product_dict)

        logger.info(f"Found {len(products)} products matching multimodal query with ratings")
        return products

    except Exception as e:
        logger.error(f"Error in multimodal search: {e}", exc_info=True)
        return []

async def semantic_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform semantic search on products based on natural language query."""
    if not client:
        raise ValueError("Recommender not initialized. Call init_recommender first.")
    
    try:
        logger.info(f"Performing semantic search for query: {query}")
        
        # search_similar_products now returns a list of dicts with 'original_id' and 'rating'
        found_products_info = await search_similar_products(query, top_k=top_k)
        
        # Get full product information and add the rating
        products = []
        for product_info in found_products_info:
            product_id = product_info['original_id']
            product = get_product_by_id(product_id)
            if product:
                product_dict = product.dict()
                # Add the rating fetched from Qdrant payload
                product_dict['rating'] = product_info.get('rating', product.rating if product.rating is not None else 0.0)
                products.append(product_dict)
        
        logger.info(f"Found {len(products)} products matching query with ratings")
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
