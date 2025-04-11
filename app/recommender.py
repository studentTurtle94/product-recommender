import logging
import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
import numpy as np
import base64
from pydantic import BaseModel, Field, ValidationError

from .products import get_product_by_id, ProductItem
from .embeddings import get_embedding, search_similar_products

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("This is a test debug message from app/recommender.py.")

# Global OpenAI client
client = None
# Define the OpenAI model name as a constant
OPENAI_MODEL_NAME = "gpt-4o"

# --- Pydantic Model for Refinement Response ---
class RefinementResponse(BaseModel):
    ranked_products: List[int] = Field(..., description="List of product indices (1-based) in order of relevance")
    explanations: Dict[str, str] = Field(..., description="Object with product indices (as strings) as keys and explanation strings as values")
    alternative_searches: List[str] = Field(..., description="List of suggested alternative search terms")

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
    1. Send text + image to LLM for analysis and refined search query generation.
    2. Use the generated query for semantic search via search_similar_products.
    """
    if not client:
        raise ValueError("Recommender not initialized. Call init_recommender first.")

    try:
        logger.info(f"Performing multimodal search for text: '{query_text[:50]}...' and image (size: {len(image_data)} bytes)")

        # Encode image to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')

        # --- Call LLM for analysis ---
        # This prompt is updated to prioritize the text query's intent, using the image as context.
        vision_prompt = f"""
        Analyze the user's text query and the provided image of a fashion item.
        Your goal is to generate an effective search query for a vector database containing fashion product descriptions.
        The search query should help find items that fulfill the user's textual request, considering the item in the image as context.

        1. Identify the main item shown in the image (e.g., 'black leather jacket', 'floral summer dress', 'running shoes').
        2. Understand the user's request from the text query (e.g., 'find matching pants', 'suggest accessories', 'look for similar style shoes').
        3. Combine these insights to create a search query.
           - If the user asks for items *related* to the image (e.g., accessories, matching items), the query should focus on the requested item type, incorporating style/color cues from the image context.
           - If the user asks for items *similar* to the image, the query should describe the item in the image, refined by the text query.

        User Text Query: "{query_text}"

        Examples:
        - Image: Red dress, Text: "accessories for this" -> Query: "elegant red earrings" or "gold jewelry formal"
        - Image: Blue jeans, Text: "find a top to wear with these" -> Query: "casual white top" or "white blouse"
        - Image: Black boots high heels, Text: "show me similar boots but brown" -> Query: "brown leather high heel boots"

        Output ONLY the generated search query string, nothing else. Ensure the output is concise and descriptive for a search engine.
        Do not use the word "similar" or "matching" in the query, query should be very specific to closest fashion item that fits user request.
        """
        
        # TODO: Determine appropriate image type (e.g., 'image/jpeg', 'image/png')
        # For now, assuming JPEG, but ideally, detect or get from frontend.
        image_type = "image/jpeg"

        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
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
            store=True,
            max_tokens=100 # Limit response length for the query
        )

        generated_query = response.choices[0].message.content.strip()
        logger.info(f"LLM generated search query: {generated_query}")

        if not generated_query:
             logger.warning("LLM did not return a usable query. Falling back to original text query.")
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

    # Prepare fallback result in case of errors
    fallback_result = {"products": products, "alternative_searches": []}

    try:
        # Create a detailed prompt with all product information
        products_text = ""
        for i, product in enumerate(products):
            price_str = f"${product['price']:.2f}" if product.get('price') else "Price not available"
            products_text += f"Product {i+1}: {product['title']} - {price_str}\n"
            products_text += f"Categories: {', '.join(product['categories'])}\n"
            # Limit description length to avoid overly long prompts
            description = product.get('description', 'No description available')
            products_text += f"Features: {description[:200]}{'...' if len(description) > 200 else ''}\n\n"

        prompt = f"""
        You are a fashion recommendation assistant. A customer has made the following request:

        "{query}"

        Based on this request, I found these products that might be relevant:

        {products_text}

        Please analyze these products and the customer's request to:
        1. Rank the products from most to least relevant for this specific request.
        2. Provide a brief explanation for each recommendation (why it matches their request).
        3. Suggest one or two additional relevant search terms the customer might want to try.

        Format your response STRICTLY as a JSON object with these exact keys:
        - "ranked_products": A list of product indices (1-based integers, e.g., [3, 1, 2]) in order of relevance.
        - "explanations": An object where keys are the product indices as strings (e.g., "1", "2", "3") and values are the explanation strings.
        - "alternative_searches": A list of suggested search term strings (e.g., ["formal black shoes", "silk evening scarf"]).

        Ensure the output is ONLY the JSON object, nothing else before or after.
        """

        # Get LLM response
        logger.debug(f"Sending prompt to LLM for refinement: {prompt[:500]}...")
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful fashion recommendation assistant. Output ONLY valid JSON matching the requested format."}, # Added instruction about JSON output
                {"role": "user", "content": prompt}
            ],
            store=True,
            response_format={"type": "json_object"}
        )

        # Extract and validate recommendations using Pydantic
        recommendations_text = response.choices[0].message.content
        logger.debug(f"Received raw response from LLM: {recommendations_text}")

        try:
            # Use model_validate_json since the input is a JSON string
            recommendations_model = RefinementResponse.model_validate_json(recommendations_text)
            logger.info("Successfully parsed and validated LLM response for refinement.")
        except ValidationError as ve:
            logger.error(f"LLM response failed Pydantic validation: {ve}")
            logger.error(f"Invalid response text: {recommendations_text}")
            return fallback_result # Return original products if validation fails
        except json.JSONDecodeError as je:
            logger.error(f"LLM response was not valid JSON: {je}")
            logger.error(f"Invalid response text: {recommendations_text}")
            return fallback_result # Return original products if JSON is invalid

        # Reorder products based on LLM ranking
        ranked_products = []
        if recommendations_model.ranked_products:
            for idx in recommendations_model.ranked_products:
                # Adjust for 0-based indexing for the products list
                adjusted_idx = idx - 1
                if 0 <= adjusted_idx < len(products):
                    product = products[adjusted_idx]
                    # Add explanation to product - explanation keys are 1-based strings
                    product_id_str = str(idx)
                    product["explanation"] = recommendations_model.explanations.get(product_id_str, "")
                    ranked_products.append(product)
                else:
                    logger.warning(f"LLM ranked product index {idx} (adjusted: {adjusted_idx}) is out of bounds for {len(products)} products.")
        else:
            logger.warning("LLM returned an empty list for ranked_products. Returning original order.")
            # If ranking fails or is empty, potentially return original order or handle as needed
            ranked_products = products # Fallback to original order if ranking is empty
            # Add empty explanations if needed, or handle as appropriate
            for p in ranked_products:
                p["explanation"] = "" # Ensure key exists

        result = {
            "products": ranked_products,
            "alternative_searches": recommendations_model.alternative_searches
        }

        return result

    except Exception as e:
        logger.error(f"Unexpected error during recommendation refinement: {e}", exc_info=True)
        return fallback_result # Return original products on any unexpected error
