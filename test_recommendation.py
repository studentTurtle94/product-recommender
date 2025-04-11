import os
import asyncio
import json
import logging
from dotenv import load_dotenv
import httpx
import mimetypes # To guess image content type
from typing import List, Dict, Optional

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
RECOMMEND_MULTIMODAL_ENDPOINT = f"{API_URL}/recommend-multimodal"

# Sample image paths (assuming script runs from workspace root)
IMAGE_DIR = "data/sample_imgs"
FLORAL_DRESS_IMG = os.path.join(IMAGE_DIR, "floral_dress.jpg")
EARRINGS_GREEN_IMG = os.path.join(IMAGE_DIR, "earrings_green.jpg")
SHOES_GREEN_HEELS_IMG = os.path.join(IMAGE_DIR, "shoes_green_heels.jpg")

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
            logger.info(f"Received {len(results.get('products', []))} recommended products. Evaluating...")

            # Call evaluator
            evaluation = await evaluate_recommendation(
                openai_client=client, # You might need to adjust how the client is passed/initialized
                original_query=query,
                results=results,
                test_type="text"
            )
            # Log evaluation instead of raw results
            logger.info(f"Evaluation for query '{query}': {evaluation}")

            return results # Still return original results if needed elsewhere
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

async def test_recommend_multimodal(query_text: str, image_path: str):
    """Test the recommend-multimodal endpoint with text and an image."""
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}. Skipping multimodal test.")
        return {}

    async with httpx.AsyncClient(timeout=30.0) as client: # Increased timeout for potential LLM processing
        logger.info(f"Testing multimodal recommendation with query: '{query_text}' and image: {os.path.basename(image_path)}")
        
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
                image_filename = os.path.basename(image_path)
                content_type, _ = mimetypes.guess_type(image_path)
                if not content_type:
                    content_type = 'application/octet-stream' # Default if type cannot be guessed
            
            files = {'image_file': (image_filename, image_data, content_type)}
            data = {"query_text": query_text, "limit": 5}
            
            response = await client.post(RECOMMEND_MULTIMODAL_ENDPOINT, data=data, files=files)

            if response.status_code == 200:
                results = response.json()
                image_basename = os.path.basename(image_path)
                logger.info(f"Received {len(results.get('products', []))} multimodal recommended products. Evaluating...")

                # Call evaluator
                evaluation = await evaluate_recommendation(
                    openai_client=client, # Adjust client passing
                    original_query=query_text,
                    results=results,
                    test_type="multimodal",
                    image_filename=image_basename
                )
                # Log evaluation
                logger.info(f"Evaluation for query '{query_text}' with image '{image_basename}': {evaluation}")

                return results
            else:
                logger.error(f"Multimodal Recommendation failed: {response.status_code} - {response.text}")
                return {"products": [], "alternative_searches": []}

        except FileNotFoundError:
            logger.error(f"Image file not found at path: {image_path}")
            return {}
        except Exception as e:
            logger.error(f"Error during multimodal test for {image_path}: {e}", exc_info=True)
            return {}

# Helper function to format product list for evaluation prompt
def format_products_for_eval(products: List[Dict[str, any]]) -> str:
    if not products:
        return "No products were recommended."
    output = ""
    for i, p in enumerate(products, 1):
        title = p.get('title', 'N/A')
        price = f"${p.get('price', 0):.2f}" if p.get('price') is not None else "Price N/A"
        explanation = p.get('explanation', '').strip()
        output += f"  {i}. Title: {title} ({price})"
        if explanation and explanation != 'No explanation':
            output += f"\\n     Explanation: {explanation}"
        output += "\\n"
    return output.strip()

# Helper function to format alternative searches for evaluation prompt
def format_alternatives_for_eval(alternatives: List[str]) -> str:
    if not alternatives:
        return "None"
    return ", ".join([f"'{alt}'" for alt in alternatives])

# The new evaluation function
async def evaluate_recommendation(
    openai_client: httpx.AsyncClient, # Reusing httpx client configured for OpenAI if needed, or initialize separate OpenAI client
    original_query: str,
    results: Dict[str, any],
    test_type: str, # "text" or "multimodal"
    image_filename: Optional[str] = None
) -> Dict[str, any]:
    """Use an LLM to evaluate the quality of the recommendation."""
    
    products = results.get('products', [])
    alternatives = results.get('alternative_searches', [])
    
    # Prepare context for the evaluation prompt
    context = f"Original User Query: '{original_query}'\\n"
    if test_type == "multimodal" and image_filename:
        context += f"Context Image: User uploaded an image named '{image_filename}'. The query refers to this image.\\n"
    else:
        context += f"Context: This was a text-only query.\\n"
        
    recommendations_summary = format_products_for_eval(products)
    alternatives_summary = format_alternatives_for_eval(alternatives)

    eval_prompt = f"""
You are an AI assistant evaluating the quality of a fashion recommendation system.
Evaluate the following recommendation response based on the user's original query and context.

{context}
---
Recommendation Response Received:
Recommended Products:
{recommendations_summary}

Suggested Alternative Searches: {alternatives_summary}
---
Evaluation Criteria:
1. Relevance: How relevant are the recommended products to the user's specific query and context (image included)? (Score 1-5, 5=Highly Relevant)
2. Explanation Quality: How helpful and accurate are the explanations provided for each product (if any)? (Score 1-5, 5=Very Helpful/Accurate, N/A if no explanations)
3. Alternative Search Relevance: Are the suggested alternative searches logical next steps or relevant explorations based on the query? (Score 1-5, 5=Highly Relevant, N/A if none)
4. Overall Coherence: Does the overall response make sense and address the user's likely intent?

Provide your evaluation as a JSON object with the following keys:
- "relevance_score": integer (1-5)
- "explanation_score": integer (1-5) or null
- "alternative_score": integer (1-5) or null
- "overall_coherence": boolean (true/false)
- "rationale": string (brief justification for your scores and coherence assessment)

Example JSON Output:
{{
  "relevance_score": 4,
  "explanation_score": 5,
  "alternative_score": 3,
  "overall_coherence": true,
  "rationale": "Recommendations are mostly relevant, with excellent explanations. Alternatives are okay but a bit generic."
}}

Please provide ONLY the JSON object as your response.
"""
    # Initialize OpenAI client (you might want to do this globally once)
    # Ensure OPENAI_API_KEY is set in your environment
    from openai import OpenAI as SyncOpenAI # Use sync client here for simplicity, or adapt test funcs to pass async client
    sync_openai_client = SyncOpenAI()

    try:
        # Using GPT-4o for evaluation
        response = sync_openai_client.chat.completions.create(
             model="gpt-4o",
             messages=[
                 {"role": "system", "content": "You are an AI evaluator."},
                 {"role": "user", "content": eval_prompt}
             ],
             response_format={"type": "json_object"},
             temperature=0.2 # Lower temperature for more consistent evaluation
        )
        eval_result_text = response.choices[0].message.content
        eval_json = json.loads(eval_result_text)
        logger.info(f"Evaluation Result: {eval_json}")
        return eval_json
    except Exception as e:
        logger.error(f"LLM Evaluation failed: {e}", exc_info=True)
        return {"error": str(e)} # Return error structure

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

    # --- Multimodal Tests ---
    logger.info("\n" + "="*80)
    logger.info("STARTING MULTIMODAL TESTS")
    logger.info("="*80)

    multimodal_test_cases = [
        {"query": "accessories for this dress", "image": FLORAL_DRESS_IMG},
        {"query": "a necklace that matches these earrings", "image": EARRINGS_GREEN_IMG},
        {"query": "find a clutch bag that goes with these shoes", "image": SHOES_GREEN_HEELS_IMG},
        {"query": "similar dresses but in blue", "image": FLORAL_DRESS_IMG}, # Test finding similar items with modification
        {"query": "casual shoes to wear with this dress", "image": FLORAL_DRESS_IMG} # Test finding related but different category
    ]

    for test_case in multimodal_test_cases:
        logger.info("\n" + "-"*80)
        logger.info(f"MULTIMODAL TEST - Query: '{test_case['query']}', Image: {os.path.basename(test_case['image'])}")
        logger.info("-"*80)
        await test_recommend_multimodal(query_text=test_case['query'], image_path=test_case['image'])
        logger.info("-"*80 + "\n")

if __name__ == "__main__":
    logger.info("Starting recommendation system tests...")
    asyncio.run(run_tests())
    logger.info("Tests completed!") 