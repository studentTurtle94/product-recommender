import requests
import json
import logging

# Configure logging (for errors and warnings)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Base URL (assuming your server runs locally on port 8000)
BASE_URL = "http://127.0.0.1:8000"
SEARCH_ENDPOINT = f"{BASE_URL}/search"

def test_semantic_search(query: str, limit: int = 5):
    """
    Tests the semantic search endpoint of the API.

    Args:
        query: The search query string.
        limit: The maximum number of results to return.
    """
    params = {
        "query": query,
        "limit": limit
    }
    print(f"Testing search endpoint with query: '{query}', limit: {limit}")

    try:
        response = requests.get(SEARCH_ENDPOINT, params=params, timeout=30) # Add timeout

        # Check if the request was successful
        if response.status_code == 200:
            try:
                results = response.json()
                products = results.get("products", [])
                message = results.get("message")

                if products:
                    print(f"Successfully received {len(products)} search results:")
                    for i, product in enumerate(products):
                        # Extract relevant info - adjust based on your ProductItem model
                        title = product.get('title', 'N/A')
                        price = product.get('price', 'N/A')
                        product_id = product.get('id', 'N/A')
                        print(f"  {i+1}. ID: {product_id}, Title: {title}, Price: ${price:.2f}" if isinstance(price, (int, float)) else f"  {i+1}. ID: {product_id}, Title: {title}, Price: {price}")
                elif message:
                     print(f"API returned a message: {message}")
                else:
                    print("Search successful but received no products or message.")

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response. Status Code: {response.status_code}")
                print(f"Raw response text:\n{response.text}")
            except Exception as e:
                 logger.error(f"An error occurred processing the response: {e}")
                 print(f"Raw response text:\n{response.text}")

        else:
            logger.error(f"Search request failed with status code: {response.status_code}")
            try:
                # Try to print error details if available in JSON format
                error_details = response.json()
                print(f"Error details: {error_details}")
            except json.JSONDecodeError:
                # Otherwise, print raw text
                print(f"Raw response text:\n{response.text}")

    except requests.exceptions.ConnectionError:
        logger.error(f"Connection Error: Could not connect to the API at {SEARCH_ENDPOINT}. Is the server running?")
    except requests.exceptions.Timeout:
        logger.error(f"Timeout Error: The request to {SEARCH_ENDPOINT} timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"An unexpected error occurred during the request: {e}")

if __name__ == "__main__":
    # --- Examples ---
    test_query_1 = "comfortable summer dress"
    test_query_2 = "men's running shoes"
    test_query_3 = "Crossbody Leather bag"
    test_query_4 = "something blue" # A more abstract query
    test_query_5 = "asdlkfjaslkdfj" # A query likely to yield no results

    print("-" * 30)
    test_semantic_search(test_query_1, limit=3)
    print("-" * 30)
    test_semantic_search(test_query_2, limit=5)
    print("-" * 30)
    test_semantic_search(test_query_3) # Default limit
    print("-" * 30)
    test_semantic_search(test_query_4, limit=2)
    print("-" * 30)
    test_semantic_search(test_query_5)
    print("-" * 30) 