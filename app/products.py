import os
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to product data file
PRODUCTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fashion_products.jsonl")

# In-memory product storage
products_db = {}

class ProductItem(BaseModel):
    """Model for a fashion product item."""
    id: str
    parent_asin: Optional[str] = None
    title: str
    description: Optional[str] = None
    price: Optional[float] = None
    categories: List[str] = Field(default_factory=list)
    main_category: Optional[str] = None
    brand: Optional[str] = None
    features: List[str] = Field(default_factory=list)
    image_url: Optional[str] = None
    rating: Optional[float] = None
    has_reviews: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "B07H2FQ6WT",
                "parent_asin": "B07H2FQ6XX",
                "title": "Women's Summer Dress",
                "description": "Comfortable cotton summer dress with floral pattern",
                "price": 29.99,
                "categories": ["Clothing", "Women", "Dresses", "Summer"],
                "main_category": "Women's Dresses",
                "brand": "FashionBrand",
                "features": ["100% Cotton", "Machine Washable", "Various Sizes"],
                "image_url": "https://example.com/image.jpg",
                "rating": 4.5
            }
        }

def load_products():
    """Load products from JSONL file into memory."""
    global products_db
    
    try:
        if not os.path.exists(PRODUCTS_FILE):
            logger.warning(f"Products file not found at {PRODUCTS_FILE}. Starting with empty database.")
            products_db = {}
            return
        
        # Clear existing data
        products_db = {}
        
        # Read JSONL file line by line
        with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse each line as a JSON object
                    product_data = json.loads(line.strip())
                    
                    # Map the fields from fashion_products.jsonl format to our ProductItem model
                    product_dict = {
                        "id": product_data.get("parent_asin", ""),  # Use parent_asin as the primary ID
                        "parent_asin": product_data.get("parent_asin", ""),
                        "title": product_data.get("title", ""),
                        "description": ", ".join(product_data.get("description", [])) if isinstance(product_data.get("description"), list) else product_data.get("description", ""),
                        "price": product_data.get("price", 0.0),
                        "categories": product_data.get("categories", []),
                        "main_category": product_data.get("main_category", ""),
                        "brand": product_data.get("store", ""),
                        "features": product_data.get("features", []),
                        "image_url": product_data.get("images", [{}])[0].get("large", "") if product_data.get("images") else "",
                        "rating": product_data.get("average_rating", 0.0),
                        "has_reviews": True  # Assuming we have reviews for all products
                    }
                    
                    # Skip products without a valid ID
                    if not product_dict["id"]:
                        continue
                    
                    product = ProductItem(**product_dict)
                    products_db[product.id] = product
                    
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON on line {line_num}")
                except Exception as e:
                    logger.error(f"Error loading product on line {line_num}: {e}")
        
        logger.info(f"Loaded {len(products_db)} products from {PRODUCTS_FILE}")
    
    except Exception as e:
        logger.error(f"Error loading products: {e}")
        products_db = {}

def get_product_by_id(product_id: str) -> Optional[ProductItem]:
    """Get a product by its ID."""
    return products_db.get(product_id)

def get_all_products() -> List[ProductItem]:
    """Get all products in the database."""
    return list(products_db.values())

def search_products_by_keyword(keyword: str, limit: int = 10) -> List[ProductItem]:
    """Simple keyword search for products."""
    keyword = keyword.lower()
    results = []
    
    for product in products_db.values():
        if (keyword in product.title.lower() or 
            (product.description and keyword in product.description.lower()) or
            any(keyword in category.lower() for category in product.categories)):
            results.append(product)
            if len(results) >= limit:
                break
    
    return results

def save_products():
    """Save the current products to the JSON file."""
    try:
        # Ensure data directory exists
        os.makedirs(os.path.dirname(PRODUCTS_FILE), exist_ok=True)
        
        # Convert products to dict for JSON serialization
        products_data = [product.dict() for product in products_db.values()]
        
        with open(PRODUCTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(products_data, f, indent=2)
        
        logger.info(f"Saved {len(products_data)} products to {PRODUCTS_FILE}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving products: {e}")
        return False
