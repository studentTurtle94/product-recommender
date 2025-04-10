import json
import random
import os
import math
import numpy as np
import statistics
from pathlib import Path
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
INPUT_FILE = Path("meta_Amazon_Fashion.jsonl")
OUTPUT_FILE = Path("fashion_products.jsonl")
SAMPLE_SIZE = 30000

def get_price_bucket(price):
    """Categorize price into buckets."""
    if price is None or price <= 0:
        return "unknown"
    elif price < 20:
        return "budget"
    elif price < 50:
        return "low_mid"
    elif price < 100:
        return "high_mid"
    else:
        return "premium"

def get_rating_bucket(rating):
    """Categorize rating into buckets."""
    if rating is None or rating <= 0:
        return "unknown"
    elif rating < 2:
        return "low"
    elif rating < 3.5:
        return "medium"
    elif rating < 4.5:
        return "high"
    else:
        return "excellent"

def get_description_length_bucket(desc):
    """Categorize description length into buckets."""
    if not desc:
        return "none"
    
    # Calculate total length of description text
    desc_text = ' '.join(desc) if isinstance(desc, list) else desc
    length = len(desc_text)
    
    if length == 0:
        return "none"
    elif length < 100:
        return "short"
    elif length < 300:
        return "medium"
    else:
        return "long"

def get_features_bucket(features):
    """Categorize number of features into buckets."""
    if not features:
        return "none"
    
    count = len(features)
    if count == 0:
        return "none"
    elif count < 3:
        return "few"
    elif count < 6:
        return "medium"
    else:
        return "many"

def get_ratings_count_bucket(count):
    """Categorize number of ratings into buckets."""
    if count is None or count <= 0:
        return "unknown"
    elif count < 10:
        return "few"
    elif count < 50:
        return "medium"
    elif count < 200:
        return "high"
    else:
        return "very_high"

def has_image_with_url(images):
    """Check if the product has at least one valid image URL."""
    if not images:
        return False
    
    for img in images:
        if img.get('large') or img.get('hi_res'):
            return True
    
    return False

def extract_price(product):
    """Extract and convert price to float, handling various formats."""
    price = product.get('price')
    
    # If price is already a number, return it
    if isinstance(price, (int, float)) and price > 0:
        return float(price)
        
    # Handle case where price is None
    if price is None:
        return None
    
    # Try to extract numeric value from string
    if isinstance(price, str):
        # Remove currency symbols and commas
        price = price.replace('$', '').replace(',', '').strip()
        try:
            return float(price)
        except ValueError:
            return None
    
    return None

def load_and_filter_data():
    """Load, filter, and preprocess the Amazon Fashion dataset."""
    logger.info(f"Processing {INPUT_FILE}")
    
    if not INPUT_FILE.exists():
        logger.error(f"Input file {INPUT_FILE} not found!")
        return False
    
    # Load all products
    products = []
    valid_products = []
    count = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
            if count % 10000 == 0:
                logger.info(f"Processed {count} lines...")
            
            try:
                product = json.loads(line.strip())
                products.append(product)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse line {count}")
    
    logger.info(f"Loaded {len(products)} products from file")
    
    # Filter for products with all required fields
    for product in products:
        # Skip products without a title
        if not product.get('title'):
            continue
            
        # Extract and validate price
        price = extract_price(product)
        if price is None:
            # For products without price, use a default value
            price = 19.99  # Default price for products missing one
            
        # Make sure the product has a main category
        main_category = product.get('main_category')
        if not main_category:
            continue
            
        # Check if product has at least one image
        if not has_image_with_url(product.get('images', [])):
            continue
            
        # Format product into our schema
        valid_product = {
            "main_category": main_category,
            "title": product.get('title', ''),
            "average_rating": float(product.get('average_rating', 0.0)) if product.get('average_rating') is not None else 0.0,
            "rating_number": int(product.get('rating_number', 0)) if product.get('rating_number') is not None else 0,
            "features": product.get('features', []) if isinstance(product.get('features'), list) else [],
            "description": product.get('description', []) if isinstance(product.get('description'), list) else [],
            "price": price,
            "store": product.get('store', 'Amazon Fashion'),
            "categories": product.get('categories', []) if isinstance(product.get('categories'), list) else [],
            "details": product.get('details', {}),
            "parent_asin": product.get('parent_asin', ''),
            "images": product.get('images', []),
            "videos": product.get('videos', []),
        }
        
        valid_products.append(valid_product)
    
    logger.info(f"Found {len(valid_products)} valid products")
    
    # If no valid products found, create some sample data
    if not valid_products:
        logger.warning("No valid products found, creating sample data...")
        return create_sample_data()
    
    # If we have fewer valid products than our sample size, use all of them
    if len(valid_products) <= SAMPLE_SIZE:
        logger.info(f"Using all {len(valid_products)} valid products")
        return write_products_to_file(valid_products)
    
    # Analyze and stratify the data
    return stratified_sample(valid_products, SAMPLE_SIZE)

def create_sample_data():
    """Create sample data when no valid products are found."""
    sample_products = [
        {
            "main_category": "Clothing",
            "title": "Summer Beach T-Shirt",
            "average_rating": 4.5,
            "rating_number": 120,
            "features": ["100% Cotton", "Lightweight", "Perfect for beach"],
            "description": ["A comfortable t-shirt ideal for summer beach outings."],
            "price": 24.99,
            "images": [{"large": "https://example.com/tshirt.jpg"}],
            "store": "Fashion Store",
            "categories": ["Clothing", "Men's", "T-Shirts"],
            "details": {"Brand": "BeachWear", "Material": "Cotton", "Size": "M"},
            "parent_asin": "sample1",
            "videos": []
        },
        {
            "main_category": "Accessories",
            "title": "Wide-Brim Sun Hat",
            "average_rating": 4.2,
            "rating_number": 85,
            "features": ["UV Protection", "Foldable", "Wide brim"],
            "description": ["Stylish sun hat with wide brim for maximum sun protection."],
            "price": 34.99,
            "images": [{"large": "https://example.com/hat.jpg"}],
            "store": "Fashion Store",
            "categories": ["Accessories", "Women's", "Hats"],
            "details": {"Brand": "SunStyle", "Material": "Straw", "Size": "One Size"},
            "parent_asin": "sample2",
            "videos": []
        },
        {
            "main_category": "Shoes",
            "title": "Men's Casual Loafers",
            "average_rating": 3.8,
            "rating_number": 210,
            "features": ["Slip-on design", "Memory foam insole", "Flexible outsole"],
            "description": ["Comfortable casual loafers perfect for everyday wear."],
            "price": 49.99,
            "images": [{"large": "https://example.com/loafers.jpg"}],
            "store": "Shoe Emporium",
            "categories": ["Shoes", "Men's", "Loafers"],
            "details": {"Brand": "ComfortStep", "Material": "Synthetic", "Size": "10"},
            "parent_asin": "sample3",
            "videos": []
        }
    ]
    
    # Write sample products to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for product in sample_products:
            f.write(json.dumps(product) + '\n')
    
    logger.info(f"Created {len(sample_products)} sample products")
    return True

def write_products_to_file(products):
    """Write products to output file."""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for product in products:
            # Remove the index field if it exists
            product.pop('index', None)
            f.write(json.dumps(product) + '\n')
    
    logger.info(f"Wrote {len(products)} products to {OUTPUT_FILE}")
    return True

def stratified_sample(valid_products, sample_size):
    """
    Perform stratified sampling across multiple dimensions to ensure a diverse
    and representative sample of products.
    """
    logger.info(f"Starting stratified sampling to select {sample_size} products...")
    
    # Step 1: Analyze distributions in the original dataset
    main_categories = defaultdict(list)
    stores = defaultdict(list)
    price_buckets = defaultdict(list)
    rating_buckets = defaultdict(list)
    ratings_count_buckets = defaultdict(list)
    description_length_buckets = defaultdict(list)
    features_buckets = defaultdict(list)
    
    # Get distribution of products across different attributes
    for i, product in enumerate(valid_products):
        # Add index to each product for reference
        product['index'] = i
        
        main_cat = product['main_category']
        store = product['store']
        price_bucket = get_price_bucket(product['price'])
        rating_bucket = get_rating_bucket(product['average_rating'])
        ratings_count_bucket = get_ratings_count_bucket(product['rating_number'])
        desc_bucket = get_description_length_bucket(product['description'])
        features_bucket = get_features_bucket(product['features'])
        
        main_categories[main_cat].append(i)
        stores[store].append(i)
        price_buckets[price_bucket].append(i)
        rating_buckets[rating_bucket].append(i)
        ratings_count_buckets[ratings_count_bucket].append(i)
        description_length_buckets[desc_bucket].append(i)
        features_buckets[features_bucket].append(i)
    
    # Log distributions for the top categories
    top_categories = {k: len(v) for k, v in sorted(main_categories.items(), key=lambda x: len(x[1]), reverse=True)[:10]}
    logger.info(f"Top categories: {top_categories}")
    logger.info(f"Price buckets: {dict([(k, len(v)) for k, v in price_buckets.items()])}")
    logger.info(f"Rating buckets: {dict([(k, len(v)) for k, v in rating_buckets.items()])}")
    logger.info(f"Ratings count buckets: {dict([(k, len(v)) for k, v in ratings_count_buckets.items()])}")
    logger.info(f"Description length buckets: {dict([(k, len(v)) for k, v in description_length_buckets.items()])}")
    logger.info(f"Features buckets: {dict([(k, len(v)) for k, v in features_buckets.items()])}")
    
    # Step 2: Calculate proportions for stratified sampling
    total_products = len(valid_products)
    
    # Calculate how many products to include from each main category
    category_counts = {}
    for category, indices in main_categories.items():
        proportion = len(indices) / total_products
        category_counts[category] = max(1, math.ceil(proportion * sample_size))
    
    # Adjust to ensure we hit our target sample size
    total_allocated = sum(category_counts.values())
    if total_allocated > sample_size:
        # Scale down proportionally if we've allocated too many
        scale_factor = sample_size / total_allocated
        for category in category_counts:
            category_counts[category] = max(1, math.floor(category_counts[category] * scale_factor))
    
    # Step 3: Perform the stratified sampling
    selected_indices = set()
    
    # Precompute frequency tables for each bucket type to optimize the scoring function
    price_bucket_freq = defaultdict(int)
    rating_bucket_freq = defaultdict(int)
    
    # Optimize scoring by using batch processing for large categories
    def process_category_batch(category, count, indices):
        nonlocal selected_indices, price_bucket_freq, rating_bucket_freq
        
        if len(indices) <= count:
            # If we have fewer products than needed, use them all
            selected_indices.update(indices)
            return
        
        # Calculate scores in batches for large categories
        remaining = count
        category_selected = set()
        
        # Initial random selection of 20% to seed diversity
        if count > 10:
            seed_count = min(int(count * 0.2), 50)
            seed_indices = random.sample(indices, min(seed_count, len(indices)))
            category_selected.update(seed_indices)
            remaining -= len(seed_indices)
            
            # Update frequency tables for the selected items
            for idx in seed_indices:
                product = valid_products[idx]
                price_bucket_freq[get_price_bucket(product['price'])] += 1
                rating_bucket_freq[get_rating_bucket(product['average_rating'])] += 1
        
        # Select the rest based on diversity scores
        if remaining > 0:
            # Calculate initial scores for all products
            scores = []
            batch_size = 1000  # Process in batches to avoid memory issues
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_scores = [(idx, score_product(idx, category, category_selected)) for idx in batch_indices if idx not in category_selected]
                scores.extend(batch_scores)
            
            # Sort by score, highest first
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take the top scoring products
            top_indices = [idx for idx, _ in scores[:remaining]]
            category_selected.update(top_indices)
            
            # Update frequency tables
            for idx in top_indices:
                product = valid_products[idx]
                price_bucket_freq[get_price_bucket(product['price'])] += 1
                rating_bucket_freq[get_rating_bucket(product['average_rating'])] += 1
        
        # Add to our global selected set
        selected_indices.update(category_selected)
    
    # Function to score a product based on our diversity criteria - optimized version
    def score_product(index, category, already_selected):
        # Products we've already selected get a negative score
        if index in already_selected:
            return -1000
            
        product = valid_products[index]
        score = 0
        
        # Reward diversity in price buckets
        price_bucket = get_price_bucket(product['price'])
        score -= price_bucket_freq[price_bucket] * 2  # Penalize if we already have many products in this price bucket
        
        # Reward diversity in ratings
        rating_bucket = get_rating_bucket(product['average_rating'])
        score -= rating_bucket_freq[rating_bucket] * 2
        
        # Reward products with descriptions
        desc_bucket = get_description_length_bucket(product['description'])
        if desc_bucket == "none":
            score -= 5  # Penalize products without descriptions
        elif desc_bucket == "short":
            score -= 2  # Slightly penalize very short descriptions
        
        # Reward products with images
        if not product['images']:
            score -= 10  # Heavily penalize products without images
        
        # Reward products with features
        features_bucket = get_features_bucket(product['features'])
        if features_bucket == "none":
            score -= 3  # Penalize products without features
        
        # Reward products with ratings
        ratings_count_bucket = get_ratings_count_bucket(product['rating_number'])
        if ratings_count_bucket == "few":
            score -= 2  # Slightly penalize products with very few ratings
            
        # Add a small random factor to avoid deterministic selection
        score += random.uniform(-1, 1)
        
        return score
    
    # Process categories in batches, largest first for better distribution
    categories_sorted = sorted(category_counts.items(), key=lambda x: len(main_categories[x[0]]), reverse=True)
    
    logger.info(f"Starting to select products by category...")
    progress_step = max(1, len(categories_sorted) // 10)  # Log progress every 10%
    
    for i, (category, count) in enumerate(categories_sorted):
        if i % progress_step == 0:
            logger.info(f"Processing category {i+1}/{len(categories_sorted)}: {category} (allocating {count} products)")
        
        category_indices = main_categories[category]
        if not category_indices:
            continue
            
        process_category_batch(category, count, category_indices)
    
    # If we still don't have enough products, add more based on diversity
    if len(selected_indices) < sample_size:
        remaining = sample_size - len(selected_indices)
        logger.info(f"Still need {remaining} more products to reach target sample size")
        
        all_indices = set(range(len(valid_products)))
        unselected = list(all_indices - selected_indices)
        
        if unselected:
            # Take a random sample for large datasets to avoid memory issues
            if len(unselected) > 10000:
                unselected = random.sample(unselected, 10000)
                
            # Calculate scores for unselected products
            scores = [(idx, score_product(idx, None, selected_indices)) for idx in unselected]
            
            # Sort by score, highest first
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Add the top-scoring products
            additional = [idx for idx, _ in scores[:remaining]]
            selected_indices.update(additional)
    
    # Final selected products
    filtered_products = [valid_products[idx] for idx in selected_indices]
    
    # Analyze final distribution
    final_categories = defaultdict(int)
    final_price_buckets = defaultdict(int)
    final_rating_buckets = defaultdict(int)
    
    for product in filtered_products:
        final_categories[product['main_category']] += 1
        final_price_buckets[get_price_bucket(product['price'])] += 1
        final_rating_buckets[get_rating_bucket(product['average_rating'])] += 1
    
    # Log final distribution summary
    top_final_categories = {k: v for k, v in sorted(final_categories.items(), key=lambda x: x[1], reverse=True)[:10]}
    logger.info(f"Final top categories: {top_final_categories}")
    logger.info(f"Final price buckets: {dict(final_price_buckets)}")
    logger.info(f"Final rating buckets: {dict(final_rating_buckets)}")
    
    return write_products_to_file(filtered_products)

if __name__ == "__main__":
    if load_and_filter_data():
        logger.info("Data processing completed successfully!")
    else:
        logger.error("Data processing failed!") 