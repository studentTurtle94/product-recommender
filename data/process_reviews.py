#!/usr/bin/env python3
import os
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths - can be overridden with command line arguments
DEFAULT_INPUT_FILE = os.path.join(os.path.dirname(__file__), "Amazon_Fashion.jsonl")
DEFAULT_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "reviews.json")
PRODUCTS_FILE = os.path.join(os.path.dirname(__file__), "fashion_products.jsonl")

def load_sampled_products_and_mappings():
    """Load product IDs and parent-child relationships from the processed products file."""
    if not os.path.exists(PRODUCTS_FILE):
        logger.warning(f"Products file not found at {PRODUCTS_FILE}. Will process all product reviews.")
        return None, {}
    
    try:
        # JSONL format - read line by line
        product_ids = set()
        asin_to_parent = {}
        product_data = {}
        
        with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    product = json.loads(line.strip())
                    
                    # Store the parent_asin and asin (if available)
                    parent_asin = product.get('parent_asin')
                    
                    # Add parent_asin to our set of valid product IDs
                    if parent_asin:
                        product_ids.add(parent_asin)
                        
                        # Store this product's metadata by parent_asin
                        if parent_asin not in product_data:
                            product_data[parent_asin] = product
                            
                        # If any variation's ASIN is present, create a mapping to parent
                        # This could be provided via a direct asin field or variations
                        if product.get('asin'):
                            asin_to_parent[product.get('asin')] = parent_asin
                            product_ids.add(product.get('asin'))
                    
                except json.JSONDecodeError:
                    continue
        
        if not product_ids:
            logger.warning("No product IDs found in the products file. Check the file format.")
            return None, {}
            
        logger.info(f"Loaded {len(product_ids)} product IDs and {len(asin_to_parent)} ASIN mappings from sampled products.")
        return product_ids, asin_to_parent
    
    except Exception as e:
        logger.error(f"Error loading product IDs and mappings: {e}")
        return None, {}

def process_amazon_reviews_file(input_file, output_file, limit_per_product=50, total_limit=None, filter_products=True):
    """Process the Amazon_Fashion.jsonl file to create our reviews database."""
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_file}")
        return False
    
    # Load product IDs and parent ASIN mappings if filtering
    product_ids, asin_to_parent = load_sampled_products_and_mappings()
    
    if filter_products:
        if product_ids:
            logger.info(f"Will filter reviews to match {len(product_ids)} sampled products.")
        else:
            logger.warning("No product IDs loaded for filtering. Processing all reviews.")
    
    try:
        # Create a dictionary to store reviews by product
        # We'll use parent_asin as key when available, otherwise use the product's own ASIN
        processed_reviews = defaultdict(list)
        total_count = 0
        product_count = 0
        filtered_count = 0
        processed_count = 0
        parent_mapped_count = 0
        
        logger.info(f"Processing reviews from {input_file}...")
        
        # Debug - print first 5 product IDs if available
        if product_ids:
            logger.info(f"Sample product IDs: {list(product_ids)[:5]}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if total_limit and processed_count >= total_limit:
                    logger.info(f"Reached total limit of {total_limit} reviews")
                    break
                
                try:
                    review = json.loads(line.strip())
                    total_count += 1
                    
                    # Skip if no text or product ID
                    if not review.get('reviewText') and not review.get('text'):
                        continue
                    
                    # Get the product ID (ASIN)
                    asin = review.get('asin')
                    if not asin:
                        continue
                    
                    # Get parent ASIN if available from our product mappings
                    parent_asin = asin_to_parent.get(asin, asin)
                    
                    # Debug - for first 5 items
                    if total_count <= 5:
                        logger.info(f"Review {total_count}: ASIN={asin}, Parent ASIN={parent_asin}, text_field={'reviewText' if 'reviewText' in review else 'text' if 'text' in review else 'none'}")
                    
                    # If parent ASIN is different from ASIN, we have a mapping
                    if parent_asin != asin:
                        parent_mapped_count += 1
                        
                        # If we're filtering products, check both ASIN and parent ASIN
                        if filter_products and product_ids is not None and asin not in product_ids and parent_asin not in product_ids:
                            filtered_count += 1
                            if total_count <= 5:
                                logger.info(f"Filtered out review for product {asin} / parent {parent_asin} - not in our sample")
                            continue
                    else:
                        # No parent mapping, just check the ASIN if filtering
                        if filter_products and product_ids is not None and asin not in product_ids:
                            filtered_count += 1
                            if total_count <= 5:
                                logger.info(f"Filtered out review for product {asin} - not in our sample")
                            continue
                    
                    # Get review text from either field
                    review_text = review.get('reviewText', '') or review.get('text', '')
                    if not review_text:
                        continue
                    
                    # Use parent_asin as the key when storing the review
                    # This groups all variant products' reviews together
                    if len(processed_reviews[parent_asin]) < limit_per_product:
                        processed_reviews[parent_asin].append({
                            'rating': review.get('overall', review.get('rating', 0)),
                            'title': review.get('summary', review.get('title', '')),
                            'text': review_text,
                            'helpful_vote': review.get('helpful', [0, 0])[0] if isinstance(review.get('helpful'), list) else review.get('helpful_vote', 0),
                            'verified_purchase': review.get('verified', review.get('verified_purchase', False)),
                            'product_asin': asin,  # Include the actual product ASIN
                            'parent_asin': parent_asin  # Include the parent ASIN
                        })
                        processed_count += 1
                        if total_count <= 5:
                            logger.info(f"Added review for product {asin} under parent {parent_asin}")
                    
                    if total_count % 10000 == 0:
                        logger.info(f"Processed {total_count} reviews, kept {processed_count}, filtered {filtered_count}...")
                
                except Exception as e:
                    logger.error(f"Error processing review at line {i}: {e}")
                    continue
        
        # Save to output file
        logger.info(f"Writing processed reviews to {output_file}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_reviews, f)
        
        product_count = len(processed_reviews)
        logger.info(f"Successfully processed {total_count} reviews.")
        logger.info(f"Kept {processed_count} reviews for {product_count} products.")
        logger.info(f"Filtered out {filtered_count} reviews for products not in our sample.")
        logger.info(f"Mapped {parent_mapped_count} reviews to parent ASINs.")
        logger.info(f"Saved processed reviews to {output_file}")
        
        return {
            "total_reviews": total_count,
            "processed_reviews": processed_count,
            "filtered_reviews": filtered_count,
            "parent_mapped_reviews": parent_mapped_count,
            "total_products": product_count,
            "output_file": output_file
        }
    
    except Exception as e:
        logger.error(f"Error processing reviews file: {e}")
        return False

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Process Amazon Fashion reviews JSON file')
    parser.add_argument('--input', '-i', default=DEFAULT_INPUT_FILE, help='Input JSONL file path')
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT_FILE, help='Output JSON file path')
    parser.add_argument('--limit-per-product', '-l', type=int, default=50, help='Maximum reviews per product')
    parser.add_argument('--total-limit', '-t', type=int, default=None, help='Maximum total reviews to process')
    parser.add_argument('--filter-products', '-f', action='store_true', default=True, 
                        help='Filter reviews to only include sampled products')
    parser.add_argument('--no-filter', dest='filter_products', action='store_false',
                        help='Process all reviews without filtering by product')
    
    args = parser.parse_args()
    
    # Process the file
    result = process_amazon_reviews_file(
        args.input, 
        args.output, 
        limit_per_product=args.limit_per_product,
        total_limit=args.total_limit,
        filter_products=args.filter_products
    )
    
    if result:
        logger.info(f"Summary: Processed {result['processed_reviews']} of {result['total_reviews']} reviews")
        logger.info(f"         Mapped {result['parent_mapped_reviews']} reviews to parent ASINs")
        logger.info(f"         Final output contains {result['total_products']} unique product groups")
        return 0
    else:
        logger.error("Processing failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 