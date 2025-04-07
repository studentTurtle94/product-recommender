import json
import os
from pathlib import Path
import logging
from collections import defaultdict
import re
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to review data
REVIEWS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "reviews.json")

# Global store for product reviews
product_reviews = defaultdict(list)
review_summaries = {}

def load_reviews():
    """Load review data from JSON file."""
    global product_reviews
    
    try:
        if not os.path.exists(REVIEWS_FILE):
            logger.warning(f"Reviews file not found at {REVIEWS_FILE}. Starting with empty review database.")
            return {}
        
        with open(REVIEWS_FILE, 'r', encoding='utf-8') as f:
            reviews_data = json.load(f)
        
        # Reset the product reviews dictionary
        product_reviews = defaultdict(list)
        
        # The reviews.json file has keys as product IDs (parent_asin) and values as lists of reviews
        for product_id, reviews in reviews_data.items():
            product_reviews[product_id] = reviews
        
        logger.info(f"Loaded reviews for {len(product_reviews)} products from {REVIEWS_FILE}")
        return product_reviews
    
    except Exception as e:
        logger.error(f"Error loading reviews: {e}")
        return {}

def extract_key_phrases(reviews: List[Dict], max_phrases: int = 10) -> List[str]:
    """Extract key phrases from reviews based on frequency and sentiment."""
    # Combine all review texts
    all_text = " ".join([r.get('text', '') for r in reviews])
    
    # Simple phrase extraction using regex patterns
    # Look for phrases like "very comfortable", "great fit", etc.
    patterns = [
        r"(very|really|super|extremely|incredibly) ([\w\s]+)",  # Very/really + adjective
        r"(great|good|excellent|perfect|terrible|bad|poor) ([\w\s]+)",  # Quality + noun
        r"(fits|looks|feels) ([\w\s]+)",  # Action + description
        r"(love|hate|like|dislike) ([\w\s]+)"  # Emotion + object
    ]
    
    phrases = []
    for pattern in patterns:
        matches = re.findall(pattern, all_text.lower())
        phrases.extend([f"{m[0]} {m[1]}" for m in matches])
    
    # Count phrase frequency
    phrase_counts = defaultdict(int)
    for phrase in phrases:
        # Clean up the phrase - limit to reasonable length
        cleaned = " ".join(phrase.split()[:3])
        if len(cleaned) > 4:  # Only consider substantial phrases
            phrase_counts[cleaned] += 1
    
    # Get the most common phrases
    top_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
    return [phrase for phrase, count in top_phrases[:max_phrases]]

def extract_usage_contexts(reviews: List[Dict], max_contexts: int = 5) -> List[str]:
    """Extract usage contexts from reviews (where/when people use the product)."""
    # Look for specific patterns indicating use cases
    patterns = [
        r"(use|used|using|wore|wear|wearing) (it|this|these) (for|at|in|during) ([\w\s]+)",
        r"(perfect|great|good|ideal) (for|at|in|during) ([\w\s]+)",
        r"(wore|used) (to|at|for) ([\w\s]+)"
    ]
    
    contexts = []
    for review in reviews:
        text = review.get('text', '').lower()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    # Extract the context part (last group)
                    context = match[-1].strip()
                    if 5 < len(context) < 30:  # Reasonable length
                        contexts.append(context)
    
    # Count frequency
    context_counts = defaultdict(int)
    for context in contexts:
        context_counts[context] += 1
    
    # Get the most common contexts
    top_contexts = sorted(context_counts.items(), key=lambda x: x[1], reverse=True)
    return [context for context, count in top_contexts[:max_contexts]]

def calculate_sentiment(reviews: List[Dict]) -> Tuple[float, Dict[str, float]]:
    """
    Calculate overall sentiment and aspect-specific sentiment from reviews.
    
    Returns:
        Tuple containing overall sentiment score (-1 to 1) and 
        dictionary of aspect-specific sentiment scores
    """
    if not reviews:
        return 0.0, {}
    
    # Calculate overall sentiment using weighted average of ratings
    total_weight = 0
    weighted_sum = 0
    
    for review in reviews:
        rating = review.get('rating', 0)
        # Helpful votes increase weight, verified purchases too
        weight = 1.0
        weight += min(review.get('helpful_vote', 0), 10) * 0.2  # Cap at 3x weight
        weight += 0.5 if review.get('verified_purchase', False) else 0
        
        weighted_sum += (rating - 3) * weight  # Center on 0 (-2 to +2)
        total_weight += weight
    
    # Normalize to -1 to 1 range
    overall_sentiment = (weighted_sum / total_weight) / 2 if total_weight > 0 else 0
    
    # Look for aspect-specific sentiment
    aspects = {
        'comfort': ['comfort', 'comfortable', 'uncomfortable', 'soft', 'hard'],
        'fit': ['fit', 'fits', 'size', 'large', 'small', 'tight', 'loose'],
        'quality': ['quality', 'durable', 'cheap', 'expensive', 'worth'],
        'style': ['style', 'stylish', 'look', 'looks', 'design', 'color', 'pattern'],
        'value': ['value', 'price', 'worth', 'expensive', 'cheap', 'cost']
    }
    
    aspect_scores = {}
    for aspect, keywords in aspects.items():
        aspect_sum = 0
        aspect_count = 0
        
        for review in reviews:
            text = review.get('text', '').lower()
            for keyword in keywords:
                if keyword in text:
                    sentiment_value = (review.get('rating', 0) - 3) / 2  # -1 to 1
                    aspect_sum += sentiment_value
                    aspect_count += 1
                    break  # Only count once per review per aspect
        
        if aspect_count > 0:
            aspect_scores[aspect] = aspect_sum / aspect_count
    
    return overall_sentiment, aspect_scores

def get_review_summary(product_id: str) -> Dict:
    """Get or generate a summary of reviews for a product."""
    global review_summaries
    
    # Return cached summary if available
    if product_id in review_summaries:
        return review_summaries[product_id]
    
    # Get reviews for this product
    reviews = product_reviews.get(product_id, [])
    if not reviews:
        return {}
    
    # Generate summary
    key_phrases = extract_key_phrases(reviews)
    usage_contexts = extract_usage_contexts(reviews)
    overall_sentiment, aspect_sentiments = calculate_sentiment(reviews)
    
    # Calculate rating distribution
    rating_distribution = defaultdict(int)
    for review in reviews:
        rating = int(review.get('rating', 0))
        if 1 <= rating <= 5:
            rating_distribution[rating] += 1
    
    # Select representative reviews (highly helpful or recent verified purchases)
    def review_score(review):
        return (review.get('helpful_vote', 0) * 10) + (5 if review.get('verified_purchase') else 0)
    
    sorted_reviews = sorted(reviews, key=review_score, reverse=True)
    representative_reviews = sorted_reviews[:3]
    
    # Create summary
    summary = {
        'review_count': len(reviews),
        'key_phrases': key_phrases,
        'usage_contexts': usage_contexts,
        'overall_sentiment': overall_sentiment,
        'aspect_sentiments': aspect_sentiments,
        'rating_distribution': dict(rating_distribution),
        'representative_reviews': [
            {
                'title': r.get('title', ''),
                'text': r.get('text', ''),
                'rating': r.get('rating', 0)
            }
            for r in representative_reviews
        ]
    }
    
    # Cache for future use
    review_summaries[product_id] = summary
    return summary

def enhance_product_embedding_text(product, product_id=None):
    """
    Enhance product text for embedding with review insights.
    This makes the embedding more useful for semantic search.
    """
    if not product_id:
        product_id = product.id
        
    base_text = f"Product: {product.title}\n"
    
    if product.description:
        base_text += f"Description: {product.description}\n"
    
    if product.categories:
        base_text += f"Categories: {', '.join(product.categories)}\n"
    
    if product.brand:
        base_text += f"Brand: {product.brand}\n"
    
    if product.features:
        base_text += f"Features: {', '.join(product.features)}\n"
    
    # Add review insights if available
    if product_id in product_reviews and product_reviews[product_id]:
        # Get review summary
        summary = get_review_summary(product_id)
        
        if summary.get('key_phrases'):
            base_text += f"Customer highlights: {', '.join(summary['key_phrases'])}\n"
        
        if summary.get('usage_contexts'):
            base_text += f"Usage contexts: {', '.join(summary['usage_contexts'])}\n"
        
        # Add sentiment insights
        if 'aspect_sentiments' in summary:
            sentiments = []
            for aspect, score in summary['aspect_sentiments'].items():
                sentiment = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
                sentiments.append(f"{aspect}: {sentiment}")
            
            if sentiments:
                base_text += f"Customer opinions: {', '.join(sentiments)}\n"
    
    return base_text

# Initialize when module is imported
load_reviews() 