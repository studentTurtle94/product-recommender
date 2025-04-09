import os
import json
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import List, Optional

from .products import load_products, get_product_by_id, get_all_products, ProductItem
from .embeddings import init_embedding_client, generate_product_embeddings, delete_all_vectors
from .recommender import init_recommender, semantic_search, refine_recommendations, multimodal_search
from .reviews import get_review_summary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Fashion Recommender API")

# Add CORS middleware to allow cross-origin requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize components on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Fashion Recommender API...")
    # Load products from JSON file
    load_products()
    # Initialize OpenAI client for embeddings
    init_embedding_client()
    # Initialize recommender
    init_recommender()
    # Load reviews data (this is already done when reviews module is imported)
    logger.info("Reviews data loaded for enhanced recommendations")
    logger.info("Initialization complete.")

# Define API endpoints
@app.get("/")
def read_root():
    return {"message": "Fashion Recommender API is running"}

@app.get("/products/{product_id}", response_model=ProductItem)
def get_product(product_id: str):
    product = get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@app.get("/products", response_model=list[ProductItem])
def get_products(limit: int = Query(10, ge=1, le=100)):
    products = get_all_products()
    return products[:limit]

@app.get("/reviews/{product_id}")
def get_reviews(product_id: str):
    """Get review summary for a product."""
    summary = get_review_summary(product_id)
    if not summary:
        raise HTTPException(status_code=404, detail="No reviews found for this product")
    return summary

@app.post("/embeddings/generate")
async def generate_embeddings(background_tasks: BackgroundTasks):
    """Generate embeddings for all products in the background."""
    background_tasks.add_task(generate_product_embeddings, get_all_products())
    return {"message": "Embedding generation started in the background"}

@app.post("/embeddings/delete")
async def delete_embeddings(background_tasks: BackgroundTasks):
    """Delete all embeddings from the vector store in the background."""
    background_tasks.add_task(delete_all_vectors)
    return {"message": "Embedding deletion started in the background"}

@app.get("/search")
async def search(query: str, limit: int = Query(5, ge=1, le=20)):
    """Search for products using semantic search."""
    products = await semantic_search(query, top_k=limit)
    if not products:
        return {"products": [], "message": "No products found matching your query"}
    return {"products": products}

@app.get("/recommend")
async def recommend(query: str, limit: int = Query(5, ge=1, le=10)):
    """Get refined recommendations based on a query."""
    # First, perform semantic search
    products = await semantic_search(query, top_k=limit)
    if not products:
        return {"products": [], "alternative_searches": [], "message": "No products found matching your query"}
    
    # Then, refine recommendations with LLM
    result = await refine_recommendations(query, products)
    return result

@app.post("/multimodal-search")
async def search_multimodal(
    query_text: str = Form(...),
    image_file: Optional[UploadFile] = File(None),
    limit: int = Form(5)
):
    """Search for products using multimodal input (text + optional image)."""
    if not image_file:
        # Fallback to regular semantic search if no image is provided
        logger.info("No image provided, falling back to text-based semantic search.")
        products = await semantic_search(query=query_text, top_k=limit)
        if not products:
            return {"products": [], "message": "No products found matching your query"}
        return {"products": products}

    # Read image data
    image_data = await image_file.read()
    logger.info(f"Received image: {image_file.filename}, size: {len(image_data)} bytes, content-type: {image_file.content_type}")

    # Perform multimodal search using the recommender function
    products = await multimodal_search(query_text=query_text, image_data=image_data, top_k=limit)

    if not products:
        return {"products": [], "message": "No products found matching your multimodal query"}
    
    # TODO: Potentially add refinement step here as well if needed, similar to /recommend endpoint

    return {"products": products}
