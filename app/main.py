import os
import json
import logging
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

if os.getenv("SERVE_STATIC_FILES", "false").lower() == "true":
    build_output_dir = "/app/app/static" 

    static_assets_path = os.path.join(build_output_dir, "static")

    index_html_path = os.path.join(build_output_dir, "index.html")

    if os.path.isdir(static_assets_path):
         app.mount("/static", StaticFiles(directory=static_assets_path), name="static")
    else:
         print(f"Warning: Static assets directory not found at {static_assets_path}")

    # Define the catch-all route to serve index.html for SPA routing
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_react_app(full_path: str):
        if os.path.exists(index_html_path):
            return FileResponse(index_html_path)
        else:
            print(f"Error: index.html not found at {index_html_path}")
            return {"message": "Frontend entry point (index.html) not found"}, 404


# Initialize components on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Fashion Recommender API...")
    load_products()
    init_embedding_client()
    init_recommender()
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

    return {"products": products}

@app.post("/recommend-multimodal")
async def recommend_multimodal(
    query_text: str = Form(...),
    image_file: Optional[UploadFile] = File(None),
    limit: int = Form(5)
):
    """Get refined recommendations using multimodal input (text + optional image)."""
    # Initial search step (multimodal or text-based fallback)
    if image_file:
        # Read image data
        image_data = await image_file.read()
        logger.info(f"Received image for multimodal recommendation: {image_file.filename}, size: {len(image_data)} bytes")
        # Perform multimodal search
        initial_products = await multimodal_search(query_text=query_text, image_data=image_data, top_k=limit)
    else:
        # Fallback to text-based semantic search if no image provided
        logger.info("No image provided for recommendation, falling back to text-based semantic search.")
        initial_products = await semantic_search(query=query_text, top_k=limit)

    if not initial_products:
        return {"products": [], "alternative_searches": [], "message": "No products found matching your initial search"}

    # Refine recommendations using the initial results and the query text
    logger.info(f"Refining {len(initial_products)} initial results for query: '{query_text}'")
    refined_result = await refine_recommendations(query=query_text, products=initial_products)

    return refined_result
