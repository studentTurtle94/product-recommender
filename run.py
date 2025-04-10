import os
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Start the FastAPI server with uvicorn, configured for Render."""
    # Get Render's PORT environment variable, default to 10000 if not set
    port = int(os.environ.get("PORT", 10000))
    # Bind to 0.0.0.0 as required by Render
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    # Start the uvicorn server - disabled reload for production
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        # reload=False # Explicitly False or remove line
    )

if __name__ == "__main__":
    main() 