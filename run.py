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
    """Start the FastAPI server with uvicorn."""
    # Get configuration from environment variables, with defaults
    port = int(os.environ.get("SERVER_PORT", 8000))
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    
    # Start the uvicorn server
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True  # Enable auto-reload during development
    )

if __name__ == "__main__":
    main() 