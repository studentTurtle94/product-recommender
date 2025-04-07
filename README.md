# Fashion Recommender

A semantic fashion recommendation system that uses natural language processing to find and recommend fashion products based on user queries. This system leverages OpenAI's embedding and vector store to provide accurate semantic search, and enhances recommendations using customer review data.

## Features

- Natural language search for fashion products
- Semantic matching using OpenAI embeddings and vector store
- LLM-powered recommendation refinement
- Review-enhanced product descriptions for better search relevance
- Sentiment analysis from customer reviews
- React frontend with intuitive UI

## Project Structure

```
fashion-recommender/
├── app/                    # Backend application
│   ├── __init__.py         # Python package initialization
│   ├── main.py             # FastAPI application entry point
│   ├── products.py         # Product data management
│   ├── embeddings.py       # Embedding generation and vector storage
│   ├── recommender.py      # Recommendation logic
│   └── reviews.py          # Review processing and enhancement
├── client/                 # React frontend
│   ├── public/             # Static files
│   └── src/                # React source code
├── data/                   # Data processing and storage
│   ├── process_data.py     # Data preprocessing script
│   └── fashion_products.json       # Processed product data
├── .env                    # Environment variables (create from .env.example)
├── .env.example            # Environment variables template
├── requirements.txt        # Python dependencies
├── run.py                  # Server startup script
├── reset_vectors.py        # Utility to reset vector store
└── start.sh                # Bash script to start the server
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- Node.js 14+
- OpenAI API key with vector store access
- Amazon Fashion product data (.jsonl format)
- Amazon Fashion review data (.jsonl format)

### Environment Setup

1. Clone the repository
2. Create a `.env` file based on `.env.example`:

```
OPENAI_API_KEY=your_openai_api_key_here
SERVER_PORT=8000
SERVER_HOST=0.0.0.0
```

### Backend Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Process the Amazon Fashion dataset:
```bash
python data/process_data.py
```

3. Process the reviews dataset:
```bash
python data/process_reviews.py --input ~/Desktop/Amazon_Fashion.jsonl --output data/reviews.json
```
   - By default, only reviews for products in your sample are processed
   - Use `--limit-per-product 50` to limit reviews per product (default: 50)
   - Use `--total-limit 1000000` to limit total reviews processed
   - Use `--no-filter` to process all reviews regardless of product sample

4. Start the backend server:
```bash
./start.sh
```

5. Generate embeddings (after server has started):
   - Make a POST request to `http://localhost:8000/embeddings/generate`
   - Or use curl: `curl -X POST http://localhost:8000/embeddings/generate`

### Frontend Setup

1. Navigate to the client directory:
```bash
cd client
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The application will be available at `http://localhost:3000`.

## Using the System

### Sample Queries

Try searching for products with natural language queries like:

- "A comfortable summer dress for a beach vacation"
- "Professional looking men's shoes for a job interview"
- "Waterproof jacket for hiking in rainy weather"
- "Stylish sunglasses that provide good UV protection"
- "Breathable athletic wear for hot yoga classes"

### Review-Enhanced Search

The system enhances product embeddings with insights from customer reviews:

- Key phrases extracted from reviews
- Usage contexts mentioned by customers
- Sentiment analysis across different product aspects (comfort, fit, quality, etc.)
- Representative review snippets

This provides a more accurate search experience that factors in real user experiences with the products.

## Reset Vector Store

If you need to reset the vector store and regenerate embeddings:

```bash
python reset_vectors.py
```

## API Endpoints

- `GET /` - API status check
- `GET /products/{product_id}` - Get a specific product
- `GET /products?limit=10` - Get a list of products
- `GET /reviews/{product_id}` - Get review summary for a product
- `GET /search?query=your_query&limit=5` - Search for products
- `GET /recommend?query=your_query&limit=5` - Get refined recommendations
- `POST /embeddings/generate` - Generate embeddings for all products
- `POST /embeddings/delete` - Delete all embeddings from vector store

## Technology Stack

- **Backend**: FastAPI, Python, OpenAI API
- **Frontend**: React, JavaScript, CSS
- **Data Processing**: Pandas, NumPy
- **Embedding Storage**: OpenAI Vector Store
- **Text Processing**: Regular expressions, custom NLP techniques

## License

This project is for educational purposes only. Product data is sourced from Amazon and should not be used for commercial purposes.
