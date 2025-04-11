# Stage 1: Build the React frontend
FROM node:18 AS frontend-builder
WORKDIR /app/client
COPY client/package.json client/package-lock.json* ./
RUN npm install
COPY client/ ./
RUN npm run build

# Stage 2: Build the FastAPI backend
FROM python:3.10-slim AS backend
WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies if needed (e.g., for libraries like Pillow)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code
COPY app /app/app
COPY data /app/data
COPY run.py .
COPY reset_vectors.py .
COPY start.sh .

# Copy the built frontend static files from the builder stage
COPY --from=frontend-builder /app/client/build /app/app/static

# Make start script executable
RUN chmod +x ./start.sh

# Expose the port the app runs on (from .env or default)
EXPOSE 8000

# Command to run the application using the start script
# Ensure start.sh uses $SERVER_HOST and $SERVER_PORT from .env or defaults
# Make sure uvicorn binds to 0.0.0.0 inside the container
CMD ["./start.sh"]