
# Script to build and run the Glaucoma Detection API Docker container
API_PORT=8999


# Stop and remove existing container if it exists
if [ "$(docker ps -aq -f name=glaucoma-api)" ]; then
    echo "Stopping and removing existing container..."
    docker stop glaucoma-api
    docker rm glaucoma-api
fi

# Build the Docker image
echo "Building Docker image..."
docker build -t glaucoma-api:latest .

# Create directories for persistent storage if they don't exist
mkdir -p $(pwd)/data/logs
mkdir -p $(pwd)/data/uploaded

# Run the container
echo "Starting Glaucoma API container..."

docker run  \
    --name glaucoma-api \
    -p ${API_PORT}:8236 \
    -v $(pwd)/data/logs:/app/logs \
    -v $(pwd)/data/uploaded:/app/uploaded \
    --restart unless-stopped \
    glaucoma-api:latest

# Check if container started successfully
if [ "$(docker ps -q -f name=glaucoma-api)" ]; then
    echo "Glaucoma API is running!"
    echo "API is accessible at http://localhost:${API_PORT}"
    echo "API documentation is available at http://localhost:${API_PORT}/docs"
else
    echo "Failed to start container. Check Docker logs for details."
fi