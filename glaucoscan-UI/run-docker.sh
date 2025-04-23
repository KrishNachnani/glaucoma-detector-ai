#!/bin/bash

# Define variables
IMAGE_NAME="glaucoma-nextjs"
CONTAINER_NAME="glaucoma-app"
PORT=8080

# Default API URL if not specified as an environment variable
API_URL=${API_URL:-"http://localhost:8999"}

# Default email settings if not specified as environment variables
MAILERSEND_API_KEY=${MAILERSEND_API_KEY:-"mlsn.f246cd6923b2043f9ae4730ca7c15f70e1fbd3017d8d8c5fb5f4cacafcb8c870"}
EMAIL_FROM=${EMAIL_FROM:-"noreply@test-zkq340eep5xgd796.mlsender.net"}
EMAIL_TO=${EMAIL_TO:-"nachnanikrish@gmail.com"}

# Stop and remove existing container if it exists
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

echo "🔨 Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .

echo "🚀 Starting container: $CONTAINER_NAME"

# Run the new container with environment variables
docker run \
  -e NEXT_PUBLIC_GLAUCOMA_API_URL=$API_URL \
  -e NEXT_PUBLIC_MAILERSEND_API_KEY=$MAILERSEND_API_KEY \
  -e NEXT_PUBLIC_EMAIL_FROM=$EMAIL_FROM \
  -e NEXT_PUBLIC_EMAIL_TO=$EMAIL_TO \
  --name $CONTAINER_NAME \
  -p $PORT:8001 \
  $IMAGE_NAME

echo "✅ Container is running!"
echo "🌐 Access the application at: http://localhost:$PORT"
echo "🔌 API URL is set to: $API_URL"