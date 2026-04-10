#!/bin/bash

# Configuration
APP_NAME="moonvit-search"
DB_HOST="62.72.42.158"
DB_PORT="5433"
DB_USER="postgres"
DB_PASS="root"
DB_NAME="clip_search"

# Get absolute paths for volumes
# Assumes the script is run from inside streamlittest_moonvit
CUR_DIR=$(pwd)
PARENT_DIR=$(dirname "$CUR_DIR")
IMAGES_DIR_HOST="$PARENT_DIR/Images"
MODELS_DIR_HOST="$PARENT_DIR/models"

echo "Building Podman image $APP_NAME..."
podman build -t $APP_NAME .

echo "Stopping existing container if running..."
podman stop $APP_NAME 2>/dev/null
podman rm $APP_NAME 2>/dev/null

echo "Starting MoonViT Hybrid Search Container..."
podman run -d \
    --name $APP_NAME \
    -p 8501:8501 \
    -v "$IMAGES_DIR_HOST:/app/Images:ro" \
    -v "$MODELS_DIR_HOST:/app/models:ro" \
    -e DB_HOST="$DB_HOST" \
    -e DB_PORT="$DB_PORT" \
    -e DB_USER="$DB_USER" \
    -e DB_PASS="$DB_PASS" \
    -e DB_NAME="$DB_NAME" \
    -e IMAGES_DIR="/app/Images" \
    -e MODEL_DIR="/app/models/moonshotaiMoonViT-SO-400M" \
    --restart always \
    $APP_NAME

echo "Deployment complete. Application available at http://your-server-ip:8501"
echo "Check logs with: podman logs -f $APP_NAME"
