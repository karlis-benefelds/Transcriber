#!/bin/bash

# =================================================================
# Google Cloud Run Deployment Script for Transcriber Application
# =================================================================
# This script automates the deployment process to Google Cloud Run
# Make sure you have gcloud CLI installed and authenticated

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values (can be overridden with environment variables)
PROJECT_ID=${GOOGLE_CLOUD_PROJECT_ID:-""}
SERVICE_NAME=${CLOUD_RUN_SERVICE_NAME:-"transcriber-app"}
REGION=${CLOUD_RUN_REGION:-"us-central1"}
GCR_HOSTNAME=${GCR_HOSTNAME:-"gcr.io"}

print_step() {
    echo -e "${BLUE}==== $1 ====${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_step "Checking Prerequisites"
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI is not installed. Please install it from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker."
        exit 1
    fi
    
    # Check if user is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null; then
        print_error "Not authenticated with gcloud. Run: gcloud auth login"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to get project ID
get_project_id() {
    if [ -z "$PROJECT_ID" ]; then
        # Try to get from gcloud config
        PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
        
        if [ -z "$PROJECT_ID" ]; then
            print_error "PROJECT_ID not set. Please set GOOGLE_CLOUD_PROJECT_ID environment variable or run:"
            echo "gcloud config set project YOUR_PROJECT_ID"
            exit 1
        fi
    fi
    
    print_success "Using project: $PROJECT_ID"
}

# Function to enable required APIs
enable_apis() {
    print_step "Enabling Required Google Cloud APIs"
    
    gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID
    gcloud services enable run.googleapis.com --project=$PROJECT_ID
    gcloud services enable secretmanager.googleapis.com --project=$PROJECT_ID
    
    print_success "APIs enabled"
}

# Function to create secrets
create_secrets() {
    print_step "Setting up Secret Manager (if not exists)"
    
    # Check if .env file exists
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating secrets with placeholder values."
        print_warning "You'll need to update these secrets with real values after deployment."
    fi
    
    # Create secret for the transcriber secrets
    if ! gcloud secrets describe transcriber-secrets --project=$PROJECT_ID > /dev/null 2>&1; then
        gcloud secrets create transcriber-secrets --project=$PROJECT_ID
        print_success "Created transcriber-secrets in Secret Manager"
    else
        print_success "transcriber-secrets already exists"
    fi
    
    print_warning "Remember to update secrets with actual values:"
    echo "gcloud secrets versions add transcriber-secrets --data-file=secrets.json --project=$PROJECT_ID"
}

# Function to build and push Docker image
build_and_push() {
    print_step "Building and Pushing Docker Image"
    
    IMAGE_URL="$GCR_HOSTNAME/$PROJECT_ID/$SERVICE_NAME:latest"
    
    # Build the image
    print_step "Building Docker image..."
    docker build -t $IMAGE_URL .
    
    # Configure Docker to use gcloud as credential helper
    gcloud auth configure-docker $GCR_HOSTNAME --quiet
    
    # Push the image
    print_step "Pushing Docker image to $IMAGE_URL..."
    docker push $IMAGE_URL
    
    print_success "Image built and pushed: $IMAGE_URL"
}

# Function to deploy to Cloud Run
deploy_to_cloud_run() {
    print_step "Deploying to Cloud Run"
    
    IMAGE_URL="$GCR_HOSTNAME/$PROJECT_ID/$SERVICE_NAME:latest"
    
    gcloud run deploy $SERVICE_NAME \
        --image=$IMAGE_URL \
        --platform=managed \
        --region=$REGION \
        --allow-unauthenticated \
        --memory=8Gi \
        --cpu=4 \
        --timeout=3600 \
        --max-instances=10 \
        --concurrency=10 \
        --port=8080 \
        --set-env-vars="FLASK_ENV=production,DEV_MODE=false,PORT=8080" \
        --update-secrets="SECRET_KEY=transcriber-secrets:latest:secret-key" \
        --update-secrets="GOOGLE_CLIENT_ID=transcriber-secrets:latest:google-client-id" \
        --update-secrets="GOOGLE_CLIENT_SECRET=transcriber-secrets:latest:google-client-secret" \
        --update-secrets="OPENAI_API_KEY=transcriber-secrets:latest:openai-api-key" \
        --project=$PROJECT_ID
    
    print_success "Deployment completed!"
}

# Function to get service URL
get_service_url() {
    print_step "Getting Service URL"
    
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
        --region=$REGION \
        --project=$PROJECT_ID \
        --format="value(status.url)")
    
    print_success "Service deployed at: $SERVICE_URL"
    echo ""
    print_warning "Next steps:"
    echo "1. Update Google OAuth redirect URIs to include: $SERVICE_URL/callback"
    echo "2. Update Secret Manager with your actual API keys"
    echo "3. Test the application health check: $SERVICE_URL/health"
    echo ""
}

# Main execution
main() {
    print_step "Starting Google Cloud Run Deployment"
    echo "Service: $SERVICE_NAME"
    echo "Region: $REGION"
    echo ""
    
    check_prerequisites
    get_project_id
    enable_apis
    create_secrets
    build_and_push
    deploy_to_cloud_run
    get_service_url
    
    print_success "Deployment script completed successfully! 🚀"
}

# Help function
show_help() {
    echo "Google Cloud Run Deployment Script for Transcriber Application"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Environment Variables:"
    echo "  GOOGLE_CLOUD_PROJECT_ID    Google Cloud Project ID"
    echo "  CLOUD_RUN_SERVICE_NAME     Cloud Run service name (default: transcriber-app)"
    echo "  CLOUD_RUN_REGION           Deployment region (default: us-central1)"
    echo "  GCR_HOSTNAME               Container registry hostname (default: gcr.io)"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  GOOGLE_CLOUD_PROJECT_ID=my-project ./deploy-cloud-run.sh"
    echo "  CLOUD_RUN_REGION=europe-west1 ./deploy-cloud-run.sh"
    echo ""
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac