# Google Cloud Run Deployment Guide

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud SDK** installed and authenticated
3. **Docker** installed on your local machine
4. **Project Setup** in Google Cloud Console

## Step-by-Step Deployment

### 1. Initial Setup

```bash
# Install Google Cloud SDK (if not already installed)
# Visit: https://cloud.google.com/sdk/docs/install

# Authenticate with Google Cloud
gcloud auth login

# Set your project ID
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

### 2. Configure Secrets

Create your secrets file based on the template:

```bash
# Copy the template
cp secrets-template.json secrets.json

# Edit secrets.json with your actual values
nano secrets.json
```

Required secrets:
- **secret-key**: Generate with `python -c "import secrets; print(secrets.token_hex(32))"`
- **google-client-id**: From Google OAuth Console
- **google-client-secret**: From Google OAuth Console  
- **openai-api-key**: From OpenAI Platform

### 3. Set Up Google OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to "APIs & Services" > "Credentials"
3. Create OAuth 2.0 Client ID for Web application
4. Add authorized redirect URIs:
   - `http://localhost:8080/callback` (for local testing)
   - `https://YOUR_CLOUD_RUN_URL/callback` (add after deployment)

### 4. Create Secret Manager Secrets

```bash
# Create the secrets in Google Secret Manager
gcloud secrets create transcriber-secrets --data-file=secrets.json

# Verify the secret was created
gcloud secrets list
```

### 5. Deploy Using Script

Use the automated deployment script:

```bash
# Make sure script is executable
chmod +x deploy-cloud-run.sh

# Set environment variables
export GOOGLE_CLOUD_PROJECT_ID=your-project-id
export CLOUD_RUN_SERVICE_NAME=transcriber-app
export CLOUD_RUN_REGION=us-central1

# Run deployment
./deploy-cloud-run.sh
```

### 6. Manual Deployment (Alternative)

If you prefer manual deployment:

```bash
# Build and tag the image
docker build -t gcr.io/YOUR_PROJECT_ID/transcriber-app:latest .

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/transcriber-app:latest

# Deploy to Cloud Run
gcloud run deploy transcriber-app \
  --image=gcr.io/YOUR_PROJECT_ID/transcriber-app:latest \
  --platform=managed \
  --region=us-central1 \
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
  --update-secrets="OPENAI_API_KEY=transcriber-secrets:latest:openai-api-key"
```

## Post-Deployment Configuration

### 1. Update OAuth Redirect URIs

After deployment, get your service URL:

```bash
gcloud run services list
```

Then update your Google OAuth configuration to include:
`https://YOUR_CLOUD_RUN_URL/callback`

### 2. Test the Deployment

```bash
# Check health endpoint
curl https://YOUR_CLOUD_RUN_URL/health

# Check application
open https://YOUR_CLOUD_RUN_URL
```

## Configuration Options

### Resource Allocation

For different workloads, adjust these settings:

```bash
# Light usage (fewer concurrent users)
--memory=4Gi --cpu=2 --max-instances=5

# Heavy usage (many concurrent transcriptions)  
--memory=16Gi --cpu=8 --max-instances=20
```

### Regions

Choose the region closest to your users:
- `us-central1` - Iowa, USA
- `us-east1` - South Carolina, USA
- `europe-west1` - Belgium
- `asia-northeast1` - Tokyo, Japan

### Custom Domains

To use a custom domain:

```bash
# Map domain to service
gcloud run domain-mappings create --service=transcriber-app --domain=transcriber.yourdomain.com --region=us-central1
```

## Monitoring and Logs

### View Logs

```bash
# Stream logs
gcloud run logs tail transcriber-app --region=us-central1

# View recent logs
gcloud run logs read transcriber-app --region=us-central1 --limit=50
```

### Monitoring

Access Cloud Run metrics in the Google Cloud Console:
1. Go to Cloud Run > Services > transcriber-app
2. Click "Metrics" tab
3. Monitor requests, latency, and errors

## Troubleshooting

### Common Issues

1. **Container startup timeout**
   - Increase `--timeout` value
   - Check application startup time in logs

2. **Memory exceeded**
   - Increase `--memory` allocation
   - Monitor memory usage patterns

3. **Authentication errors**
   - Verify Secret Manager permissions
   - Check OAuth redirect URI configuration

4. **File upload issues**
   - Ensure Cloud Run timeout is sufficient (3600s)
   - Check file size limits

### Debug Commands

```bash
# Check service details
gcloud run services describe transcriber-app --region=us-central1

# Check secret values (be careful - will show actual secrets)
gcloud secrets versions access latest --secret=transcriber-secrets

# Test container locally
docker run -p 8080:8080 -e PORT=8080 gcr.io/YOUR_PROJECT_ID/transcriber-app:latest
```

## Security Best Practices

1. **Secrets Management**
   - Never commit secrets to version control
   - Use Secret Manager for all sensitive data
   - Rotate secrets regularly

2. **Network Security**
   - Consider using Cloud Run with VPC
   - Implement proper CORS policies
   - Use HTTPS only

3. **Access Control**
   - Limit OAuth to Minerva domains only
   - Implement proper session management
   - Monitor access logs

## Cost Optimization

1. **Set appropriate limits**
   ```bash
   --max-instances=10  # Prevent runaway costs
   --concurrency=10    # Handle multiple requests per instance
   ```

2. **Monitor usage**
   - Set up billing alerts
   - Review Cloud Run pricing
   - Consider regional pricing differences

3. **Optimize for cold starts**
   - Use minimum instances for critical applications
   - Optimize Docker image size
   - Use Cloud Run gen2 execution environment

## Scaling Considerations

- **CPU-only deployment**: Current configuration suitable for moderate usage
- **GPU acceleration**: Consider migrating to Compute Engine with GPUs for heavy workloads
- **Multi-region**: Deploy to multiple regions for global availability

## Support

For deployment issues:
1. Check the [Cloud Run documentation](https://cloud.google.com/run/docs)
2. Review application logs in Cloud Console
3. Open GitHub issue for application-specific problems