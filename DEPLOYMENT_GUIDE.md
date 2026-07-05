# MarineVision Deployment Guide

## Table of Contents
1. [Local Development](#local-development)
2. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Environment Setup](#environment-setup)
5. [Troubleshooting](#troubleshooting)

## Local Development

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git (optional, for version control)
- ~2GB free disk space
- 4GB+ RAM recommended

### Step-by-Step Setup

#### 1. Navigate to Project Directory
```bash
cd "MarineVision Intelligent Underwater Detection"
```

#### 2. Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Expected installation time: 5-15 minutes depending on internet speed.

#### 4. Verify Model File
Ensure `models/MarineVision_YOLOv11l_best.pt` exists:
```bash
# On Windows
dir models\

# On macOS/Linux
ls -la models/
```

#### 5. Run the Application
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

#### 6. Test the Application
- Navigate through all pages
- Upload a test image
- Verify model loads without errors
- Check detection functionality

### Development Workflow

#### Making Changes
1. Edit Python files as needed
2. Streamlit automatically reloads on save
3. Use browser's refresh if changes don't appear

#### Debugging
```bash
# Run with verbose logging
streamlit run app.py --logger.level=debug

# Run in development mode
streamlit run app.py --client.showErrorDetails=true
```

#### Performance Testing
```bash
# Measure load time
time streamlit run app.py --client.toolbarMode=minimal
```

## Streamlit Cloud Deployment

### Prerequisites
- GitHub account
- GitHub repository with MarineVision code
- Streamlit Cloud account (free tier available)

### Deployment Steps

#### 1. Prepare GitHub Repository

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#00E5FF"
backgroundColor = "#03131f"
secondaryBackgroundColor = "#06263c"
textColor = "#e6f7ff"
font = "sans serif"

[client]
toolbarMode = "minimal"
showErrorDetails = false
showSidebarNavigation = true

[server]
maxUploadSize = 200
enableXsrfProtection = true
```

Create `.gitignore`:
```
venv/
__pycache__/
*.pyc
.DS_Store
.streamlit/secrets.toml
```

#### 2. Push to GitHub
```bash
git init
git add .
git commit -m "Initial MarineVision deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/MarineVision.git
git push -u origin main
```

#### 3. Deploy on Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Click "New app"
3. Select:
   - Repository: `YOUR_USERNAME/MarineVision`
   - Branch: `main`
   - Main file path: `app.py`
4. Click "Deploy"

#### 4. Configure App Settings
- App name: MarineVision
- URL slug: marinev-underwater-detection
- Sharing: Public or Private

#### 5. Monitor Deployment
- Check deployment logs
- Verify app loads successfully
- Test all pages and features

### Custom Domain (Optional)
1. Go to app settings
2. Click "Custom domain"
3. Add your domain (e.g., `marinev.app`)
4. Follow DNS configuration instructions

## Docker Deployment

### Prerequisites
- Docker installed
- Docker Hub account (for pushing images)
- ~3GB disk space for Docker image

### Build Docker Image

Create `Dockerfile` in project root:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run Locally

```bash
# Build image
docker build -t marinev:latest .

# Run container
docker run -p 8501:8501 \
  -e STREAMLIT_SERVER_PORT=8501 \
  -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
  marinev:latest

# Access at http://localhost:8501
```

### Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag image
docker tag marinev:latest your_username/marinev:latest

# Push to registry
docker push your_username/marinev:latest

# Pull and run from Hub
docker run -p 8501:8501 your_username/marinev:latest
```

### Kubernetes Deployment (Advanced)

Create `k8s-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: marinev
spec:
  replicas: 2
  selector:
    matchLabels:
      app: marinev
  template:
    metadata:
      labels:
        app: marinev
    spec:
      containers:
      - name: marinev
        image: your_username/marinev:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

Deploy:
```bash
kubectl apply -f k8s-deployment.yaml
kubectl expose deployment marinev --type=LoadBalancer --port=8501
```

## Environment Setup

### System Requirements

**Minimum:**
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 4GB
- Storage: 200MB + model file (145MB)
- Internet: Broadband connection

**Recommended:**
- CPU: Intel i7 / AMD Ryzen 7
- RAM: 8GB+
- Storage: SSD with 500MB free
- GPU: NVIDIA GPU with CUDA support
- Internet: High-speed connection

### Python Environment Variables

Create `.env` file:
```env
# Model configuration
MODEL_PATH=models/MarineVision_YOLOv11l_best.pt

# Inference settings
INFERENCE_DEVICE=auto
INFERENCE_DTYPE=float32

# Streamlit configuration
STREAMLIT_LOGGER_LEVEL=info
STREAMLIT_CLIENT_TOOLBAR_MODE=minimal
STREAMLIT_SERVER_MAXUPLOADSIZE=200

# PyTorch configuration
PYTORCH_ENABLE_MPS_FALLBACK=1
OMP_NUM_THREADS=4
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Troubleshooting

### Model Loading Issues

**Problem**: `Model file not found`
```
Error: Could not find the trained model file at models/MarineVision_YOLOv11l_best.pt
```

**Solutions**:
1. Verify file exists: `ls -la models/`
2. Check file size: Should be ~145MB
3. Verify permissions: File should be readable
4. Reinstall model if corrupted

### Memory Issues

**Problem**: `CUDA out of memory`
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Use CPU inference: Set `INFERENCE_DEVICE=cpu`
2. Reduce batch size
3. Compress input images before upload
4. Restart application to free memory

### Dependency Conflicts

**Problem**: `ImportError: No module named 'ultralytics'`

**Solutions**:
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Reinstall requirements
pip install --force-reinstall -r requirements.txt

# Check for version conflicts
pip check
```

### Slow Inference

**Symptom**: Detection takes >10 seconds

**Solutions**:
1. Use GPU acceleration (install CUDA)
2. Reduce image size before upload
3. Convert model to ONNX for faster inference
4. Enable model quantization
5. Check CPU/GPU utilization with `nvidia-smi`

### CSS Not Loading

**Problem**: Styling missing, dark theme not applied

**Solutions**:
1. Verify `assets/styles.css` exists
2. Clear browser cache: `Ctrl+Shift+Delete`
3. Hard refresh page: `Ctrl+Shift+R`
4. Check file permissions on CSS
5. Restart Streamlit server

### Deployment Failures

**Streamlit Cloud Errors**:
```bash
# Check logs
streamlit logs

# View deployment logs in Streamlit Cloud UI
# Settings > Logs

# Common issues:
# - Missing model file in repository
# - Incompatible Python version
# - Timeout during deployment (>25 minutes)
# - Memory exceeded on free tier
```

**Solutions**:
1. Use `.gitignore` to exclude large files
2. Pin Python version: Create `runtime.txt` with `python-3.10`
3. Pre-download model in setup script
4. Upgrade to paid tier for more resources
5. Use `@st.cache_resource` for efficient loading

### Performance Optimization

**Caching Strategy**:
```python
@st.cache_resource
def load_model():
    return load_detection_model()

@st.cache_data
def process_image(image_bytes):
    return Image.open(BytesIO(image_bytes))
```

**Monitoring**:
```bash
# Check memory usage
ps aux | grep streamlit

# Monitor GPU usage
nvidia-smi

# Check disk space
du -sh .

# Profile code
python -m cProfile -s cumulative app.py
```

## Production Checklist

- [ ] Model file verified and size correct
- [ ] All requirements pinned to specific versions
- [ ] `.gitignore` configured properly
- [ ] Environment variables set up
- [ ] CSS and assets included
- [ ] All pages tested and working
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Security checks passed
- [ ] Performance benchmarked
- [ ] Documentation updated
- [ ] README includes setup instructions
- [ ] License file included
- [ ] GitHub repository properly configured
- [ ] Streamlit Cloud deployment working
- [ ] Custom domain configured (if applicable)
- [ ] Monitoring and alerts set up
- [ ] Backup strategy in place

## Support & Resources

- **Documentation**: [Streamlit Docs](https://docs.streamlit.io)
- **YOLO**: [Ultralytics YOLOv11](https://docs.ultralytics.com)
- **GitHub**: Push to repository for version control
- **Issues**: Open GitHub issues for bug reports
- **Discussions**: Community help on GitHub Discussions

## Next Steps

1. ✅ Local testing complete
2. 🚀 Deploy to Streamlit Cloud
3. 📊 Monitor performance
4. 🔄 Iterate based on feedback
5. 🌟 Share with community

---

**Last Updated**: 2024
**Status**: Production Ready
**Version**: 1.0
