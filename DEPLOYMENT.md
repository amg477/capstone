# Streamlit App Deployment Guide

## Option 1: Streamlit Cloud (Recommended - Easiest)

### Steps:

1. **Prepare your repository:**
   - Make sure your code is pushed to GitHub (or GitLab/Bitbucket)
   - Ensure `requirements.txt` is in the root directory
   - Make sure data files are in `data_storage/final_data/` directory

2. **Sign up for Streamlit Cloud:**
   - Go to https://share.streamlit.io/
   - Sign in with your GitHub account

3. **Deploy your app:**
   - Click "New app"
   - Select your repository
   - Main file path: `streamlit_app/app.py`
   - Click "Deploy"

4. **Configure (if needed):**
   - The app will automatically detect `requirements.txt`
   - If you have a Python version requirement, create `runtime.txt` with: `python-3.11`

### Important Notes:
- Streamlit Cloud has a 1GB file size limit per app
- If your data files are too large, you may need to:
  1. Use Git LFS for large files
  2. Host data files elsewhere (S3, Google Cloud Storage) and load them via URL
  3. Use a different deployment method

## Option 2: Heroku

### Steps:

1. **Create Procfile:**
   ```
   web: streamlit run streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create setup.sh:**
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

3. **Deploy:**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## Option 3: AWS/Docker

### Create Dockerfile:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Option 4: Local Deployment with ngrok (for testing)

```bash
# Install ngrok
brew install ngrok  # or download from ngrok.com

# Run your Streamlit app
streamlit run streamlit_app/app.py

# In another terminal, expose it
ngrok http 8501
```

## Data Files Considerations:

Since your data files are large, consider:
1. **Git LFS** for version control (if using Git)
2. **External storage** (S3, GCS) and load via URL
3. **Optimize data files** - convert CSVs to Parquet for faster loading

