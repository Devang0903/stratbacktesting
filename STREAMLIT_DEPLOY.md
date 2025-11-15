# Streamlit Cloud Deployment Guide

## Important Note: GitHub Enterprise

You're using **GitHub Enterprise** (`github.gatech.edu`), which may not be directly supported by Streamlit Community Cloud (which typically connects to `github.com`).

## Deployment Options

### Option 1: Streamlit Community Cloud (Try First)

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io
   - Sign in with your GitHub account

2. **Check Repository Access**
   - Streamlit Cloud may not see GitHub Enterprise repositories
   - If you see your `tradingstrats` repo, proceed to step 3
   - If not, see Option 2 or 3 below

3. **Deploy Your App**
   - Click "New app"
   - Select repository: `dajmera6/tradingstrats`
   - **Main file path**: `app.py`
   - **Python version**: 3.10 (or latest available)
   - Click "Deploy"

4. **Wait for Deployment**
   - First deployment takes 5-10 minutes
   - App will be live at: `https://YOUR_APP_NAME.streamlit.app`

### Option 2: Streamlit for Teams (If Available)

If your organization (Georgia Tech) has Streamlit for Teams:
1. Contact your IT department about Streamlit deployment
2. They can set up internal deployment

### Option 3: Manual Deployment (Alternative)

If Streamlit Cloud doesn't work, you can deploy manually:

#### A. Deploy to Heroku/Railway/Render

**Railway (Recommended - Easy & Free tier available):**
1. Go to https://railway.app
2. Sign up with GitHub
3. New Project → Deploy from GitHub repo
4. Select your repository
5. Add build command: `pip install -r requirements.txt`
6. Add start command: `streamlit run app.py --server.port $PORT`
7. Deploy

**Render:**
1. Go to https://render.com
2. Sign up with GitHub
3. New Web Service → Connect GitHub repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
6. Deploy

#### B. Run Locally and Share (Quick Demo)

For quick demos, you can run locally and share:

```bash
# Install ngrok (if not installed)
# brew install ngrok  # macOS
# or download from https://ngrok.com

# Run your app
streamlit run app.py

# In another terminal, expose it
ngrok http 8501

# Share the ngrok URL (e.g., https://abc123.ngrok.io)
```

### Option 4: Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Then deploy to any container platform (AWS, GCP, Azure, etc.)

## Quick Test: Try Streamlit Cloud First

**Step 1:** Go to https://share.streamlit.io and sign in

**Step 2:** Click "New app"

**Step 3:** See if your repository appears in the list

**Step 4:** If yes → Deploy! If no → Use Option 3 (Railway/Render)

## Troubleshooting

**"Repository not found":**
- Streamlit Cloud doesn't support GitHub Enterprise
- Use Railway, Render, or manual deployment

**"Import errors":**
- Check that all files in `src/` are committed
- Verify `requirements.txt` has all dependencies

**"App won't start":**
- Check Streamlit Cloud logs
- Ensure `app.py` is in the root directory
- Verify Python version compatibility

## Recommended: Try Railway First

Railway is the easiest alternative and works with GitHub Enterprise:
1. https://railway.app
2. Sign up with GitHub
3. Deploy from your repo
4. Done!

