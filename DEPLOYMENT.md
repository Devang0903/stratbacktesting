# Deployment Guide for Streamlit Community Cloud

## Prerequisites

1. GitHub account
2. Streamlit Community Cloud account (free at https://share.streamlit.io)

## Step 1: Initialize Git Repository

```bash
cd /Users/devang/Desktop/SIE/TradingStrat
git init
git add .
git commit -m "Initial commit: DJIA Mean-Reversion Trading Strategy"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (e.g., `djia-mean-reversion-strategy`)
3. **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Copy the repository URL

## Step 3: Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/djia-mean-reversion-strategy.git
git branch -M main
git push -u origin main
```

## Step 4: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `djia-mean-reversion-strategy`
5. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.10 (or latest available)
6. Click "Deploy"

## Step 5: Configure App Settings (Optional)

After deployment, you can:
- Set custom domain
- Configure secrets (if needed for API keys)
- Set environment variables

## Important Notes

- The app will use cached data if available
- First run may take longer if fetching data
- Data files are excluded from git (see .gitignore)
- The app will create data/ and plots/ directories automatically

## Troubleshooting

**App won't start:**
- Check that `app.py` is in the root directory
- Verify `requirements.txt` includes all dependencies
- Check Streamlit Cloud logs for errors

**Import errors:**
- Ensure all modules in `src/` are included
- Check that `__init__.py` exists in `src/` directory

**Data not loading:**
- App will fetch from Yahoo Finance if cache not available
- First run may take time to fetch data

