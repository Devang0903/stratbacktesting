# Quick Deployment Steps

## 1. Commit Your Code

```bash
git add .
git commit -m "Initial commit: DJIA Mean-Reversion Trading Strategy"
```

## 2. Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `djia-mean-reversion-strategy` (or your choice)
3. **DO NOT** check "Initialize with README"
4. Click "Create repository"

## 3. Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/djia-mean-reversion-strategy.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## 4. Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Main file: `app.py`
6. Click "Deploy"

## That's it! 🎉

Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

**Note**: First run may take a few minutes to install dependencies and fetch data.
