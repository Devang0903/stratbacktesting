#!/bin/bash
# Quick deployment script for Streamlit Community Cloud

echo "🚀 Preparing for Streamlit Cloud Deployment"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Add all necessary files
echo "📦 Staging files..."
git add .gitignore
git add requirements.txt
git add README.md
git add DEPLOYMENT.md
git add app.py
git add run_app.sh
git add src/
git add .streamlit/

# Show what will be committed
echo ""
echo "Files to be committed:"
git status --short

echo ""
read -p "Do you want to commit these files? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git commit -m "Initial commit: DJIA Mean-Reversion Trading Strategy with Streamlit app"
    echo "✅ Committed!"
    echo ""
    echo "📤 Next steps:"
    echo "1. Create a repository on GitHub (https://github.com/new)"
    echo "2. Run: git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git"
    echo "3. Run: git push -u origin main"
    echo "4. Go to https://share.streamlit.io and deploy your app"
    echo ""
    echo "See DEPLOYMENT.md for detailed instructions"
else
    echo "❌ Commit cancelled"
fi

