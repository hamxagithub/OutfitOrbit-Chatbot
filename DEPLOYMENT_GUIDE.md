# 🚀 Hugging Face Spaces Deployment Guide

## 📋 Prerequisites

Before deploying, ensure you have:

1. ✅ **FAISS Vector Store** - The `faiss_vectorstore/` directory with:
   - `index.faiss` (vector index file)
   - `index.pkl` (document metadata)
   - `config.json` (optional configuration)

2. ✅ **Required Files**:
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `README.md` (space documentation)

## 🎯 Step-by-Step Deployment

### Step 1: Create Hugging Face Account
1. Go to https://huggingface.co/
2. Sign up or log in
3. Navigate to "Spaces" section

### Step 2: Create New Space
1. Click "Create new Space"
2. Choose:
   - **Space name**: `fashion-advisor-rag` (or your preferred name)
   - **License**: Apache 2.0 (recommended)
   - **SDK**: Gradio
   - **Space hardware**: CPU Basic (free) or upgrade for faster performance
3. Click "Create Space"

### Step 3: Upload Files

#### Option A: Web Interface (Easiest)
1. In your new Space, click "Files and versions"
2. Click "Upload files"
3. Upload these files:
   ```
   app.py
   requirements.txt
   README.md
   ```
4. Create folder `faiss_vectorstore/` and upload:
   ```
   index.faiss
   index.pkl
   config.json (if you have it)
   ```

#### Option B: Git (Advanced)
```bash
# Clone your space
git clone https://huggingface.co/spaces/<your-username>/<space-name>
cd <space-name>

# Copy files
cp path/to/app.py .
cp path/to/requirements.txt .
cp path/to/README.md .
cp -r path/to/faiss_vectorstore .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

### Step 4: Wait for Build
1. Hugging Face will automatically:
   - Install dependencies from `requirements.txt`
   - Download LLM models (Phi-3 or FLAN-T5)
   - Start the Gradio app
2. First build takes **5-10 minutes** (downloading models)
3. Watch the "Logs" tab for progress

### Step 5: Test Your Space
1. Once "Running" appears, test with examples:
   - "What should I wear to a business meeting?"
   - "What colors go well with navy blue?"
2. Share your Space URL with others!

## 📁 Required Directory Structure

Your Space should look like this:
```
your-space/
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── README.md              # Space documentation
└── faiss_vectorstore/     # Vector store (REQUIRED)
    ├── index.faiss        # Vector index (~50-200MB)
    ├── index.pkl          # Document metadata (~10-50MB)
    └── config.json        # Optional configuration
```

## ⚙️ Configuration Options

### Hardware Requirements:
- **CPU Basic (Free)**: Works but slower (5-10s per response)
- **CPU Upgrade**: Faster responses (2-3s per response)
- **GPU T4**: Fastest (1-2s per response) - recommended for production

### Model Selection:
The app automatically tries models in order:
1. **microsoft/Phi-3-mini-4k-instruct** (3.8GB) - Best quality
2. **google/flan-t5-large** (780MB) - Good balance
3. **google/flan-t5-base** (250MB) - Fastest

Lower memory = faster download but lower quality responses.

## 🐛 Troubleshooting

### Issue: "Application startup failed"
**Solution**: Check logs for missing dependencies
```bash
# Add missing package to requirements.txt
```

### Issue: "Vector store not found"
**Solution**: Ensure `faiss_vectorstore/` folder is uploaded with all files:
- `index.faiss`
- `index.pkl`

### Issue: "Out of memory"
**Solutions**:
1. Upgrade to GPU hardware
2. Use smaller model (FLAN-T5-base)
3. Reduce `max_length` in `app.py` (line 71)

### Issue: "Model download timeout"
**Solution**: 
1. Wait longer (first run downloads ~4GB)
2. Restart the Space
3. Use smaller model in `BACKUP_MODELS` list

## 🔧 Customization

### Change Models:
Edit `BACKUP_MODELS` in `app.py` (around line 51):
```python
BACKUP_MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",  # Your preferred model
    "google/flan-t5-large",
    "google/flan-t5-base",
]
```

### Adjust Response Length:
Edit parameters in `generate_llm_answer()` (around line 252):
```python
max_tokens = 350  # Increase for longer responses
temperature = 0.75  # Increase for more creative responses
```

### Change UI Theme:
Edit Gradio theme in `app.py` (line 422):
```python
theme=gr.themes.Soft()  # Options: Soft, Default, Glass, Monochrome
```

## 📊 Monitoring

### Check Logs:
1. Go to your Space page
2. Click "Logs" tab
3. Monitor for errors or warnings

### Analytics:
- View usage stats in Space settings
- Track response times
- Monitor uptime

## 🔒 Privacy & Security

### Free Tier Limits:
- CPU Basic: ~100 concurrent users
- Storage: 50GB max
- No persistent storage (files deleted on restart)

### Making Private:
1. Go to Space settings
2. Change visibility to "Private"
3. Share only with specific users

## 📈 Performance Optimization

### For Faster Responses:
1. **Upgrade hardware** to GPU T4
2. **Use smaller model** (FLAN-T5-base)
3. **Reduce top_k** in `CONFIG` (line 30):
   ```python
   "top_k": 10,  # Instead of 15
   ```
4. **Cache embeddings** (already implemented)

### For Better Quality:
1. Use **Phi-3-mini** model (default)
2. Increase **temperature** for creativity
3. Add more **context documents** (increase top_k)

## 🆘 Support

If you encounter issues:
1. Check Hugging Face Spaces docs: https://huggingface.co/docs/hub/spaces
2. Check model requirements: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
3. Post in Hugging Face forums: https://discuss.huggingface.co/

## ✅ Checklist Before Deployment

- [ ] `app.py` uploaded
- [ ] `requirements.txt` uploaded
- [ ] `README.md` uploaded
- [ ] `faiss_vectorstore/index.faiss` uploaded
- [ ] `faiss_vectorstore/index.pkl` uploaded
- [ ] Space visibility set (public/private)
- [ ] Hardware tier selected
- [ ] Space name chosen

## 🎉 Success!

Once deployed, your Space URL will be:
```
https://huggingface.co/spaces/<your-username>/<space-name>
```

Share it with the world! 🌍

---

**Estimated Deployment Time**: 10-15 minutes (first time)  
**Storage Required**: ~300MB (vector store) + 4GB (models)  
**Cost**: FREE on CPU Basic tier
