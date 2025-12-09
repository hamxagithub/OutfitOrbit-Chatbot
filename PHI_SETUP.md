# PHI Model Setup for Hugging Face Spaces (LOCAL LOADING)

Your Fashion Advisor RAG is now configured to **load Microsoft PHI models locally** with CPU optimizations.

## Quick Setup Steps

### 1. Dependencies
The `requirements.txt` now includes all necessary packages:
- `transformers>=4.36.0` - PHI model support
- `accelerate>=0.25.0` - Efficient loading
- `bitsandbytes>=0.41.0` - 8-bit quantization (optional but recommended)

### 2. Model Selection (Optional)
By default, the app loads **microsoft/phi-2** (2.7B params - best for CPU).

To use a different PHI model, add to Space Settings → Variables:
- **LOCAL_PHI_MODEL** = `microsoft/Phi-3-mini-4k-instruct` (3.8B - needs more RAM)

### 3. Deploy to Hugging Face Spaces

1. Push your code to the Space repository
2. The model will auto-download on first launch (takes 2-3 minutes)
3. Watch logs for: `✅ PHI model initialized successfully: microsoft/phi-2`
4. Test with a fashion question!

**No API keys needed!** The model runs entirely locally on your Space.

## Why PHI Models (Local)?

✅ **Excellent instruction-following** - Better at structured, detailed answers  
✅ **Compact size** - 2.7B-3.8B parameters = fits in CPU memory  
✅ **No API needed** - Completely self-contained and private  
✅ **Natural language generation** - Produces fluent ~320-420 word responses  
✅ **8-bit quantization** - Reduces memory usage by ~50%
✅ **Optimized for CPU** - Works on free Hugging Face Spaces

## Expected Behavior

- **Initialization**: `🔄 Initializing local PHI model: microsoft/phi-2`
- **Loading time**: 30-60 seconds on first launch (downloads ~5GB model)
- **Generation**: Fast inference directly on your Space's CPU
- **Fallback**: Scaffold-and-polish → extractive answer if PHI fails
- **Speed**: ~10-25 seconds per response (after model loaded)

## Troubleshooting

### "Out of memory" or Space crashes
→ PHI-2 (2.7B) should work on free CPU Spaces. If it crashes:
  - Wait and retry (Space may be under load)
  - Use `LOCAL_PHI_MODEL=microsoft/phi-2` explicitly
  - Enable 8-bit quantization (already on by default)

### "Missing required library: bitsandbytes"
→ Some environments don't support bitsandbytes. The app will fall back to float32 automatically.

### Model download takes too long
→ First launch downloads ~5GB. This is normal. Subsequent launches are fast (model cached).

### Short/empty answers
→ Check logs for PHI model errors. The app will use scaffold-and-polish or extractive fallback.

## Testing Locally (Optional)

Test on your machine before deploying:

```powershell
# Windows PowerShell
pip install -r requirements.txt
python app.py
```

```bash
# Linux/Mac
pip install -r requirements.txt
python app.py
```

The model will download automatically (~5GB for phi-2).

## Advanced: Using PHI-3 Instead of PHI-2

PHI-3 is larger (3.8B params) and may need more memory:

```powershell
# Set environment variable
$env:LOCAL_PHI_MODEL = "microsoft/Phi-3-mini-4k-instruct"
python app.py
```

Or add to Space Settings → Variables:
```
LOCAL_PHI_MODEL = microsoft/Phi-3-mini-4k-instruct
```

---

**Ready to deploy!** Push to Hugging Face Spaces and let the model auto-load. 🚀
