---
title: Fashion Advisor RAG
emoji: 👗
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 👗 Fashion Advisor - Complete RAG System

An intelligent fashion advisor powered by Retrieval-Augmented Generation (RAG) with FAISS vector search and local language models.

## 🌟 Features

- **🦜 LangChain Integration**: Production-ready RAG orchestration
- **🔍 FAISS Vector Store**: Fast similarity search across 15,000+ fashion documents
- **📚 Multi-query Decomposition**: Enhanced retrieval with query expansion
- **🤖 Local LLM**: Free models (Phi-3, FLAN-T5) - no API keys needed
- **🛡️ Anti-hallucination**: Multi-layer verification system
- **💬 Natural Conversation**: Clean chat interface powered by Gradio

## 🎯 What I Can Help With

- **Occasions**: Business meetings, weddings, casual outings, presentations
- **Colors**: Combinations, seasonal palettes, skin tone matching
- **Styling**: Layering, accessories, proportions, trends
- **Body Types**: Flattering silhouettes and fits
- **Wardrobe**: Essential pieces, capsule wardrobes, outfit planning

## 🏗️ Technology Stack

- **Framework**: LangChain 0.1.0
- **Vector Store**: FAISS with 15,000+ documents
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: Microsoft Phi-3-mini / Google FLAN-T5
- **UI**: Gradio 4.44.0
- **Data**: HuggingFace Datasets + curated fashion articles

## 📊 Data Sources

- ✅ 3,000+ fashion products from e-commerce data
- ✅ 5,000+ sustainable fashion Q&A pairs
- ✅ 200+ curated fashion articles
- ✅ 40+ expert fashion principles (verified)

## 🚀 Architecture

```
User Query → Query Expansion → Embedding → FAISS Search → 
Document Ranking → LLM Generation → Response
```

### Pipeline Steps:
1. **Query Processing**: Original + semantic expansion
2. **Embedding**: 384-dimensional vectors (normalized)
3. **Retrieval**: Top-15 documents per query variant
4. **Deduplication**: Remove overlapping content
5. **Ranking**: Score by relevance + verification status
6. **Generation**: 4-attempt progressive LLM generation
7. **Fallback**: Direct synthesis if LLM fails

## 💪 Performance

- ⚡ Average response time: 2-3 seconds
- 🎯 High retrieval accuracy with verified sources
- 🛡️ Multi-layer hallucination prevention
- 🔄 Automatic fallback mechanisms

## 🔧 Setup Instructions

### Prerequisites
Ensure you have the FAISS vector store directory:
```
faiss_vectorstore/
  ├── index.faiss
  ├── index.pkl
  └── config.json
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

### Hugging Face Spaces Deployment
1. Upload `app.py`, `requirements.txt`, and `README.md`
2. Upload your `faiss_vectorstore/` directory with all files
3. Space will automatically install dependencies and launch

## 📝 Example Questions

- "What should I wear to a business meeting?"
- "What colors go well with navy blue?"
- "What are essential wardrobe items for fall?"
- "How to dress for a summer wedding?"
- "What's the best outfit for a university presentation?"

## 🛠️ Technical Details

### Models Used:
- **Primary LLM**: microsoft/Phi-3-mini-4k-instruct (3.8B params)
- **Backup LLM**: google/flan-t5-large (780M params)
- **Fallback LLM**: google/flan-t5-base (250M params)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)

### Key Parameters:
- Temperature: 0.75-1.0 (progressive)
- Max tokens: 350-600 (progressive)
- Top-p: 0.92-0.97
- Retrieval: Top-15 docs per query
- Context: Top-8 scored documents

## 🔒 Privacy & Security

- ✅ No external API calls (runs locally)
- ✅ No user data stored
- ✅ Open-source models only
- ✅ Verified content sources

## 📄 License

This project uses open-source models and datasets. Please refer to individual model licenses:
- Microsoft Phi-3: MIT License
- Google FLAN-T5: Apache 2.0
- Sentence Transformers: Apache 2.0

## 🤝 Contributing

Built with ❤️ for the fashion AI community. Feel free to fork and improve!

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Note**: First run may take 1-2 minutes to download models (~4GB total). Subsequent runs are instant.
