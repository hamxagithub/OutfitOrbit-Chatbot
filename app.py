"""
Fashion Advisor RAG - Hugging Face Deployment
Complete RAG system with FAISS vector store and local LLM
"""

import gradio as gr
import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pickle

# Core ML libraries
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": None,  # Will be set during initialization
    "vector_store_path": ".",  # Root directory (files are in root on HF Spaces)
    "top_k": 15,
    "temperature": 0.75,
    "max_tokens": 350,
}

# ============================================================================
# INITIALIZE MODELS
# ============================================================================

def initialize_llm():
    """Initialize free local LLM with transformers pipeline"""
    logger.info("🔄 Initializing FREE local language model...")
    
    BACKUP_MODELS = [
        "microsoft/Phi-3-mini-4k-instruct",  # Primary - 3.8B, very efficient
        "google/flan-t5-large",  # Backup - 780M, good quality
        "google/flan-t5-base",  # Fallback - 250M, fast
    ]
    
    for model_name in BACKUP_MODELS:
        try:
            logger.info(f"   Trying {model_name}...")
            device = 0 if torch.cuda.is_available() else -1
            
            llm_client = pipeline(
                "text-generation",
                model=model_name,
                device=device,
                max_length=512,
                truncation=True,
            )
            
            CONFIG["llm_model"] = model_name
            logger.info(f"✅ FREE LLM initialized: {model_name}")
            logger.info(f"   Device: {'GPU' if device == 0 else 'CPU'}")
            return llm_client
            
        except Exception as e:
            logger.warning(f"⚠️ Failed {model_name}: {str(e)[:100]}")
            continue
    
    logger.error("⚠️ All models failed - will use fallback generation")
    return None

def initialize_embeddings():
    """Initialize sentence transformer embeddings"""
    logger.info("🔄 Initializing embeddings model...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG["embedding_model"],
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    logger.info(f"✅ Embeddings initialized: {CONFIG['embedding_model']}")
    return embeddings

def load_vector_store(embeddings):
    """Load FAISS vector store with Pydantic monkey-patch"""
    logger.info("🔄 Loading FAISS vector store...")
    
    vector_store_path = CONFIG["vector_store_path"]
    
    # Check for required FAISS files
    index_file = os.path.join(vector_store_path, "index.faiss")
    pkl_file = os.path.join(vector_store_path, "index.pkl")
    
    if not os.path.exists(index_file):
        logger.error(f"❌ index.faiss not found at {index_file}")
        raise FileNotFoundError(f"FAISS index file not found: {index_file}")
    
    if not os.path.exists(pkl_file):
        logger.error(f"❌ index.pkl not found at {pkl_file}")
        raise FileNotFoundError(f"FAISS metadata file not found: {pkl_file}")
    
    logger.info(f"✅ Found index.faiss ({os.path.getsize(index_file)/1024/1024:.2f} MB)")
    logger.info(f"✅ Found index.pkl ({os.path.getsize(pkl_file)/1024:.2f} KB)")
    
    try:
        # Try standard loading first
        vectorstore = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"✅ FAISS vector store loaded successfully")
        return vectorstore
        
    except (KeyError, AttributeError, Exception) as e:
        logger.warning(f"⚠️ Pydantic compatibility issue: {str(e)[:100]}")
        logger.info("🔄 Applying Pydantic monkey-patch and retrying...")
        
        # STEP 1: Monkey-patch Pydantic to handle missing __fields_set__
        try:
            import pydantic.v1.main as pydantic_main
            
            # Save original __setstate__
            original_setstate = pydantic_main.BaseModel.__setstate__
            
            def patched_setstate(self, state):
                """Patched __setstate__ that handles missing __fields_set__"""
                # Add missing __fields_set__ if not present
                if '__fields_set__' not in state:
                    state['__fields_set__'] = set(state.get('__dict__', {}).keys())
                # Call original
                return original_setstate(self, state)
            
            # Apply patch
            pydantic_main.BaseModel.__setstate__ = patched_setstate
            logger.info("   ✅ Pydantic monkey-patch applied")
            
        except Exception as patch_error:
            logger.warning(f"   ⚠️ Pydantic patch failed: {patch_error}")
        
        # STEP 2: Try loading again with patch
        try:
            vectorstore = FAISS.load_local(
                vector_store_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"✅ FAISS vector store loaded with Pydantic patch")
            return vectorstore
            
        except Exception as e2:
            logger.error(f"   ✗ Still failed after patch: {str(e2)[:100]}")
            
            # STEP 3: Last resort - manual reconstruction
            logger.info("🔄 Using manual reconstruction (last resort)...")
            
            import faiss
            import pickle
            from langchain_community.docstore.in_memory import InMemoryDocstore
            
            # Load FAISS index
            index = faiss.read_index(index_file)
            logger.info(f"   ✅ FAISS index loaded")
            
            # Load pickle with raw binary parsing
            with open(pkl_file, "rb") as f:
                import io
                import struct
                
                # Read raw bytes
                raw_bytes = f.read()
                logger.info(f"   Read {len(raw_bytes)} bytes from pickle")
                
                # Try to extract text content directly (bypass Pydantic completely)
                # This is a fallback that extracts document strings
                import re
                
                # Find all text patterns that look like documents
                text_pattern = rb'([A-Za-z0-9\s\.\,\;\:\!\?\-\'\"\(\)]{50,})'
                matches = re.findall(text_pattern, raw_bytes)
                
                if len(matches) > 100:
                    logger.info(f"   Found {len(matches)} potential document fragments")
                    
                    # Create simple documents from extracted text
                    new_docstore_dict = {}
                    index_to_docstore_id = {}
                    
                    for idx, match in enumerate(matches[:15000]):  # Limit to 15k docs
                        try:
                            content = match.decode('utf-8', errors='ignore').strip()
                            if len(content) > 50:  # Only keep substantial content
                                doc_id = str(idx)
                                new_doc = Document(
                                    page_content=content,
                                    metadata={}
                                )
                                new_docstore_dict[doc_id] = new_doc
                                index_to_docstore_id[idx] = doc_id
                        except:
                            continue
                    
                    logger.info(f"   ✅ Reconstructed {len(new_docstore_dict)} documents from raw data")
                    
                    docstore = InMemoryDocstore(new_docstore_dict)
                    
                    vectorstore = FAISS(
                        embedding_function=embeddings,
                        index=index,
                        docstore=docstore,
                        index_to_docstore_id=index_to_docstore_id
                    )
                    
                    logger.info(f"✅ FAISS vector store reconstructed from raw data")
                    return vectorstore
                else:
                    raise Exception("Could not extract enough document content from pickle")

# ============================================================================
# RAG PIPELINE FUNCTIONS
# ============================================================================

def retrieve_knowledge_langchain(
    query: str,
    vectorstore,
    top_k: int = 15
) -> Tuple[List[Document], float]:
    """
    Retrieve relevant documents using LangChain FAISS with query expansion
    """
    logger.info(f"🔍 Retrieving knowledge for: '{query}'")
    
    # Create query variants for better coverage
    query_variants = [
        query,  # Original
        f"fashion advice clothing outfit style for {query}",  # Semantic expansion
    ]
    
    all_docs = []
    
    # Retrieve for each variant
    for variant in query_variants:
        try:
            docs_and_scores = vectorstore.similarity_search_with_score(variant, k=top_k)
            
            for doc, score in docs_and_scores:
                similarity = 1.0 / (1.0 + score)
                doc.metadata['similarity'] = similarity
                doc.metadata['query_variant'] = variant
                all_docs.append(doc)
                
        except Exception as e:
            logger.error(f"Retrieval error for variant '{variant}': {e}")
    
    # Deduplicate by content
    unique_docs = {}
    for doc in all_docs:
        content_key = doc.page_content[:100]
        if content_key not in unique_docs:
            unique_docs[content_key] = doc
        else:
            # Keep document with higher similarity
            if doc.metadata.get('similarity', 0) > unique_docs[content_key].metadata.get('similarity', 0):
                unique_docs[content_key] = doc
    
    final_docs = list(unique_docs.values())
    
    # Sort by similarity
    final_docs.sort(key=lambda x: x.metadata.get('similarity', 0), reverse=True)
    
    # Calculate confidence
    if final_docs:
        avg_similarity = sum(d.metadata.get('similarity', 0) for d in final_docs) / len(final_docs)
        confidence = min(avg_similarity, 1.0)
    else:
        confidence = 0.0
    
    logger.info(f"✅ Retrieved {len(final_docs)} unique documents (confidence: {confidence:.2f})")
    
    return final_docs, confidence

def generate_llm_answer(
    query: str,
    retrieved_docs: List[Document],
    llm_client,
    attempt: int = 1
) -> Optional[str]:
    """
    Generate answer using local LLM with retrieved context
    """
    if not llm_client:
        logger.error("  → LLM client not initialized")
        return None
    
    # Build focused context
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Score documents by relevance
    scored_docs = []
    for doc in retrieved_docs[:20]:
        content = doc.page_content.lower()
        doc_words = set(content.split())
        overlap = len(query_words.intersection(doc_words))
        
        # Boost for verified/curated
        if doc.metadata.get('verified', False):
            overlap += 10
        
        # Boost for longer content
        if len(doc.page_content) > 200:
            overlap += 3
        
        scored_docs.append((doc, overlap))
    
    # Sort and take top 8
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc[0] for doc in scored_docs[:8]]
    
    # Build context
    context_parts = []
    for doc in top_docs:
        content = doc.page_content.strip()
        if len(content) > 400:
            content = content[:400] + "..."
        context_parts.append(content)
    
    context_text = "\n\n".join(context_parts)
    
    # Progressive parameters based on attempt
    if attempt == 1:
        temperature = 0.75
        max_tokens = 350
        top_p = 0.92
        repetition_penalty = 1.1
    elif attempt == 2:
        temperature = 0.85
        max_tokens = 450
        top_p = 0.94
        repetition_penalty = 1.15
    elif attempt == 3:
        temperature = 0.92
        max_tokens = 550
        top_p = 0.96
        repetition_penalty = 1.2
    else:
        temperature = 1.0
        max_tokens = 600
        top_p = 0.97
        repetition_penalty = 1.25
    
    # Create prompt
    user_prompt = f"""[INST] Question: {query}

Fashion Knowledge:
{context_text}

Answer the question using the knowledge above. Be specific and helpful (100-250 words). [/INST]"""

    try:
        logger.info(f"  → Calling {CONFIG['llm_model']} (temp={temperature}, tokens={max_tokens})...")
        
        # Call pipeline
        output = llm_client(
            user_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            return_full_text=False,
            pad_token_id=llm_client.tokenizer.eos_token_id
        )
        
        # Extract generated text
        response = output[0]['generated_text'].strip()
        
        if not response:
            logger.warning(f"  ✗ Empty response (attempt {attempt})")
            return None
        
        # Minimal validation
        if len(response) < 20:
            logger.warning(f"  ✗ Response too short: {len(response)} chars")
            return None
        
        # Check for apologies/refusals
        apology_phrases = ["i cannot", "i can't", "i'm sorry", "i apologize", "i don't have"]
        if any(phrase in response.lower()[:100] for phrase in apology_phrases):
            logger.warning(f"  ✗ Apology detected")
            return None
        
        logger.info(f"  ✅ Generated answer ({len(response)} chars)")
        return response
        
    except Exception as e:
        logger.error(f"  ✗ Generation error: {e}")
        return None

def synthesize_direct_answer(
    query: str,
    retrieved_docs: List[Document]
) -> str:
    """
    Fallback: Synthesize answer directly from most relevant documents
    """
    logger.info("  → Using fallback: direct synthesis")
    
    if not retrieved_docs:
        return "I don't have enough information to answer that question accurately."
    
    # Get most relevant document
    best_doc = retrieved_docs[0]
    content = best_doc.page_content.strip()
    
    # Create answer from top document
    if len(content) > 500:
        answer = content[:500] + "..."
    else:
        answer = content
    
    return answer

def generate_answer_langchain(
    query: str,
    vectorstore,
    llm_client
) -> str:
    """
    Main RAG pipeline: Retrieve → Generate → Fallback
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing query: '{query}'")
    logger.info(f"{'='*80}")
    
    # Step 1: Retrieve documents
    retrieved_docs, confidence = retrieve_knowledge_langchain(
        query,
        vectorstore,
        top_k=CONFIG["top_k"]
    )
    
    if not retrieved_docs:
        return "I couldn't find relevant information to answer your question."
    
    # Step 2: Try LLM generation (4 attempts)
    llm_answer = None
    for attempt in range(1, 5):
        logger.info(f"\n  🤖 LLM Generation Attempt {attempt}/4")
        llm_answer = generate_llm_answer(query, retrieved_docs, llm_client, attempt)
        
        if llm_answer:
            logger.info(f"  ✅ LLM answer generated successfully")
            break
        else:
            logger.warning(f"  → Attempt {attempt}/4 failed, retrying...")
    
    # Step 3: Fallback if all attempts fail
    if not llm_answer:
        logger.error(f"  ✗ All 4 LLM attempts failed - using fallback")
        llm_answer = synthesize_direct_answer(query, retrieved_docs)
    
    return llm_answer

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def fashion_chatbot(message: str, history: List[List[str]]) -> str:
    """
    Chatbot function for Gradio interface
    """
    try:
        if not message or not message.strip():
            return "Please ask a fashion-related question!"
        
        # Generate answer using RAG pipeline
        answer = generate_answer_langchain(
            message.strip(),
            vectorstore,
            llm_client
        )
        
        return answer
        
    except Exception as e:
        logger.error(f"Error in chatbot: {e}")
        return f"Sorry, I encountered an error: {str(e)}"

# ============================================================================
# INITIALIZE AND LAUNCH
# ============================================================================

# Global variables
llm_client = None
embeddings = None
vectorstore = None

def startup():
    """Initialize all models and load vector store"""
    global llm_client, embeddings, vectorstore
    
    logger.info("🚀 Starting Fashion Advisor RAG...")
    
    # Initialize embeddings
    embeddings = initialize_embeddings()
    
    # Load vector store
    vectorstore = load_vector_store(embeddings)
    
    # Initialize LLM
    llm_client = initialize_llm()
    
    logger.info("✅ All components initialized successfully!")

# Initialize on startup
startup()

# Create Gradio interface
demo = gr.ChatInterface(
    fn=fashion_chatbot,
    title="👗 Fashion Advisor - RAG System",
    description="""
    **Ask me anything about fashion!** 🌟
    
    I can help with:
    - Outfit recommendations for occasions
    - Color combinations and styling
    - Seasonal fashion advice
    - Body type and fit guidance
    - Wardrobe essentials
    
    *Powered by RAG with FAISS vector search and local LLM*
    """,
    examples=[
        "What should I wear to a business meeting?",
        "What colors go well with navy blue?",
        "What are essential wardrobe items for fall?",
        "How to dress for a summer wedding?",
        "What's the best outfit for a university presentation?",
    ],
    theme=gr.themes.Soft(),
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear Chat",
)

# Launch
if __name__ == "__main__":
    demo.launch()
