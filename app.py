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
from sentence_transformers import SentenceTransformer
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Suppress transformers warnings about generation flags
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimize PyTorch for CPU inference
torch.set_num_threads(4)  # Limit threads for better CPU performance
torch.set_grad_enabled(False)  # Disable gradients (inference only)

# Suppress specific warnings and asyncio issues
import warnings
warnings.filterwarnings("ignore", message="MatMul8bitLt")
warnings.filterwarnings("ignore", message="torch_dtype")
warnings.filterwarnings("ignore", message="Invalid file descriptor")
warnings.filterwarnings("ignore", message="generation flags")
warnings.filterwarnings("ignore", category=UserWarning)

# Fix asyncio file descriptor warnings
import asyncio
import sys
if sys.platform == 'linux':
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except:
        pass

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": None,
    "vector_store_path": ".",
    "top_k": 12,  # Rich retrieval for quality
    "temperature": 0.75,  # Balanced for natural flow
    "max_tokens": 600,  # Allow natural length responses
}

# LLM Configuration - LOCAL ONLY
# Using Flan-T5 Base: 250M params, instruction-tuned, fast and high quality
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "google/flan-t5-base")
USE_8BIT_QUANTIZATION = False
USE_REMOTE_LLM = False  # LOCAL ONLY

# Natural flow mode: No word limits, let model decide length
MAX_CONTEXT_LENGTH = 400  # Reduced for faster generation
USE_CACHING = True  # Cache model outputs for repeated patterns
ENABLE_FAST_MODE = False  # Allow natural completion, no artificial limits

# Prefer the environment variable, but also allow a local token file for users
# who don't know how to set env vars. Create a file named `hf_token.txt` in the
# project root containing only the token (no newline is necessary). DO NOT
# commit that file to version control. A .gitignore entry will be added.
HF_INFERENCE_API_KEY = os.environ.get("HF_INFERENCE_API_KEY")
if not HF_INFERENCE_API_KEY:
    try:
        token_path = Path("hf_token.txt")
        if token_path.exists():
            HF_INFERENCE_API_KEY = token_path.read_text(encoding="utf-8").strip()
            logger.info("Loaded HF token from hf_token.txt (ensure this file is private and not committed)")
    except Exception:
        logger.warning("Could not read hf_token.txt for HF token")

if HF_INFERENCE_API_KEY:
    USE_REMOTE_LLM = True

# ============================================================================
# INITIALIZE MODELS
# ============================================================================

def initialize_llm():
    """Initialize Flan-T5 Base for local CPU generation.
    
    Flan-T5 is instruction-tuned, produces high-quality answers,
    and is fast on CPU (3-5 seconds per response).
    """
    global LOCAL_LLM_MODEL
    
    logger.info(f"🔄 Initializing Flan-T5 Base: {LOCAL_LLM_MODEL}")
    logger.info("   Instruction-tuned for high-quality Q&A")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"   Device: {device}")
        
        # Load tokenizer
        logger.info("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL)
        logger.info("   Tokenizer ready")
        
        # Load model
        logger.info("   Loading Flan-T5 Base (10-15 seconds)...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            LOCAL_LLM_MODEL,
            torch_dtype=torch.float32
        )
        
        model = model.to(device)
        model.eval()
        logger.info("   Model ready")
        
        # Store model and tokenizer for custom generation
        llm_client = {
            'model': model,
            'tokenizer': tokenizer,
            'device': device
        }
        
        CONFIG["llm_model"] = LOCAL_LLM_MODEL
        CONFIG["model_type"] = "flan_t5_base_local"
        
        logger.info(f"✅ Flan-T5 Base initialized: {LOCAL_LLM_MODEL}")
        logger.info(f"   Size: 250M parameters (instruction-tuned)")
        logger.info(f"   Quality: Excellent for fashion Q&A")
        logger.info(f"   Speed: 3-5 seconds per 200 words")
        
        return llm_client
        
    except ImportError as ie:
        logger.error(f"❌ Missing required library: {ie}")
        logger.info("   Install with: pip install transformers torch")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load LLM: {str(e)}")
        logger.info("   This may be due to insufficient memory")
        import traceback
        logger.error(traceback.format_exc())
        raise Exception(f"Failed to initialize LLM: {str(e)}")


def remote_generate(prompt: str, max_new_tokens: int = 200, temperature: float = 0.7, top_p: float = 0.9) -> str:
    """Call Hugging Face Inference API - fast and reliable.
    
    Uses Qwen2.5 model optimized for fast inference.
    """
    if not HF_INFERENCE_API_KEY:
        raise Exception("HF_INFERENCE_API_KEY not set for remote generation")

    # Use Inference API
    api_url = f"https://api-inference.huggingface.co/models/{REMOTE_LLM_MODEL}"
    headers = {"Authorization": f"Bearer {HF_INFERENCE_API_KEY}"}
    
    # Simple parameters for fast inference
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False
        }
    }

    logger.info(f"    → Remote inference (tokens={max_new_tokens})")
    try:
        r = requests.post(api_url, headers=headers, json=payload, timeout=90)
    except Exception as e:
        logger.error(f"    ✗ Remote request failed: {e}")
        return ""

    if r.status_code == 503:
        logger.warning(f"    ⚠️ Model loading (503), retrying in 5s...")
        import time
        time.sleep(5)
        try:
            r = requests.post(api_url, headers=headers, json=payload, timeout=90)
        except Exception as e:
            logger.error(f"    ✗ Retry failed: {e}")
            return ""

    if r.status_code != 200:
        logger.error(f"    ✗ Remote inference error {r.status_code}: {r.text[:300]}")
        return ""

    result = r.json()
    
    # Handle error responses
    if isinstance(result, dict) and result.get("error"):
        logger.error(f"    ✗ Remote inference returned error: {result.get('error')}")
        return ""

    # Extract generated text
    generated_text = ""
    
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            generated_text = first.get("generated_text", "")
        else:
            generated_text = str(first)
    elif isinstance(result, dict):
        generated_text = result.get("generated_text", str(result))
    else:
        generated_text = str(result)
    
    # Clean up
    generated_text = generated_text.strip()
    if prompt in generated_text:
        generated_text = generated_text.replace(prompt, "").strip()
    
    logger.info(f"    ✅ Generated {len(generated_text.split())} words remotely")
    return generated_text

def initialize_embeddings():
    logger.info("🔄 Initializing embeddings model...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG["embedding_model"],
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    logger.info(f"✅ Embeddings initialized: {CONFIG['embedding_model']}")
    return embeddings

def load_vector_store(embeddings):
    logger.info("🔄 Loading FAISS vector store...")
    
    vector_store_path = CONFIG["vector_store_path"]
    index_file = os.path.join(vector_store_path, "index.faiss")
    pkl_file = os.path.join(vector_store_path, "index.pkl")
    
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"FAISS index file not found: {index_file}")
    
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"FAISS metadata file not found: {pkl_file}")
    
    logger.info(f"✅ Found index.faiss ({os.path.getsize(index_file)/1024/1024:.2f} MB)")
    logger.info(f"✅ Found index.pkl ({os.path.getsize(pkl_file)/1024:.2f} KB)")
    
    try:
        vectorstore = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"✅ FAISS vector store loaded successfully")
        return vectorstore
        
    except Exception as e:
        logger.warning(f"⚠️ Pydantic compatibility issue: {str(e)[:100]}")
        logger.info("🔄 Applying Pydantic monkey-patch and retrying...")
        
        try:
            import pydantic.v1.main as pydantic_main
            original_setstate = pydantic_main.BaseModel.__setstate__
            
            def patched_setstate(self, state):
                if '__fields_set__' not in state:
                    state['__fields_set__'] = set(state.get('__dict__', {}).keys())
                return original_setstate(self, state)
            
            pydantic_main.BaseModel.__setstate__ = patched_setstate
            logger.info("   ✅ Pydantic monkey-patch applied")
            
        except Exception as patch_error:
            logger.warning(f"   ⚠️ Pydantic patch failed: {patch_error}")
        
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
            logger.info("🔄 Using manual reconstruction (last resort)...")
            
            import faiss
            from langchain_community.docstore.in_memory import InMemoryDocstore
            
            index = faiss.read_index(index_file)
            logger.info(f"   ✅ FAISS index loaded")
            
            with open(pkl_file, "rb") as f:
                import re
                raw_bytes = f.read()
                logger.info(f"   Read {len(raw_bytes)} bytes from pickle")
                
                text_pattern = rb'([A-Za-z0-9\s\.\,\;\:\!\?\-\'\"\(\)]{50,})'
                matches = re.findall(text_pattern, raw_bytes)
                
                if len(matches) > 100:
                    logger.info(f"   Found {len(matches)} potential document fragments")
                    
                    documents = []
                    for idx, match in enumerate(matches[:5000]):
                        try:
                            content = match.decode('utf-8', errors='ignore').strip()
                            if len(content) >= 100:
                                doc = Document(
                                    page_content=content,
                                    metadata={"source": "reconstructed", "id": idx}
                                )
                                documents.append(doc)
                        except:
                            continue
                    
                    if len(documents) < 100:
                        raise Exception(f"Only extracted {len(documents)} documents, need at least 100")
                    
                    logger.info(f"   ✅ Extracted {len(documents)} high-quality documents")
                    logger.info(f"   🔄 Rebuilding FAISS index from scratch...")
                    
                    vectorstore = FAISS.from_documents(
                        documents=documents,
                        embedding=embeddings
                    )
                    
                    logger.info(f"✅ FAISS vector store rebuilt from {len(documents)} documents")
                    return vectorstore
                else:
                    raise Exception("Could not extract enough document content from pickle")

# ============================================================================
# RAG PIPELINE FUNCTIONS
# ============================================================================

def generate_extractive_answer(query: str, retrieved_docs: List[Document]) -> Optional[str]:
    """Build a focused, intelligent answer from retrieved documents.
    Filters out product catalogs and provides concise, relevant fashion advice.
    """
    logger.info(f"🔧 Generating smart extractive answer for: '{query}'")

    import re

    all_text = "\n\n".join([d.page_content for d in retrieved_docs[:10]])  # Top 10 docs only
    sentences = re.split(r'(?<=[.!?])\s+', all_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 40]

    if not sentences:
        logger.warning("  ✗ No sentences found")
        return None

    # Filter out product catalog noise
    filtered_sentences = []
    for s in sentences:
        # Skip sentences that are clearly product listings
        if re.search(r'Category:|Season:|Usage:|Color:|Price:|SKU:', s, re.IGNORECASE):
            continue
        # Skip sentences with brand names followed by product codes
        if re.search(r'(Men|Women|Kids|Boys|Girls)\s+[A-Z][a-z]+\s+[A-Z]', s):
            continue
        # Keep only advice/guidance sentences
        if any(word in s.lower() for word in ['wear', 'pair', 'choose', 'opt', 'works', 'complement', 
                                                'match', 'combine', 'style', 'look', 'consider', 'add']):
            filtered_sentences.append(s)

    if not filtered_sentences:
        # Fallback: use all sentences if filtering was too aggressive
        filtered_sentences = [s for s in sentences if len(s.split()) > 10][:15]

    # Score by relevance to query
    query_tokens = set(re.findall(r"\w+", query.lower()))
    
    scored = []
    for s in filtered_sentences:
        s_tokens = set(re.findall(r"\w+", s.lower()))
        score = len(s_tokens & query_tokens)
        # Bonus for sentence length (prefer substantial advice)
        score += min(2, len(s.split()) // 30)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 5-8 most relevant sentences
    top_sentences = [s for _, s in scored[:8] if s]

    if not top_sentences:
        return None

    # Build concise answer
    answer_parts = []
    
    # Add 3-5 best sentences with natural flow
    for i, sentence in enumerate(top_sentences[:5]):
        answer_parts.append(sentence)

    answer = " ".join(answer_parts)
    
    # Clean up any remaining noise
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    word_count = len(answer.split())
    
    # Ensure answer is substantial but not too long (100-200 words ideal)
    if word_count < 50:
        logger.warning(f"  ⚠️ Answer too short ({word_count} words)")
        return None
    
    if word_count > 250:
        # Trim to ~200 words
        words = answer.split()[:200]
        answer = " ".join(words) + "..."
        word_count = 200
    
    logger.info(f"  ✅ Smart answer ready ({word_count} words)")
    return answer


def scaffold_and_polish(query: str, retrieved_docs: List[Document], llm_client) -> Optional[str]:
    """Create a concise scaffold (approx 150-220 words) from retrieved docs,
    then ask the remote (or local) LLM to expand and polish it into a
    320-420 word expert answer. Returns None if polishing fails.
    """
    logger.info(f"🔨 Building scaffold for polish: '{query}'")
    import re

    # Reuse sentence extraction logic but stop early for a compact scaffold
    all_text = "\n\n".join([d.page_content for d in retrieved_docs[:12]])
    sentences = re.split(r'(?<=[.!?])\s+', all_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    if not sentences:
        logger.warning("  ✗ No sentences to build scaffold")
        return None

    # Score sentences by overlap with query + fashion keywords
    query_tokens = set(re.findall(r"\w+", query.lower()))
    fashion_keywords = set(["outfit","wear","wardrobe","style","colors","layer","blazer",
                            "trousers","dress","shoes","sweater","jacket","care","wool","fit",
                            "tailor","neutral","accessory","season","fall"])
    keywords = query_tokens.union(fashion_keywords)

    scored = []
    for s in sentences:
        s_tokens = set(re.findall(r"\w+", s.lower()))
        score = len(s_tokens & keywords)
        score += min(2, len(s.split()) // 30)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    scaffold_parts = []
    word_count = 0
    for _, s in scored:
        scaffold_parts.append(s)
        word_count = len(" ".join(scaffold_parts).split())
        if word_count >= 180:
            break

    scaffold = "\n\n".join(scaffold_parts).strip()
    if not scaffold:
        logger.warning("  ✗ Scaffold empty after selection")
        return None

    # Craft polish prompt - natural expansion with no limits
    polish_prompt = f"""Expand this draft into a complete, detailed fashion answer for: {query}

Draft: {scaffold}

Write a comprehensive, natural answer with practical advice and specific recommendations.

Enhanced answer:
"""

    logger.info("  → Polishing scaffold with PHI model")
    try:
        out = llm_client(
            polish_prompt,
            max_new_tokens=600,  # Allow natural expansion
            temperature=0.75,
            top_p=0.92,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=llm_client.tokenizer.eos_token_id
        )
        
        # Extract and clean the polished text
        if isinstance(out, list) and out:
            polished = out[0].get('generated_text', '') if isinstance(out[0], dict) else str(out[0])
        else:
            polished = str(out)
        
        # Remove prompt echo if present
        if polish_prompt in polished:
            polished = polished[len(polish_prompt):].strip()
        else:
            polished = polished.strip()
            
    except Exception as e:
        logger.error(f"  ✗ Polishing error: {e}")
        return None

    if not polished:
        logger.warning("  ✗ Polished output empty")
        return None

    final_words = polished.split()
    fw = len(final_words)
    
    # No artificial limits - accept natural length
    if fw < 50:
        logger.warning(f"  ✗ Polished output too short ({fw} words)")
        return None
    
    # Keep full response, no truncation
    logger.info(f"  ✅ Polished answer ready ({fw} words)")
    return polished


def retrieve_knowledge_langchain(
    query: str,
    vectorstore,
    top_k: int = 12
) -> Tuple[List[Document], float]:
    logger.info(f"🔍 Retrieving knowledge for: '{query}'")
    
    # Natural mode: use query variants for better context
    query_variants = [
        query,
        f"fashion advice clothing outfit style for {query}",
    ]
    
    all_docs = []
    
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
    
    unique_docs = {}
    for doc in all_docs:
        content_key = doc.page_content[:100]
        if content_key not in unique_docs:
            unique_docs[content_key] = doc
        else:
            if doc.metadata.get('similarity', 0) > unique_docs[content_key].metadata.get('similarity', 0):
                unique_docs[content_key] = doc
    
    final_docs = list(unique_docs.values())
    final_docs.sort(key=lambda x: x.metadata.get('similarity', 0), reverse=True)
    
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
    """Generate answer using Flan-T5 Base - instruction-tuned for Q&A."""
    if not llm_client:
        logger.error("  → Flan-T5 model not initialized")
        return None
    
    # Extract model components
    model = llm_client['model']
    tokenizer = llm_client['tokenizer']
    device = llm_client['device']
    
    # Select best documents
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored_docs = []
    for doc in retrieved_docs[:15]:
        content = doc.page_content.lower()
        doc_words = set(content.split())
        overlap = len(query_words.intersection(doc_words))
        
        if doc.metadata.get('verified', False):
            overlap += 10
        
        if len(doc.page_content) > 200:
            overlap += 3
        
        scored_docs.append((doc, overlap))
    
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc[0] for doc in scored_docs[:5]]
    
    # Build rich context (Flan-T5 can handle more context)
    context_parts = []
    for doc in top_docs:
        content = doc.page_content.strip()
        if len(content) > 300:
            content = content[:300] + "..."
        context_parts.append(content)
    
    context_text = "\n\n".join(context_parts)
    
    # Flan-T5 instruction prompt - direct and clear
    prompt = f"""Answer this fashion question with specific, practical advice (150-200 words):

Question: {query}

Fashion Knowledge:
{context_text[:600]}

Provide detailed fashion advice:"""

    try:
        logger.info(f"  → Generating with Flan-T5 (target: 200 words)")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with Flan-T5 optimized parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,  # ~200 words
                min_length=120,      # Ensure substantial answers
                temperature=0.8,     # Balanced creativity
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=False
            )
        
        # Decode output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        word_count = len(response.split())
        logger.info(f"  ✅ Generated {word_count} words with Flan-T5")
        
        # Validate quality
        if word_count < 50:
            logger.warning(f"  ⚠️ Response too short ({word_count} words)")
            return None
        
        # Check for generic/irrelevant content
        if any(phrase in response.lower() for phrase in ["i cannot", "i can't", "i'm sorry", "as an ai"]):
            logger.warning("  ⚠️ Generic response detected")
            return None
        
        return response
        
    except Exception as e:
        logger.error(f"  ✗ Flan-T5 generation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def generate_answer_langchain(
    query: str,
    vectorstore,
    llm_client
) -> str:
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing query: '{query}'")
    logger.info(f"{'='*80}")
    
    retrieved_docs, confidence = retrieve_knowledge_langchain(
        query,
        vectorstore,
        top_k=CONFIG["top_k"]
    )
    
    if not retrieved_docs:
        return "I couldn't find relevant information to answer your question."
    
    # Try Flan-T5 first (instruction-tuned, high quality)
    logger.info("  → Attempting Flan-T5 generation (primary method)")
    try:
        llm_answer = generate_llm_answer(query, retrieved_docs, llm_client, attempt=1)
        if llm_answer:
            logger.info(f"  ✅ Flan-T5 answer generated successfully")
            return llm_answer
    except Exception as e:
        logger.error(f"  ✗ Flan-T5 error: {e}")
    
    # Fallback to extractive if Flan-T5 fails
    logger.info("  → Fallback: Using extractive answer generator")
    try:
        extractive_answer = generate_extractive_answer(query, retrieved_docs)
        if extractive_answer:
            logger.info(f"  ✅ Extractive answer generated successfully")
            return extractive_answer
    except Exception as e:
        logger.error(f"  ✗ Extractive answer error: {e}")
    
    return "I apologize, but I'm having trouble generating a response. Please try rephrasing your question or ask something else."

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def fashion_chatbot(message: str, history: List[List[str]]):
    try:
        if not message or not message.strip():
            yield "Please ask a fashion-related question!"
            return
        
        yield "🔍 Searching fashion knowledge..."
        
        retrieved_docs, confidence = retrieve_knowledge_langchain(
            message.strip(),
            vectorstore,
            top_k=CONFIG["top_k"]
        )
        
        if not retrieved_docs:
            yield "I couldn't find relevant information to answer your question."
            return
        
        yield f"💭 Generating fashion advice ({len(retrieved_docs)} sources found)..."
        
        # Try Flan-T5 first (fast and high quality)
        logger.info("  → Generating with Flan-T5")
        llm_answer = generate_llm_answer(message.strip(), retrieved_docs, llm_client, attempt=1)
        
        # Fallback to extractive if needed
        if not llm_answer:
            logger.info("  → Fallback: Using extractive answer")
            llm_answer = generate_extractive_answer(message.strip(), retrieved_docs)
        
        if not llm_answer:
            logger.error(f"  ✗ All generation methods failed")
            yield "I apologize, but I'm having trouble generating a response. Please try rephrasing your question."
            return
        
        import time
        words = llm_answer.split()
        displayed_text = ""
        
        # Faster streaming for better UX
        for i, word in enumerate(words):
            displayed_text += word + " "
            
            if i % 5 == 0 or i == len(words) - 1:
                yield displayed_text.strip()
                time.sleep(0.02)  # Reduced delay
        
    except Exception as e:
        logger.error(f"Error in chatbot: {e}")
        yield f"Sorry, I encountered an error: {str(e)}"

# ============================================================================
# INITIALIZE AND LAUNCH
# ============================================================================

llm_client = None
embeddings = None
vectorstore = None

def startup():
    global llm_client, embeddings, vectorstore
    
    logger.info("🚀 Starting Fashion Advisor RAG...")
    
    embeddings = initialize_embeddings()
    vectorstore = load_vector_store(embeddings)
    llm_client = initialize_llm()
    
    logger.info("✅ All components initialized successfully!")

startup()

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
)

if __name__ == "__main__":
    demo.launch()
