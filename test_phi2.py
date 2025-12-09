"""
Quick test script to verify PHI-2 model generation works properly.
Run this before deploying to Hugging Face Spaces to test locally.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_phi2_generation():
    """Test PHI-2 model with the same configuration as app.py"""
    
    model_name = "microsoft/phi-2"
    logger.info(f"Loading {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Tokenizer loaded: vocab_size={len(tokenizer)}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32
    )
    
    model.eval()
    logger.info("Model loaded successfully")
    
    # Create pipeline
    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        batch_size=1,
        clean_up_tokenization_spaces=True
    )
    
    logger.info("Pipeline created")
    
    # Test generation with PHI-2 optimized format
    test_prompt = """You are a fashion expert assistant.

Question: What are the best colors for a winter wardrobe?

Context: Winter fashion typically features darker, richer colors. Navy blue, charcoal gray, and deep burgundy are excellent choices. Neutrals like camel and cream work well for layering.

Provide a helpful, detailed fashion answer with practical advice and specific recommendations."""
    
    # Format for PHI-2
    formatted_prompt = f"Instruct: {test_prompt}\nOutput:"
    
    logger.info("\n" + "="*60)
    logger.info("Testing PHI-2 generation...")
    logger.info("="*60)
    logger.info(f"Prompt: {test_prompt[:100]}...")
    
    # Generate
    output = llm(
        formatted_prompt,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        return_full_text=False
    )
    
    # Extract text
    if output and isinstance(output, list) and len(output) > 0:
        generated = output[0].get('generated_text', '')
        
        # Clean up
        if formatted_prompt in generated:
            generated = generated.replace(formatted_prompt, '').strip()
        
        for prefix in ['Output:', 'Answer:', 'Response:', 'Instruct:']:
            if generated.startswith(prefix):
                generated = generated[len(prefix):].strip()
                break
        
        word_count = len(generated.split())
        
        logger.info("\n" + "="*60)
        logger.info("GENERATION RESULT")
        logger.info("="*60)
        logger.info(f"Generated {word_count} words ({len(generated)} chars)")
        logger.info("\n--- Generated Text ---")
        logger.info(generated)
        logger.info("="*60)
        
        if word_count >= 20:
            logger.info("\n✅ SUCCESS: PHI-2 is generating properly!")
            return True
        else:
            logger.warning("\n⚠️ WARNING: Output too short, may indicate an issue")
            return False
    else:
        logger.error("\n❌ FAILED: No output from model")
        return False

if __name__ == "__main__":
    try:
        success = test_phi2_generation()
        if success:
            print("\n✅ PHI-2 test passed! Model should work on Hugging Face Spaces.")
            print("   (Note: Generation will be slower on Spaces' CPU)")
        else:
            print("\n⚠️ PHI-2 test had issues. Check the logs above.")
            print("   You may need to try a different model or configuration.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("   Make sure you have: pip install transformers torch accelerate")
