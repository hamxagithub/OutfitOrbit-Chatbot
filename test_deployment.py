"""
Local Testing Script for Fashion Advisor RAG
Run this before deploying to Hugging Face to verify everything works
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a required file exists"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        size_mb = size / (1024 * 1024)
        print(f"✅ {description}: {filepath} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """Check if a required directory exists"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        files = list(Path(dirpath).iterdir())
        print(f"✅ {description}: {dirpath} ({len(files)} files)")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {dirpath}")
        return False

def check_python_imports():
    """Check if all required packages are installed"""
    print("\n🔍 Checking Python packages...")
    
    required_packages = [
        ("gradio", "gradio"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("sentence_transformers", "sentence-transformers"),
        ("langchain", "langchain"),
        ("langchain_community", "langchain-community"),
        ("faiss", "faiss-cpu"),
    ]
    
    all_installed = True
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print(f"✅ {package_name} installed")
        except ImportError:
            print(f"❌ {package_name} NOT INSTALLED - run: pip install {package_name}")
            all_installed = False
    
    return all_installed

def check_vector_store():
    """Check if FAISS vector store is properly set up"""
    print("\n🔍 Checking FAISS vector store...")
    
    vector_store_path = "./faiss_vectorstore"
    
    if not os.path.exists(vector_store_path):
        print(f"❌ Vector store directory not found: {vector_store_path}")
        print("\n📝 To create vector store:")
        print("   1. Run your notebook cell that creates the FAISS index")
        print("   2. Look for: vectorstore.save_local('faiss_vectorstore')")
        print("   3. Copy the generated folder to this directory")
        return False
    
    required_files = [
        ("index.faiss", "FAISS index file"),
        ("index.pkl", "Document metadata"),
    ]
    
    all_found = True
    for filename, description in required_files:
        filepath = os.path.join(vector_store_path, filename)
        if not check_file_exists(filepath, description):
            all_found = False
    
    return all_found

def test_import_app():
    """Try to import the app.py file"""
    print("\n🔍 Testing app.py import...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Try to import (will fail if syntax errors)
        import app
        print("✅ app.py imports successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing app.py: {e}")
        return False

def main():
    """Run all checks"""
    print("="*80)
    print("🧪 Fashion Advisor RAG - Pre-Deployment Checks")
    print("="*80)
    
    # Check required files
    print("\n📁 Checking required files...")
    files_ok = all([
        check_file_exists("app.py", "Main application"),
        check_file_exists("requirements.txt", "Dependencies file"),
        check_file_exists("README.md", "Documentation"),
    ])
    
    # Check vector store
    vector_store_ok = check_vector_store()
    
    # Check Python packages
    packages_ok = check_python_imports()
    
    # Test app import
    import_ok = test_import_app()
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    checks = {
        "Required files": files_ok,
        "FAISS vector store": vector_store_ok,
        "Python packages": packages_ok,
        "App import": import_ok,
    }
    
    all_passed = all(checks.values())
    
    for check_name, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {check_name}: {'PASS' if status else 'FAIL'}")
    
    print("\n" + "="*80)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("\n✅ Ready for deployment to Hugging Face Spaces")
        print("\n📝 Next steps:")
        print("   1. Create a new Space on Hugging Face")
        print("   2. Upload all files (app.py, requirements.txt, README.md)")
        print("   3. Upload the faiss_vectorstore/ folder")
        print("   4. Wait for automatic build and deployment")
        print("\n📖 See DEPLOYMENT_GUIDE.md for detailed instructions")
    else:
        print("⚠️ SOME CHECKS FAILED")
        print("\n❌ Please fix the issues above before deploying")
        print("\n📝 Common fixes:")
        if not vector_store_ok:
            print("   • Run notebook to generate FAISS vector store")
            print("   • Copy faiss_vectorstore/ folder to current directory")
        if not packages_ok:
            print("   • Install missing packages: pip install -r requirements.txt")
        if not import_ok:
            print("   • Check app.py for syntax errors")
            print("   • Ensure all imports are available")
    
    print("="*80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
