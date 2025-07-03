#!/usr/bin/env python3
"""
Train RAG System with MIE Dataset
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.config.settings import load_config
from rag.vectorstore.embedding_store import MIEVectorStore

def main():
    """Train the RAG system with MIE data"""
    
    print("🚀 Training RAG System with MIE Dataset")
    
    # Load configuration
    config = load_config()
    
    # Initialize vector store
    vector_store = MIEVectorStore(config)
    
    # Train with MIE dataset
    data_path = Path(config["data"]["raw_dir"]) / config["data"]["dataset_file"]
    
    if data_path.exists():
        print(f"📊 Loading data from: {data_path}")
        vector_store.train_with_mie_data(str(data_path))
        print("✅ RAG training completed!")
    else:
        print(f"❌ Data file not found: {data_path}")
        return False
    
    # Test retrieval
    print("\n🧪 Testing RAG retrieval...")
    test_query = "military attack between countries"
    results = vector_store.search(test_query, top_k=3)
    
    print(f"Query: '{test_query}'")
    print(f"Found {len(results)} similar articles:")
    
    for i, result in enumerate(results, 1):
        metadata = result["metadata"]
        print(f"  {i}. {metadata.get('title', 'No title')} (MIE: {metadata.get('label', 'Unknown')})")
        print(f"     Similarity: {result['similarity']:.3f}")
    
    return True

if __name__ == "__main__":
    main() 