"""
ChromaDB Offline Demo (No Internet Required)

This version demonstrates the complete ChromaDB workflow without requiring internet access.
It uses a simple custom embedding function that works offline.

For production use with real embeddings, see:
- chroma_demo.py (with sentence-transformers)
- chroma_demo_simple.py (with ChromaDB's default function)
"""

import chromadb
from chromadb.api.types import EmbeddingFunction
import hashlib
from typing import List


class SimpleEmbeddingFunction(EmbeddingFunction):
    """
    A simple embedding function that works offline.
    
    This creates deterministic embeddings based on text hashing.
    NOT suitable for production - use sentence-transformers instead.
    """
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate simple embeddings from text."""
        embeddings = []
        for text in input:
            # Create a simple 384-dimensional embedding
            # based on the text's hash and character properties
            embedding = self._text_to_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def _text_to_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Convert text to a simple embedding vector."""
        # Use hash to create deterministic but distributed values
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Create embedding from hash and text properties
        embedding = []
        for i in range(dim):
            # Mix hash with position and text properties for variation
            seed = int(text_hash[i % len(text_hash)], 16)
            char_sum = sum(ord(c) for c in text) if text else 0
            value = (seed + i + char_sum) / (16 * dim)
            # Normalize to roughly -1 to 1 range
            value = (value - 0.5) * 2
            embedding.append(value)
        
        return embedding


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    """Main function demonstrating ChromaDB workflow offline."""
    
    print("=" * 70)
    print(" ChromaDB Complete Workflow Demo (Offline Mode)")
    print("=" * 70)
    print("\nNote: This demo uses a simple embedding function for offline testing.")
    print("For production use, install and use sentence-transformers.")
    
    # ========================================================================
    # STEP 1: Connect and Create a Collection
    # ========================================================================
    print_section("STEP 1: Connect to ChromaDB and Create a Collection")
    
    # Create a ChromaDB client (using in-memory mode for demo)
    client = chromadb.Client()
    print("✓ ChromaDB client created successfully")
    
    # Create a collection with custom embedding function
    collection_name = "demo_collection"
    
    # Delete collection if it exists
    try:
        client.delete_collection(name=collection_name)
        print(f"✓ Deleted existing collection '{collection_name}'")
    except:
        pass
    
    # Use our simple offline embedding function
    embedding_fn = SimpleEmbeddingFunction()
    
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"description": "A demo collection for semantic search"}
    )
    print(f"✓ Created collection: '{collection_name}'")
    print(f"✓ Using custom offline embedding function (384 dimensions)")
    
    # ========================================================================
    # STEP 2: Prepare Text Dataset
    # ========================================================================
    print_section("STEP 2: Prepare Text Dataset")
    
    # Sample dataset: Scientific facts and information
    documents = [
        "The Earth revolves around the Sun in an elliptical orbit.",
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning is a subset of artificial intelligence.",
        "The Pacific Ocean is the largest and deepest ocean on Earth.",
        "DNA carries genetic information in living organisms.",
        "The speed of light in vacuum is approximately 299,792 kilometers per second.",
        "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
        "The human brain contains approximately 86 billion neurons.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "The Great Wall of China is visible from low Earth orbit."
    ]
    
    # Metadata for each document
    metadatas = [
        {"category": "astronomy", "topic": "solar system"},
        {"category": "programming", "topic": "languages"},
        {"category": "technology", "topic": "AI"},
        {"category": "geography", "topic": "oceans"},
        {"category": "biology", "topic": "genetics"},
        {"category": "physics", "topic": "constants"},
        {"category": "literature", "topic": "authors"},
        {"category": "biology", "topic": "neuroscience"},
        {"category": "biology", "topic": "plants"},
        {"category": "geography", "topic": "landmarks"}
    ]
    
    # Generate unique IDs for each document
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    print(f"✓ Prepared {len(documents)} documents")
    print(f"  Sample document: '{documents[0]}'")
    
    # ========================================================================
    # STEP 3: Populate the Database
    # ========================================================================
    print_section("STEP 3: Populate the Database with Documents and Metadata")
    
    # Add documents to collection (embeddings are generated automatically)
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✓ Added {len(documents)} documents to the collection")
    print(f"✓ Embeddings were generated automatically")
    print(f"  Collection now contains {collection.count()} items")
    
    # ========================================================================
    # STEP 4: Run Semantic Searches
    # ========================================================================
    print_section("STEP 4: Run Semantic Searches")
    
    # Example queries demonstrating semantic search capabilities
    queries = [
        "How do plants create energy?",
        "Tell me about computer programming",
        "What is the largest body of water?"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 70)
        
        # Perform semantic search (query is embedded automatically)
        results = collection.query(
            query_texts=[query],
            n_results=3,  # Get top 3 most relevant results
            include=["documents", "metadatas", "distances"]
        )
        
        # ====================================================================
        # STEP 5: Retrieve and Interpret the Results
        # ====================================================================
        print("\n📊 Search Results (ranked by relevance):")
        
        # Iterate through results and display them
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            # Calculate similarity score
            # Lower distance = higher similarity
            similarity = 1 / (1 + distance)
            
            print(f"\n  Rank {i}:")
            print(f"  📄 Document: {doc}")
            print(f"  🏷️  Category: {metadata.get('category', 'N/A')}")
            print(f"  🎯 Topic: {metadata.get('topic', 'N/A')}")
            print(f"  📏 Distance: {distance:.4f}")
            print(f"  ⭐ Similarity Score: {similarity:.4f}")
            
            # Interpret the relevance
            if similarity > 0.7:
                relevance = "HIGHLY RELEVANT"
                emoji = "🟢"
            elif similarity > 0.5:
                relevance = "MODERATELY RELEVANT"
                emoji = "🟡"
            else:
                relevance = "SOMEWHAT RELEVANT"
                emoji = "🟠"
            
            print(f"  {emoji} Relevance: {relevance}")
    
    # ========================================================================
    # Filtering by Metadata
    # ========================================================================
    print_section("BONUS: Filtering Results by Metadata")
    
    query = "What scientific facts do you know?"
    print(f"\n🔍 Query: '{query}'")
    print(f"🔧 Filter: Only documents in 'biology' category")
    print("-" * 70)
    
    # Search with metadata filter
    filtered_results = collection.query(
        query_texts=[query],
        n_results=3,
        where={"category": "biology"},  # Only return biology documents
        include=["documents", "metadatas", "distances"]
    )
    
    print("\n📊 Filtered Search Results:")
    for i, (doc, metadata) in enumerate(zip(
        filtered_results['documents'][0],
        filtered_results['metadatas'][0]
    ), 1):
        print(f"\n  {i}. {doc}")
        print(f"     Category: {metadata['category']}, Topic: {metadata['topic']}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_section("How to Interpret Results")
    
    print("""
🎓 Understanding Vector Database Results:

1. **Distance/Similarity**:
   - Lower distance = More similar to the query
   - ChromaDB uses L2 (Euclidean) distance by default
   - Similarity score (0-1): Higher is better

2. **Ranking**:
   - Results are automatically sorted by relevance
   - The first result is the most semantically similar
   - Even the last result may be relevant (check similarity score)

3. **Semantic Search vs. Keyword Search**:
   - Finds conceptually related content, not just matching words
   - With proper embeddings, query "How do plants create energy?" 
     finds "Photosynthesis..." even without matching words
   - Works based on semantic meaning, not string matching

4. **Metadata Filtering**:
   - Combine semantic search with structured filters
   - Example: Find similar documents in a specific category
   - Useful for narrowing down results in large databases

5. **About This Demo's Embeddings**:
   - This demo uses simple hash-based embeddings for offline testing
   - These are NOT as accurate as real ML-based embeddings
   - For production, use sentence-transformers or OpenAI embeddings
   
6. **Production Recommendations**:
   - Use sentence-transformers models (e.g., all-MiniLM-L6-v2)
   - Or use API-based embeddings (OpenAI, Cohere, etc.)
   - Evaluate embedding quality for your specific use case
    """)
    
    print_section("Demo Complete! 🎉")
    print("""
This demo shows the complete ChromaDB workflow:
✓ Collection creation
✓ Document embedding and storage
✓ Semantic search queries
✓ Result interpretation
✓ Metadata filtering

For production use:
1. Replace SimpleEmbeddingFunction with sentence-transformers
2. Use PersistentClient for data persistence
3. Implement proper error handling and logging

Example with persistence:
    client = chromadb.PersistentClient(path="./chroma_db")

See chroma_demo.py for the full implementation with real embeddings.
    """)


if __name__ == "__main__":
    main()
