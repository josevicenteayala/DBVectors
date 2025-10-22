"""
ChromaDB Complete Workflow Demo

This script demonstrates a complete workflow for using ChromaDB as a vector database:
1. Connect and create a collection
2. Embed a text dataset using sentence-transformers
3. Populate the database with vectors and metadata
4. Embed a query and run a semantic search
5. Retrieve and interpret the text results

Author: DBVectors Demo
Date: 2025
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import time


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    """Main function demonstrating the complete ChromaDB workflow."""
    
    # ========================================================================
    # STEP 1: Connect and Create a Collection
    # ========================================================================
    print_section("STEP 1: Connect to ChromaDB and Create a Collection")
    
    # Create a ChromaDB client (using in-memory mode for demo)
    # For persistent storage, use: client = chromadb.PersistentClient(path="./chroma_db")
    client = chromadb.Client()
    print("✓ ChromaDB client created successfully")
    
    # Create a collection for storing our documents and their embeddings
    # Collections are like tables in traditional databases
    collection_name = "demo_collection"
    
    # Delete collection if it already exists (for demo purposes)
    try:
        client.delete_collection(name=collection_name)
        print(f"✓ Deleted existing collection '{collection_name}'")
    except:
        pass
    
    # Create a new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "A demo collection for semantic search"}
    )
    print(f"✓ Created collection: '{collection_name}'")
    
    # ========================================================================
    # STEP 2: Prepare and Embed a Text Dataset
    # ========================================================================
    print_section("STEP 2: Embed a Text Dataset Using sentence-transformers")
    
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
    
    # Metadata for each document (can include source, category, timestamp, etc.)
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
    
    print(f"✓ Prepared {len(documents)} documents for embedding")
    print(f"  Sample document: '{documents[0]}'")
    
    # Load a pre-trained sentence transformer model
    # This model converts text into 384-dimensional vectors
    print("\n⏳ Loading sentence-transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded: all-MiniLM-L6-v2 (384-dimensional embeddings)")
    
    # Generate embeddings for all documents
    print("\n⏳ Generating embeddings for all documents...")
    start_time = time.time()
    embeddings = model.encode(documents, show_progress_bar=True)
    elapsed_time = time.time() - start_time
    
    print(f"✓ Generated {len(embeddings)} embeddings in {elapsed_time:.2f} seconds")
    print(f"  Embedding dimension: {embeddings[0].shape[0]}")
    print(f"  Sample embedding (first 5 values): {embeddings[0][:5]}")
    
    # ========================================================================
    # STEP 3: Populate the Database with Vectors and Metadata
    # ========================================================================
    print_section("STEP 3: Populate the Database with Vectors and Metadata")
    
    # Add documents, embeddings, and metadata to the collection
    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),  # Convert numpy array to list
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✓ Added {len(documents)} documents to the collection")
    print(f"  Collection now contains {collection.count()} items")
    
    # ========================================================================
    # STEP 4: Embed a Query and Run a Semantic Search
    # ========================================================================
    print_section("STEP 4: Embed a Query and Run a Semantic Search")
    
    # Example queries demonstrating semantic search capabilities
    queries = [
        "How do plants create energy?",
        "Tell me about computer programming",
        "What is the largest body of water?"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 70)
        
        # Generate embedding for the query
        query_embedding = model.encode([query])
        
        # Perform semantic search
        # n_results: number of top results to return
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
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
            # Calculate similarity score (1 - distance for L2 distance)
            # Lower distance = higher similarity
            similarity = 1 / (1 + distance)  # Normalize to 0-1 range
            
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
    # Additional Features: Filtering by Metadata
    # ========================================================================
    print_section("BONUS: Filtering Results by Metadata")
    
    query = "What scientific facts do you know?"
    print(f"\n🔍 Query: '{query}'")
    print(f"🔧 Filter: Only documents in 'biology' category")
    print("-" * 70)
    
    query_embedding = model.encode([query])
    
    # Search with metadata filter
    filtered_results = collection.query(
        query_embeddings=query_embedding.tolist(),
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
    # Summary and Explanation
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
   - Query: "How do plants create energy?" → Finds "Photosynthesis..."
   - Works even when exact words don't match

4. **Metadata Filtering**:
   - Combine semantic search with structured filters
   - Example: Find similar documents in a specific category
   - Useful for narrowing down results in large databases

5. **Best Practices**:
   - Similarity > 0.7: Highly relevant, likely what you're looking for
   - Similarity 0.5-0.7: Moderately relevant, may be useful
   - Similarity < 0.5: Less relevant, might be off-topic
   
   Note: These thresholds depend on your model and use case.
   Experiment to find what works best for your application.

6. **Use Cases**:
   - Question answering systems
   - Document retrieval and search
   - Recommendation systems
   - Content deduplication
   - Semantic clustering
    """)
    
    print_section("Demo Complete! 🎉")
    print("""
Next steps:
1. Modify the documents array with your own data
2. Try different queries to test semantic search
3. Experiment with metadata filters
4. Use PersistentClient for data persistence across runs
5. Explore advanced features like custom embedding functions

For persistent storage, replace the Client() with:
    client = chromadb.PersistentClient(path="./chroma_db")
    """)


if __name__ == "__main__":
    main()
