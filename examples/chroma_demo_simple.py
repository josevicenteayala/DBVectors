"""
ChromaDB Simple Demo (Without External Model Downloads)

This is a simplified version of the demo that uses ChromaDB's default embedding function
instead of downloading external models. This makes it easier to test in restricted environments.

For the full demo with sentence-transformers, see chroma_demo.py
"""

import chromadb
from chromadb.utils import embedding_functions


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    """Main function demonstrating ChromaDB workflow with default embeddings."""
    
    # ========================================================================
    # STEP 1: Connect and Create a Collection
    # ========================================================================
    print_section("STEP 1: Connect to ChromaDB and Create a Collection")
    
    # Create a ChromaDB client (using in-memory mode for demo)
    client = chromadb.Client()
    print("✓ ChromaDB client created successfully")
    
    # Create a collection with default embedding function
    collection_name = "demo_collection"
    
    # Delete collection if it exists
    try:
        client.delete_collection(name=collection_name)
        print(f"✓ Deleted existing collection '{collection_name}'")
    except:
        pass
    
    # Use ChromaDB's default embedding function (no external downloads needed)
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    collection = client.create_collection(
        name=collection_name,
        embedding_function=default_ef,
        metadata={"description": "A demo collection for semantic search"}
    )
    print(f"✓ Created collection: '{collection_name}'")
    print(f"✓ Using default embedding function")
    
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
    print(f"✓ Embeddings were generated automatically by ChromaDB")
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
   
   Note: These thresholds depend on your embedding function and use case.
    """)
    
    print_section("Demo Complete! 🎉")
    print("""
This demo used ChromaDB's default embedding function for simplicity.

For production use, consider:
1. Using sentence-transformers for better embeddings (see chroma_demo.py)
2. PersistentClient for data persistence across runs
3. Custom embedding functions for domain-specific needs

To use persistent storage:
    client = chromadb.PersistentClient(path="./chroma_db")
    """)


if __name__ == "__main__":
    main()
