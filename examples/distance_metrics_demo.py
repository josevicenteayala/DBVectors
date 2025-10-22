"""
ChromaDB Distance Metrics Demonstration

This script demonstrates how different distance metrics (L2, Cosine, Inner Product)
affect vector similarity search results in ChromaDB.

It shows:
1. How each metric calculates similarity
2. Different results for the same query with different metrics
3. When to use each metric
4. Visual comparison of results

Author: DBVectors Demo
Date: 2025
"""

import chromadb
import numpy as np
from typing import List, Tuple


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def calculate_metrics_manually(vec1: np.ndarray, vec2: np.ndarray) -> dict:
    """Calculate all three distance metrics manually for educational purposes."""
    # L2 (Euclidean) Distance
    l2_distance = np.sqrt(np.sum((vec1 - vec2) ** 2))
    
    # Cosine Distance
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    cosine_distance = 1 - cosine_similarity
    
    # Inner Product (dot product)
    inner_product = dot_product
    
    return {
        "l2": l2_distance,
        "cosine_distance": cosine_distance,
        "cosine_similarity": cosine_similarity,
        "inner_product": inner_product
    }


def demo_basic_vectors():
    """Demonstrate metrics with simple 3D vectors."""
    print_section("PART 1: Understanding Distance Metrics with Simple Vectors")
    
    # Define simple test vectors
    vectors = {
        "A": np.array([1.0, 0.0, 0.0]),  # Unit vector in X direction
        "B": np.array([2.0, 0.0, 0.0]),  # Same direction, double magnitude
        "C": np.array([0.0, 1.0, 0.0]),  # Unit vector in Y direction
        "D": np.array([1.0, 1.0, 0.0]),  # 45-degree angle
    }
    
    print("\nTest Vectors:")
    for name, vec in vectors.items():
        print(f"  Vector {name}: {vec}")
    
    # Calculate metrics for Vector A vs all others
    print("\n" + "-" * 80)
    print("Comparing Vector A [1, 0, 0] with all vectors:")
    print("-" * 80)
    
    query = vectors["A"]
    
    print(f"\n{'Vector':<10} {'L2 Dist':<12} {'Cosine Dist':<15} {'Cosine Sim':<15} {'Inner Prod':<12}")
    print("-" * 80)
    
    for name, vec in vectors.items():
        metrics = calculate_metrics_manually(query, vec)
        print(f"{name:<10} {metrics['l2']:<12.4f} {metrics['cosine_distance']:<15.4f} "
              f"{metrics['cosine_similarity']:<15.4f} {metrics['inner_product']:<12.4f}")
    
    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS:")
    print("=" * 80)
    print("""
1. L2 DISTANCE:
   - A vs A: 0.0000 (identical)
   - A vs B: 1.0000 (same direction, but different magnitude affects distance)
   - A vs C: 1.4142 (perpendicular, sqrt(2))
   - A vs D: 1.0000 (45-degree angle)
   
2. COSINE DISTANCE:
   - A vs A: 0.0000 (identical)
   - A vs B: 0.0000 (same direction, magnitude ignored! ⭐)
   - A vs C: 1.0000 (perpendicular, maximum distance)
   - A vs D: 0.2929 (45-degree angle)
   
3. INNER PRODUCT:
   - A vs A: 1.0000 (dot product)
   - A vs B: 2.0000 (larger magnitude gives higher value ⭐)
   - A vs C: 0.0000 (perpendicular, no alignment)
   - A vs D: 1.0000 (partial alignment)
    """)


def demo_chromadb_comparison():
    """Demonstrate how ChromaDB returns different results with different metrics."""
    print_section("PART 2: ChromaDB Search Results with Different Distance Metrics")
    
    # Create ChromaDB client
    client = chromadb.Client()
    
    # Create three collections with different metrics
    collections = {
        "L2": client.create_collection(name="demo_l2", metadata={"hnsw:space": "l2"}),
        "Cosine": client.create_collection(name="demo_cosine", metadata={"hnsw:space": "cosine"}),
        "Inner Product": client.create_collection(name="demo_ip", metadata={"hnsw:space": "ip"})
    }
    
    # Define test vectors with descriptions
    test_vectors = [
        ([1.0, 0.0, 0.0], "Vector A: Unit vector in X direction"),
        ([2.0, 0.0, 0.0], "Vector B: Same direction as A, double magnitude"),
        ([0.0, 1.0, 0.0], "Vector C: Unit vector in Y direction"),
        ([1.0, 1.0, 0.0], "Vector D: 45-degree angle from X axis"),
        ([0.5, 0.5, 0.0], "Vector E: Same direction as D, half magnitude"),
    ]
    
    # Add vectors to all collections
    for name, collection in collections.items():
        collection.add(
            embeddings=[vec for vec, _ in test_vectors],
            documents=[desc for _, desc in test_vectors],
            ids=[f"vec_{i}" for i in range(len(test_vectors))]
        )
    
    # Query vector: [1.0, 0.0, 0.0]
    query_vector = [[1.0, 0.0, 0.0]]
    
    print("\n🔍 Query Vector: [1.0, 0.0, 0.0] (same as Vector A)")
    print("\nSearching for top 5 most similar vectors in each collection...")
    
    # Query each collection
    for metric_name, collection in collections.items():
        print(f"\n{'─' * 80}")
        print(f"Results using {metric_name.upper()} metric:")
        print('─' * 80)
        
        results = collection.query(
            query_embeddings=query_vector,
            n_results=5,
            include=["documents", "distances", "embeddings"]
        )
        
        for rank, (doc, distance, embedding) in enumerate(zip(
            results['documents'][0],
            results['distances'][0],
            results['embeddings'][0]
        ), 1):
            print(f"\n  Rank {rank}:")
            print(f"  {doc}")
            print(f"  Vector: {embedding}")
            print(f"  Distance: {distance:.4f}")
            
            # Add interpretation
            if metric_name == "L2":
                print(f"  → Lower distance = more similar")
            elif metric_name == "Cosine":
                similarity = 1 - distance
                print(f"  → Cosine similarity: {similarity:.4f} (higher = more similar)")
            elif metric_name == "Inner Product":
                print(f"  → Higher inner product = more similar (ChromaDB stores negative)")


def demo_text_embeddings():
    """Demonstrate distance metrics with realistic text embeddings."""
    print_section("PART 3: Real-World Example with Text Embeddings")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("\n⏳ Loading sentence-transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Model loaded successfully\n")
        
        # Sample documents
        documents = [
            "Python is a programming language",
            "Java is a programming language",
            "The sky is blue",
            "Dogs are loyal animals",
            "Programming languages are used for software development"
        ]
        
        print("Sample Documents:")
        for i, doc in enumerate(documents, 1):
            print(f"  {i}. {doc}")
        
        # Generate embeddings
        embeddings = model.encode(documents)
        print(f"\n✓ Generated embeddings with {embeddings.shape[1]} dimensions")
        
        # Create collections
        client = chromadb.Client()
        
        collection_l2 = client.create_collection(name="text_l2", metadata={"hnsw:space": "l2"})
        collection_cosine = client.create_collection(
            name="text_cosine", 
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add documents
        for collection in [collection_l2, collection_cosine]:
            collection.add(
                documents=documents,
                embeddings=embeddings.tolist(),
                ids=[f"doc_{i}" for i in range(len(documents))]
            )
        
        # Test query
        query = "What are coding languages?"
        print(f"\n{'═' * 80}")
        print(f"🔍 Query: '{query}'")
        print('═' * 80)
        
        query_embedding = model.encode([query])
        
        # Compare results
        for metric_name, collection in [("L2", collection_l2), ("Cosine", collection_cosine)]:
            print(f"\n{metric_name} Distance Results:")
            print("-" * 80)
            
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=3,
                include=["documents", "distances"]
            )
            
            for rank, (doc, distance) in enumerate(zip(
                results['documents'][0],
                results['distances'][0]
            ), 1):
                if metric_name == "Cosine":
                    similarity = 1 - distance
                    print(f"  {rank}. {doc}")
                    print(f"     Distance: {distance:.4f} | Similarity: {similarity:.4f}")
                else:
                    print(f"  {rank}. {doc}")
                    print(f"     Distance: {distance:.4f}")
        
        print("\n" + "=" * 80)
        print("ANALYSIS:")
        print("=" * 80)
        print("""
For text embeddings from sentence-transformers:
- Both metrics identify programming-related documents as most relevant
- Cosine is typically preferred for text because:
  • It focuses on semantic direction/meaning
  • It's invariant to document length
  • It works well with normalized embeddings
- L2 can be affected by embedding magnitude variations
        """)
        
    except ImportError:
        print("\n⚠️  sentence-transformers not available.")
        print("Install with: pip install sentence-transformers")
        print("Skipping text embedding demonstration.")


def demo_normalized_vectors():
    """Demonstrate why Inner Product works well with normalized vectors."""
    print_section("PART 4: Inner Product with Normalized Vectors")
    
    print("\nInner Product is equivalent to Cosine Similarity for normalized vectors!\n")
    
    # Create some test vectors
    vectors_raw = [
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 4.0, 6.0]),  # Same direction, different magnitude
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    
    # Normalize vectors
    vectors_normalized = [v / np.linalg.norm(v) for v in vectors_raw]
    
    query_raw = np.array([1.0, 2.0, 3.0])
    query_normalized = query_raw / np.linalg.norm(query_raw)
    
    print("Raw Vectors vs Normalized Vectors:")
    print("-" * 80)
    print(f"{'Vector':<15} {'Raw':<30} {'Normalized':<30}")
    print("-" * 80)
    
    for i, (raw, norm) in enumerate(zip(vectors_raw, vectors_normalized)):
        print(f"Vector {i}:{'':<7} {str(raw):<30} {str(norm)}")
    
    print("\nQuery Vector:")
    print(f"  Raw:        {query_raw}")
    print(f"  Normalized: {query_normalized}")
    
    # Compare metrics
    print("\n" + "=" * 80)
    print("Comparison: Inner Product vs Cosine Similarity")
    print("=" * 80)
    
    print(f"\n{'Vector':<10} {'IP (raw)':<15} {'IP (norm)':<15} {'Cosine Sim':<15} {'IP = Cosine?':<15}")
    print("-" * 80)
    
    for i, (raw, norm) in enumerate(zip(vectors_raw, vectors_normalized)):
        ip_raw = np.dot(query_raw, raw)
        ip_norm = np.dot(query_normalized, norm)
        
        # Calculate cosine similarity
        cosine_sim = np.dot(query_raw, raw) / (np.linalg.norm(query_raw) * np.linalg.norm(raw))
        
        match = "✓ YES" if abs(ip_norm - cosine_sim) < 0.0001 else "✗ NO"
        
        print(f"Vector {i}:{'':>3} {ip_raw:<15.4f} {ip_norm:<15.4f} {cosine_sim:<15.4f} {match:<15}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("""
When vectors are normalized (unit length):
✓ Inner Product = Cosine Similarity
✓ Inner Product is FASTER (one less normalization step)
✓ Use Inner Product (IP) metric for pre-normalized embeddings

Many embedding models output normalized vectors by default:
- sentence-transformers (optional)
- OpenAI embeddings (normalized)
- Some image models (normalized)

For these cases, Inner Product is the fastest option!
    """)


def demo_use_cases():
    """Show recommended use cases for each metric."""
    print_section("PART 5: When to Use Each Distance Metric")
    
    use_cases = {
        "L2 (Euclidean Distance)": {
            "description": "Measures straight-line distance in vector space",
            "best_for": [
                "Image embeddings",
                "General-purpose similarity",
                "When magnitude matters",
                "Spatial data",
                "Feature vectors where scale is meaningful"
            ],
            "example_code": """
collection = client.create_collection(
    name="images",
    metadata={"hnsw:space": "l2"}
)
            """,
            "note": "Default metric in ChromaDB"
        },
        "Cosine Similarity": {
            "description": "Measures angle between vectors, ignoring magnitude",
            "best_for": [
                "Text embeddings (most common)",
                "Document similarity",
                "Semantic search",
                "Natural language processing",
                "Any scenario where direction matters more than magnitude"
            ],
            "example_code": """
collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)
            """,
            "note": "Most popular for text/NLP applications"
        },
        "Inner Product": {
            "description": "Dot product of vectors (higher = more similar)",
            "best_for": [
                "Normalized/unit vectors",
                "Maximum Inner Product Search (MIPS)",
                "Pre-normalized embeddings (OpenAI, etc.)",
                "Recommendation systems",
                "When speed is critical and vectors are normalized"
            ],
            "example_code": """
collection = client.create_collection(
    name="recommendations",
    metadata={"hnsw:space": "ip"}
)
            """,
            "note": "Fastest option for normalized vectors"
        }
    }
    
    for metric, info in use_cases.items():
        print(f"\n{'═' * 80}")
        print(f"📊 {metric}")
        print('═' * 80)
        print(f"\nDescription: {info['description']}")
        print(f"\n✓ Best for:")
        for use_case in info['best_for']:
            print(f"  • {use_case}")
        print(f"\nExample Code:")
        print(info['example_code'])
        print(f"\n💡 Note: {info['note']}")
    
    print("\n" + "=" * 80)
    print("QUICK DECISION GUIDE:")
    print("=" * 80)
    print("""
┌─────────────────────────────────────────────────────────────────────────┐
│  Are you working with text/documents?                                   │
│  ├─ YES → Use COSINE                                                    │
│  └─ NO → Continue...                                                    │
│                                                                          │
│  Are your vectors already normalized?                                   │
│  ├─ YES → Use INNER PRODUCT (fastest)                                   │
│  └─ NO → Continue...                                                    │
│                                                                          │
│  Does vector magnitude/scale matter for your use case?                  │
│  ├─ YES → Use L2                                                        │
│  └─ NO → Use COSINE                                                     │
│                                                                          │
│  Not sure? → Start with COSINE (most versatile)                         │
└─────────────────────────────────────────────────────────────────────────┘
    """)


def main():
    """Run all demonstrations."""
    print("=" * 80)
    print(" ChromaDB Distance Metrics Comprehensive Demo")
    print("=" * 80)
    print("""
This demo will show you:
1. How different distance metrics calculate similarity
2. How they affect search results in ChromaDB
3. Real-world examples with text embeddings
4. When to use each metric
    """)
    
    try:
        # Part 1: Basic vector comparison
        demo_basic_vectors()
        
        # Part 2: ChromaDB comparison
        demo_chromadb_comparison()
        
        # Part 3: Text embeddings
        demo_text_embeddings()
        
        # Part 4: Normalized vectors
        demo_normalized_vectors()
        
        # Part 5: Use cases
        demo_use_cases()
        
        print_section("Demo Complete! 🎉")
        print("""
Key Takeaways:
1. L2 (Euclidean): Best for general purpose, sensitive to magnitude
2. Cosine: Best for text/NLP, ignores magnitude, focuses on direction
3. Inner Product: Best for normalized vectors, fastest option

For text applications: Use Cosine
For images: Use L2
For normalized embeddings: Use Inner Product

Next steps:
- Read the full documentation: docs/chroma_vector_search_mechanism.md
- Try the examples with your own data
- Experiment with different metrics for your use case
        """)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
