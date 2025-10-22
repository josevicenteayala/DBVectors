# DBVectors

This repository contains information about how vector databases work and provides practical examples and demos.

## Overview

Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They enable semantic search, similarity matching, and are fundamental to modern AI applications like RAG (Retrieval-Augmented Generation), recommendation systems, and more.

## Contents

### Documentation

- **[Vector DB Comparison](docs/vector_db_comparison.md)**: Comprehensive comparison of Chroma vs. Qdrant for macOS, with recommendations for beginners
- **[macOS Installation Guide](docs/macos_installation.md)**: Step-by-step installation instructions for ChromaDB on macOS
- **[ChromaDB Vector Search Mechanism](docs/chroma_vector_search_mechanism.md)**: Deep dive into how ChromaDB calculates nearest vectors, including HNSW indexing, distance metrics (L2, Cosine, Inner Product), and the complete search process
- **[Quick Reference Guide](docs/chroma_quick_reference.md)**: Concise reference for distance metrics, HNSW parameters, and common use cases

### Examples

- **[ChromaDB Complete Demo](examples/chroma_demo.py)**: Full workflow demonstration including:
  1. Connecting and creating a collection
  2. Embedding a text dataset using sentence-transformers
  3. Populating the database with vectors and metadata
  4. Embedding queries and running semantic searches
  5. Retrieving and interpreting results to show relevance
- **[Distance Metrics Demo](examples/distance_metrics_demo.py)**: Interactive demonstration comparing L2, Cosine, and Inner Product distance metrics with practical examples

## Quick Start

### Prerequisites

- Python 3.7 or higher
- macOS (guide optimized for macOS, but works on other platforms too)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/josevicenteayala/DBVectors.git
   cd DBVectors
   ```

2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Demo

Run the complete ChromaDB workflow demo:

```bash
python examples/chroma_demo.py
```

This will demonstrate:
- Setting up a ChromaDB collection
- Embedding documents using sentence-transformers
- Performing semantic searches
- Interpreting similarity results

## Why ChromaDB?

After comparing several vector databases, we chose **ChromaDB** for this demo because:

- ✅ **Easy Setup**: Single `pip install` command
- ✅ **Beginner-Friendly**: Intuitive API and excellent documentation
- ✅ **No Dependencies**: Pure Python, no Docker or compilation needed
- ✅ **macOS Native**: Works perfectly on all macOS versions
- ✅ **Production-Ready**: Can scale from prototypes to production

See the [comparison document](docs/vector_db_comparison.md) for a detailed analysis.

## Learning Path

1. **Start Here**: Read the [Vector DB Comparison](docs/vector_db_comparison.md)
2. **Installation**: Follow the [macOS Installation Guide](docs/macos_installation.md)
3. **Deep Dive**: Understand [ChromaDB's Vector Search Mechanism](docs/chroma_vector_search_mechanism.md)
4. **Hands-On**: Run the [ChromaDB Demo](examples/chroma_demo.py)
5. **Explore Metrics**: Try the [Distance Metrics Demo](examples/distance_metrics_demo.py)
6. **Experiment**: Modify the demos with your own data and queries

## Key Concepts

- **Embeddings**: Vector representations of text that capture semantic meaning
- **Semantic Search**: Finding similar content based on meaning, not just keywords
- **Vector Database**: Specialized storage for efficient similarity search
- **Distance Metrics**: Methods for measuring similarity (L2, Cosine, Inner Product)
- **HNSW Index**: Hierarchical Navigable Small World graphs for fast nearest neighbor search
- **Metadata Filtering**: Combining semantic search with structured filters

## Use Cases

Vector databases enable powerful applications:

- 🤖 **Chatbots with RAG**: Answer questions using your own documents
- 🔍 **Semantic Search**: Find relevant content even with different wording
- 💡 **Recommendation Systems**: Suggest similar items based on user preferences
- 📊 **Document Analysis**: Cluster and categorize large document collections
- 🎯 **Content Deduplication**: Find and merge similar content

## Contributing

Contributions are welcome! Feel free to:
- Add new examples
- Improve documentation
- Compare additional vector databases
- Share your use cases

## Resources

- [ChromaDB Official Documentation](https://docs.trychroma.com)
- [sentence-transformers Documentation](https://www.sbert.net/)
- [Vector Database Explained](https://www.pinecone.io/learn/vector-database/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
