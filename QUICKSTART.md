# Quick Start Guide

Get up and running with ChromaDB vector database in 5 minutes!

## Prerequisites

- macOS 10.13 or later
- Python 3.7+
- Terminal

## Installation (2 minutes)

### Option 1: Basic Installation (Offline Demo)

```bash
# Clone the repository
git clone https://github.com/josevicenteayala/DBVectors.git
cd DBVectors

# Install ChromaDB only
pip install chromadb

# Run the offline demo
python examples/chroma_demo_offline.py
```

### Option 2: Full Installation (Best Results)

```bash
# Clone the repository
git clone https://github.com/josevicenteayala/DBVectors.git
cd DBVectors

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Run the full demo
python examples/chroma_demo.py
```

## What You'll See

The demo will show you:

1. ✅ How to connect to ChromaDB
2. ✅ Creating a collection for your data
3. ✅ Converting text to vector embeddings
4. ✅ Storing documents with metadata
5. ✅ Running semantic searches
6. ✅ Interpreting search results

## Example Output

```
🔍 Query: 'How do plants create energy?'
----------------------------------------------------------------------

📊 Search Results (ranked by relevance):

  Rank 1:
  📄 Document: Photosynthesis is the process by which plants convert sunlight into energy.
  🏷️  Category: biology
  🎯 Topic: plants
  📏 Distance: 0.4521
  ⭐ Similarity Score: 0.8456
  🟢 Relevance: HIGHLY RELEVANT
```

## Try Your Own Queries

After running the demo, modify it to try your own data:

```python
# In the demo file, replace the documents array:
documents = [
    "Your first document here",
    "Your second document here",
    "Your third document here"
]

# And try your own queries:
queries = [
    "Your search query here"
]
```

## Next Steps

1. **Read the Comparison**: [Vector DB Comparison](docs/vector_db_comparison.md)
2. **Detailed Installation**: [macOS Installation Guide](docs/macos_installation.md)
3. **Explore Examples**: See [examples/README.md](examples/README.md) for all demo versions
4. **Build Your App**: Use the demo code as a template for your project

## Common Use Cases

### 1. Question Answering System

```python
# Store your knowledge base
documents = [
    "Our product ships worldwide in 2-3 days",
    "Returns are accepted within 30 days",
    "We offer 24/7 customer support"
]

# Search for answers
query = "What is your shipping policy?"
results = collection.query(query_texts=[query], n_results=1)
```

### 2. Document Search

```python
# Store your documents
documents = load_documents_from_folder("./my_documents/")

# Find relevant documents
query = "Machine learning best practices"
results = collection.query(query_texts=[query], n_results=5)
```

### 3. Content Recommendations

```python
# Store content items
documents = ["Article about Python", "Article about JavaScript", ...]

# Find similar content
current_article = "Deep dive into Python decorators"
similar = collection.query(query_texts=[current_article], n_results=3)
```

## Troubleshooting

### Installation Issues

**Problem**: `pip install` fails
```bash
# Solution: Upgrade pip first
pip install --upgrade pip
```

**Problem**: Permission denied
```bash
# Solution: Use a virtual environment
python3 -m venv venv
source venv/bin/activate
```

### Runtime Issues

**Problem**: Model download is slow
- First run downloads models (~100-500MB)
- Subsequent runs are instant
- Or use `chroma_demo_offline.py` for no downloads

**Problem**: Out of memory
- Reduce the number of documents
- Use a smaller embedding model
- Close other applications

## Getting Help

- 📖 [Full Documentation](docs/)
- 💬 [ChromaDB Discord](https://discord.gg/MMeYNTmh3x)
- 🐛 [Report Issues](https://github.com/josevicenteayala/DBVectors/issues)

## What Makes ChromaDB Great?

- 🚀 **Fast Setup**: One pip install command
- 🎯 **Easy to Use**: Simple, intuitive API
- 🔍 **Powerful Search**: Find content by meaning, not just keywords
- 📦 **No Infrastructure**: No servers or Docker required
- 🐍 **Pure Python**: Works on any platform

## Ready to Build?

You now have everything you need to:
- ✅ Store and search documents semantically
- ✅ Build a question-answering system
- ✅ Create a recommendation engine
- ✅ Implement document similarity search

Happy coding! 🎉
