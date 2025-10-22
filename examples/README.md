# ChromaDB Examples

This directory contains different versions of the ChromaDB demo, each suited for different scenarios.

## Available Demos

### 1. chroma_demo.py (Recommended for Production)

The complete demo using **sentence-transformers** for high-quality embeddings.

**Features:**
- ✅ Best semantic search quality
- ✅ Uses pre-trained ML models
- ✅ Complete workflow with all 5 steps
- ✅ Detailed explanations and interpretations

**Requirements:**
- Internet connection (to download models on first run)
- ~500MB disk space for models

**Run:**
```bash
pip install -r requirements.txt
python examples/chroma_demo.py
```

**Best for:**
- Production applications
- Learning about real semantic search
- Getting accurate results

---

### 2. chroma_demo_simple.py (Easy Setup)

Simplified version using ChromaDB's **default embedding function**.

**Features:**
- ✅ Simpler code
- ✅ Built-in embeddings
- ✅ Good search quality
- ✅ Complete workflow

**Requirements:**
- Internet connection (to download ONNX models on first run)
- Less storage than sentence-transformers

**Run:**
```bash
pip install chromadb
python examples/chroma_demo_simple.py
```

**Best for:**
- Quick testing
- When you don't want to install sentence-transformers
- Prototyping

---

### 3. chroma_demo_offline.py (No Internet Required)

Fully offline version with a **custom embedding function** for testing.

**Features:**
- ✅ Works completely offline
- ✅ No model downloads needed
- ✅ Demonstrates the workflow
- ⚠️ Lower search quality (hash-based embeddings)

**Requirements:**
- Only chromadb package
- No internet connection needed

**Run:**
```bash
pip install chromadb
python examples/chroma_demo_offline.py
```

**Best for:**
- Testing in restricted environments
- Understanding the ChromaDB workflow
- Development without internet access

**Note:** This uses simple hash-based embeddings for demonstration. For real semantic search, use chroma_demo.py or chroma_demo_simple.py.

---

## Quick Comparison

| Feature | chroma_demo.py | chroma_demo_simple.py | chroma_demo_offline.py |
|---------|----------------|----------------------|------------------------|
| Search Quality | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good | ⭐⭐ Basic |
| Setup Complexity | Medium | Easy | Very Easy |
| Internet Required | Yes (first run) | Yes (first run) | No |
| Disk Space | ~500MB | ~100MB | Minimal |
| Production Ready | ✅ Yes | ✅ Yes | ❌ Demo only |

## What Each Demo Teaches

All demos cover the same 5-step workflow:

1. **Connect and create a collection**
   - Initialize ChromaDB client
   - Create a collection with metadata

2. **Embed a text dataset**
   - Prepare documents and metadata
   - Generate vector embeddings

3. **Populate the database**
   - Store documents with embeddings
   - Associate metadata

4. **Embed queries and run semantic search**
   - Convert queries to vectors
   - Find similar documents

5. **Retrieve and interpret results**
   - Understand distance metrics
   - Interpret similarity scores
   - Apply metadata filters

## Recommendations

- **New to vector databases?** Start with `chroma_demo_offline.py` to understand the concepts without setup hassle.

- **Ready to see real results?** Move to `chroma_demo.py` for the best semantic search experience.

- **Building an application?** Use `chroma_demo.py` as your starting point and customize it for your needs.

- **Testing in CI/CD?** Use `chroma_demo_offline.py` for automated tests without external dependencies.

## Next Steps

After running these demos:

1. **Modify the documents**: Replace the sample data with your own documents
2. **Try different queries**: Test the semantic search with various questions
3. **Experiment with metadata**: Add more metadata fields and filters
4. **Implement persistence**: Use `PersistentClient` to save data between runs
5. **Explore advanced features**: 
   - Custom distance metrics
   - Batch operations
   - Collection management

## Troubleshooting

### "No module named 'chromadb'"
```bash
pip install chromadb
```

### "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### Internet connection errors
- Use `chroma_demo_offline.py` instead
- Or pre-download models when you have internet access

### Slow first run
- The first run downloads embedding models
- Subsequent runs will be much faster
- Models are cached locally

## Additional Resources

- [ChromaDB Documentation](https://docs.trychroma.com)
- [sentence-transformers Documentation](https://www.sbert.net/)
- [Vector Database Comparison](../docs/vector_db_comparison.md)
- [macOS Installation Guide](../docs/macos_installation.md)
