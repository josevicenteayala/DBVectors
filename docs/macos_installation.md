# ChromaDB Installation Guide for macOS

This guide provides step-by-step instructions for installing ChromaDB and its dependencies on macOS.

## Prerequisites

- macOS 10.13 (High Sierra) or later
- Python 3.7 or higher
- pip (Python package manager)
- Terminal application

## Step 1: Verify Python Installation

Open Terminal and check your Python version:

```bash
python3 --version
```

Expected output: `Python 3.7.x` or higher

If Python is not installed or the version is too old:
- Download Python from [python.org](https://www.python.org/downloads/macos/)
- Or install via Homebrew: `brew install python3`

## Step 2: Create a Virtual Environment (Recommended)

Virtual environments keep project dependencies isolated.

```bash
# Navigate to your project directory
cd ~/path/to/your/project

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

## Step 3: Install ChromaDB

With the virtual environment activated:

```bash
pip install chromadb
```

This will install ChromaDB and all its dependencies.

## Step 4: Install sentence-transformers

For text embeddings, install the sentence-transformers library:

```bash
pip install sentence-transformers
```

This may take a few minutes as it downloads pre-trained models.

## Step 5: Verify Installation

Test that everything is installed correctly:

```bash
python3 -c "import chromadb; print('ChromaDB version:', chromadb.__version__)"
python3 -c "import sentence_transformers; print('sentence-transformers installed successfully!')"
```

## Step 6: Create a Test Script

Create a file named `test_chroma.py`:

```python
import chromadb

# Create a Chroma client
client = chromadb.Client()

# Create a test collection
collection = client.create_collection(name="test_collection")

# Add a test document
collection.add(
    documents=["Hello, ChromaDB on macOS!"],
    metadatas=[{"source": "test"}],
    ids=["test1"]
)

# Query the collection
results = collection.query(
    query_texts=["Hello"],
    n_results=1
)

print("✅ ChromaDB is working correctly!")
print("Results:", results)
```

Run the test:

```bash
python3 test_chroma.py
```

## Alternative Installation Methods

### Using pip with --user flag

If you don't want to use a virtual environment:

```bash
pip install --user chromadb sentence-transformers
```

### Using Poetry

If you use Poetry for dependency management:

```bash
poetry add chromadb sentence-transformers
```

### Using Conda

If you use Anaconda or Miniconda:

```bash
conda create -n chroma python=3.10
conda activate chroma
pip install chromadb sentence-transformers
```

## Troubleshooting

### Issue 1: "command not found: pip"

**Solution:** Use `pip3` instead of `pip`:
```bash
pip3 install chromadb
```

### Issue 2: Permission denied errors

**Solution:** Use a virtual environment or install with `--user` flag:
```bash
pip install --user chromadb
```

### Issue 3: "No module named 'chromadb'" when running scripts

**Solution:** Make sure you're using the same Python environment where you installed ChromaDB:
```bash
# Activate your virtual environment
source venv/bin/activate

# Then run your script
python3 your_script.py
```

### Issue 4: SSL certificate errors during installation

**Solution:** Update pip and try again:
```bash
pip install --upgrade pip
pip install chromadb
```

### Issue 5: "Illegal instruction" error on older Macs

**Solution:** This can happen on older Intel Macs. Try:
```bash
pip install chromadb --no-binary :all:
```

Or upgrade to Python 3.9 or higher.

### Issue 6: Slow download/installation

**Solution:** The sentence-transformers library downloads large model files. This is normal. Ensure you have:
- At least 2GB of free disk space
- A stable internet connection

## Updating ChromaDB

To update to the latest version:

```bash
pip install --upgrade chromadb
```

## Uninstalling ChromaDB

To completely remove ChromaDB:

```bash
pip uninstall chromadb sentence-transformers
```

## Next Steps

Now that ChromaDB is installed, you can:

1. Explore the example code in the `examples/` directory
2. Read the ChromaDB documentation at [docs.trychroma.com](https://docs.trychroma.com)
3. Try the complete workflow demo in `chroma_demo.py`

## System Requirements

- **Disk Space**: ~500MB for ChromaDB and sentence-transformers
- **RAM**: At least 2GB available (4GB+ recommended for larger datasets)
- **Python**: 3.7, 3.8, 3.9, 3.10, or 3.11

## Support

If you encounter issues not covered in this guide:

- Check the [ChromaDB GitHub Issues](https://github.com/chroma-core/chroma/issues)
- Visit the [ChromaDB Discord community](https://discord.gg/MMeYNTmh3x)
- Review the [official documentation](https://docs.trychroma.com)

## Additional Resources

- [ChromaDB Official Website](https://www.trychroma.com/)
- [ChromaDB Python Client Documentation](https://docs.trychroma.com/reference/py-client)
- [sentence-transformers Documentation](https://www.sbert.net/)
