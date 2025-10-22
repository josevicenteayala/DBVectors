# ChromaDB Vector Search Quick Reference

## Distance Metrics at a Glance

### L2 (Euclidean Distance) - **DEFAULT**

```python
collection = client.create_collection(
    name="my_collection",
    metadata={"hnsw:space": "l2"}  # or omit for default
)
```

- **Formula**: `√(Σ(Aᵢ - Bᵢ)²)`
- **Range**: [0, ∞)
- **Lower is better**: Yes
- **Sensitive to magnitude**: Yes
- **Best for**: Images, general purpose, when scale matters
- **Example**: Distance between [1,0] and [2,0] = 1.0

### Cosine Distance - **RECOMMENDED FOR TEXT**

```python
collection = client.create_collection(
    name="my_collection",
    metadata={"hnsw:space": "cosine"}
)
```

- **Formula**: `1 - (A·B)/(||A||×||B||)`
- **Range**: [0, 2]
- **Lower is better**: Yes
- **Sensitive to magnitude**: No
- **Best for**: Text, documents, semantic search, NLP
- **Example**: Distance between [1,0] and [2,0] = 0.0 (same direction!)

### Inner Product (Dot Product)

```python
collection = client.create_collection(
    name="my_collection",
    metadata={"hnsw:space": "ip"}
)
```

- **Formula**: `Σ(Aᵢ × Bᵢ)` (ChromaDB stores negative)
- **Range**: (-∞, ∞)
- **Lower is better**: Yes (negative stored)
- **Sensitive to magnitude**: Yes
- **Best for**: Normalized vectors, MIPS, recommendation systems
- **Example**: Distance between [1,0] and [2,0] = -2.0

## Quick Decision Tree

```
Working with text/documents?
├─ YES → Use COSINE ✓
└─ NO → Continue...

Vectors already normalized?
├─ YES → Use INNER PRODUCT (fastest) ✓
└─ NO → Continue...

Does magnitude matter?
├─ YES → Use L2 ✓
└─ NO → Use COSINE ✓

Not sure? → Use COSINE (most versatile)
```

## HNSW Parameters

### Default Configuration (Balanced)

```python
# These are defaults, no need to specify
metadata = {
    "hnsw:space": "l2",
    "hnsw:M": 16,                 # Connections per node
    "hnsw:construction_ef": 200,  # Build quality
    "hnsw:search_ef": 10         # Search quality
}
```

### High Quality Configuration (Slower, More Accurate)

```python
metadata = {
    "hnsw:space": "cosine",
    "hnsw:M": 32,                 # More connections
    "hnsw:construction_ef": 400,  # Better index
    "hnsw:search_ef": 100        # More thorough search
}
# Use when: Accuracy is critical, small-medium datasets
```

### Fast Configuration (Faster, Slightly Less Accurate)

```python
metadata = {
    "hnsw:space": "cosine",
    "hnsw:M": 8,                  # Fewer connections
    "hnsw:construction_ef": 100,  # Faster build
    "hnsw:search_ef": 10         # Faster search
}
# Use when: Speed is critical, large datasets
```

## Common Use Cases

### Text Semantic Search

```python
collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)
```

### Image Similarity

```python
collection = client.create_collection(
    name="images",
    metadata={"hnsw:space": "l2"}
)
```

### Recommendation System (Normalized Embeddings)

```python
collection = client.create_collection(
    name="recommendations",
    metadata={"hnsw:space": "ip"}
)
```

## Performance Characteristics

| Metric | Search Speed | Memory | Accuracy | Text | Images |
|--------|-------------|--------|----------|------|--------|
| **L2** | Fast | Medium | High | Good | ✓ Best |
| **Cosine** | Fast | Medium | High | ✓ Best | Good |
| **Inner Product** | ✓ Fastest | Medium | High | Good* | Good* |

*When vectors are normalized

## Distance Comparison Example

Given vectors:
- A: [1, 0, 0]
- B: [2, 0, 0] (same direction, 2x magnitude)
- C: [0, 1, 0] (perpendicular)

| Comparison | L2 | Cosine | Inner Product |
|------------|-----|--------|---------------|
| **A vs A** | 0.0 | 0.0 | -1.0 |
| **A vs B** | 1.0 | 0.0 | -2.0 |
| **A vs C** | 1.414 | 1.0 | 0.0 |

**Key Insight**: Cosine treats A and B as identical (same direction), while L2 and IP distinguish them by magnitude.

## Interpreting Results

### L2 Distance
```python
distance = 0.234
# Lower = more similar
# 0 = identical
# Typical range for normalized vectors: 0-2
```

### Cosine Distance
```python
distance = 0.234
similarity = 1 - distance  # 0.766
# Distance: Lower = more similar
# Similarity: Higher = more similar
# 0 distance (1.0 similarity) = same direction
```

### Inner Product
```python
distance = -2.345  # ChromaDB stores negative
inner_product = -distance  # 2.345
# Higher inner product = more similar
# ChromaDB stores negative for consistent "lower is better" API
```

## Memory Estimation

```python
# Per vector memory usage
memory_per_vector = (
    dimensions × 4 bytes +    # Float32
    M × 8 bytes +            # Graph connections
    metadata_size            # Variable
)

# Example: 384 dimensions, M=16
= 384×4 + 16×8 + 100
= 1,764 bytes (~1.7 KB per vector)

# For 1 million vectors: ~1.7 GB
```

## Search Complexity

- **Brute Force**: O(N × D) - Check every vector
- **HNSW**: O(log N × D) - Only check ~1-5% of vectors
- **Speedup**: 100x-1000x faster than brute force

## Best Practices Checklist

- [ ] Choose metric based on use case (text → cosine)
- [ ] Start with default HNSW parameters
- [ ] Use PersistentClient for production
- [ ] Batch add operations for large datasets
- [ ] Monitor search quality and latency
- [ ] Pre-normalize vectors when using inner product
- [ ] Document your distance metric choice

## Common Pitfalls

❌ **Using L2 for text embeddings**
- Sensitive to embedding magnitude variations
- Use cosine instead

❌ **Using cosine for image embeddings**
- Magnitude can be meaningful for images
- Use L2 instead

❌ **Not normalizing with inner product**
- Inner product favors larger magnitudes
- Normalize vectors or use cosine

❌ **Setting HNSW parameters too high**
- Diminishing returns after certain point
- Start with defaults, tune if needed

❌ **Ignoring search quality**
- Monitor recall and precision
- Adjust parameters if quality drops

## Quick Troubleshooting

**Slow searches?**
- Decrease `search_ef`
- Use smaller M
- Consider sharding for large datasets

**Poor accuracy?**
- Increase `search_ef`
- Increase M
- Use higher `construction_ef`

**High memory usage?**
- Decrease M
- Remove unnecessary metadata
- Use lower-dimensional embeddings

**Unexpected results?**
- Verify correct distance metric for use case
- Check if vectors need normalization
- Validate embedding quality

## Further Reading

- [Full Documentation](chroma_vector_search_mechanism.md)
- [Vector DB Comparison](vector_db_comparison.md)
- [Installation Guide](macos_installation.md)
- [Distance Metrics Demo](../examples/distance_metrics_demo.py)

---

**Quick Reference Version**: 1.0  
**Last Updated**: 2025-10-22
