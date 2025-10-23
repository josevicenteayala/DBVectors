"""hnswlib example

Requirements:
- hnswlib (pip install hnswlib)
- numpy

Run:
python examples/hnswlib_example.py
"""
import numpy as np

try:
    import hnswlib
except Exception:
    raise SystemExit("hnswlib is required. Install with: pip install hnswlib")


def main():
    d = 64
    num_elements = 20000

    np.random.seed(123)
    data = np.random.random((num_elements, d)).astype('float32')
    queries = np.random.random((5, d)).astype('float32')

    p = hnswlib.Index(space='l2', dim=d)
    p.init_index(max_elements=num_elements, ef_construction=200, M=16)
    p.add_items(data, np.arange(num_elements))

    p.set_ef(50)
    labels, distances = p.knn_query(queries, k=5)
    print('Labels:', labels)
    print('Distances:', distances)


if __name__ == '__main__':
    main()
