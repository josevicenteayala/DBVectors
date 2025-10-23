"""FAISS IVF+PQ example

This script demonstrates a minimal, runnable example using FAISS to build an
IVF+PQ index, add random vectors, and perform a nearest neighbor search.

Requirements:
- faiss-cpu (pip install faiss-cpu)
- numpy

Run:
python examples/faiss_ivfpq_example.py
"""
import numpy as np

try:
    import faiss
except Exception as e:
    raise SystemExit("faiss is required. Install with: pip install faiss-cpu")


def main():
    d = 64
    nb = 20000
    nq = 5

    np.random.seed(1234)
    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')

    nlist = 256
    m = 8  # number of subquantizers for PQ
    nbits = 8

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

    # Train on a subset
    train_samples = xb[: min(10000, nb)]
    index.train(train_samples)

    # add vectors
    index.add(xb)

    # tune search params
    index.nprobe = 8

    k = 5
    D, I = index.search(xq, k)
    print('Distances:', D)
    print('Indices:', I)


if __name__ == '__main__':
    main()
