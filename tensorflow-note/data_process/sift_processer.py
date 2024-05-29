#!/usr/bin/env python3

# taking from https://github.com/erikbern/ann-benchmarks/blob/master/ann_benchmarks/datasets.py#L164
def sift():
    import os
    import struct
    import tarfile
    from urllib.request import urlretrieve
    import numpy as np
    def _load_fn(t, fn):
        m = t.getmember(fn)
        f = t.extractfile(m)
        k, = struct.unpack("i", f.read(4))
        n = m.size // (4 + 4 * k)
        f.seek(0)
        return n, k, f
    def _load_fvecs(t, fn):
        n, k, f = _load_fn(t, fn)
        v = np.zeros((n, k), dtype=np.float32)
        for i in range(n):
            f.read(4)  # ignore vec length
            v[i] = struct.unpack("f" * k, f.read(k * 4))
        return v
    def _load_ivecs(t, fn):
        n, k, f = _load_fn(t, fn)
        v = np.zeros((n, k), dtype=np.int32)
        for i in range(n):
            f.read(4)  # ignore vec length
            v[i] = struct.unpack("i" * k, f.read(k * 4))
        return v
    # download dataset if we have not downloaded it yes
    url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    fn = os.path.join("./test_data", "sift.tar.gz")
    if not os.path.exists(fn):
        print(f"downloading {url} -> {fn}...")
        urlretrieve(url, fn)
    # load dataset
    with tarfile.open(fn, "r:gz") as t:
        train = _load_fvecs(t, "sift/sift_base.fvecs")
        test = _load_fvecs(t, "sift/sift_query.fvecs")
        neighbors = _load_ivecs(t, "sift/sift_groundtruth.ivecs") + 1
    print(f"train.shape: {train.shape}")
    print(f"test.shape: {test.shape}")
    print(f"neighbors.shape: {neighbors.shape}")
    return train, test, neighbors