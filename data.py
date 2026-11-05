from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import hashlib

import numpy as np
from PIL import Image
from scipy import sparse
from scipy.linalg import cholesky, cho_solve
from scipy.spatial.distance import cdist

ARCHIVE = Path(__file__).parent / "data" / "bach_original.zip"
DATASETS = ["long", "wide", "moons", "speech", "image"]


@dataclass
class Instance:
    name: str
    kind: str
    b: np.ndarray
    M: object
    N: object = None
    meta: dict = None

    @property
    def n(self):
        return len(self.b)

    def value(self, z):
        """Independent scalar evaluator, also useful for correctness tests."""
        z = np.asarray(z, dtype=float)
        A = np.flatnonzero(z)
        if self.kind == "cut":
            return float(
                self.b @ z + z @ (np.asarray(self.M.sum(axis=1)).ravel() - self.M @ z)
            )
        if self.kind == "speech":
            count = np.count_nonzero(np.asarray(self.M[A].sum(axis=0)))
            return float(self.b @ z + 22 * np.sqrt(count))
        total = self.b @ z
        if len(A):
            for K in (self.M, self.N):
                total += (
                    2 * np.log(np.diag(cholesky(K[np.ix_(A, A)], lower=True))).sum()
                )
        return float(total)

    def chain(self, order):
        """Return all prefix values and adjacent second differences (not probe values).

        Cuts use incremental sums; coverage tracks word counts; logdets use one
        Cholesky per matrix. No n separate factorizations per greedy call.
        """
        b = self.b[order]
        if self.kind == "cut":
            C = self.M[order][:, order]
            d = np.asarray(self.M.sum(axis=1)).ravel()[order]
            before = (
                np.asarray(sparse.tril(C, -1).sum(axis=1)).ravel()
                if sparse.issparse(C)
                else np.tril(C, -1).sum(axis=1)
            )
            increments = b + d - 2 * before
            edges = np.asarray(self.M[order[:-1], order[1:]]).ravel()
            delta = 2 * edges
        elif self.kind == "speech":
            W = self.M
            covered = np.zeros(W.shape[1], dtype=bool)
            counts, alternate = [0], []
            for k, i in enumerate(order):
                if k + 1 < self.n:
                    j = order[k + 1]
                    words = W.indices[W.indptr[j] : W.indptr[j + 1]]
                    alternate.append(counts[-1] + np.count_nonzero(~covered[words]))
                words = W.indices[W.indptr[i] : W.indptr[i + 1]]
                covered[words] = True
                counts.append(np.count_nonzero(covered))
            v = np.r_[0.0, np.cumsum(b)] + 22 * np.sqrt(counts)
            q = np.r_[0.0, np.cumsum(b)][:-2] + b[1:] + 22 * np.sqrt(alternate)
            return v, v[1:-1] + q - v[:-2] - v[2:]
        else:
            increments = b.copy()
            delta = np.zeros(self.n - 1)
            for K in (self.M, self.N):
                L = cholesky(K[np.ix_(order, order)], lower=True, check_finite=False)
                increments += 2 * np.log(np.diag(L))
                delta += np.log1p(
                    (L[np.arange(1, self.n), np.arange(self.n - 1)] / np.diag(L)[1:])
                    ** 2
                )
        return np.r_[0.0, np.cumsum(increments)], delta

    def restrict(self, limit, seed):
        """A restriction fixes all omitted variables to zero; it is not the full benchmark."""
        if not limit or limit >= self.n:
            return self
        ids = np.sort(np.random.default_rng(seed).choice(self.n, limit, replace=False))
        b = self.b[ids].copy()
        if self.kind == "cut":
            M = self.M[ids][:, ids]
            b += (
                np.asarray(self.M.sum(axis=1)).ravel()[ids]
                - np.asarray(M.sum(axis=1)).ravel()
            )
        else:
            M = self.M[ids] if self.kind == "speech" else self.M[np.ix_(ids, ids)]
        N = None if self.N is None else self.N[np.ix_(ids, ids)]
        meta = dict(
            self.meta,
            original_n=self.n,
            free_indices=ids.tolist(),
            restriction="omitted variables fixed to zero",
        )
        return Instance(self.name, self.kind, b, M, N, meta)

    def fingerprint(self):
        h = hashlib.sha256(self.b.tobytes())
        for M in (self.M, self.N):
            if M is not None:
                for a in (
                    (M.data, M.indices, M.indptr) if sparse.issparse(M) else (M,)
                ):
                    h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


def load(name, seed=1, image_mode="corrected", image_size=50):
    rng = np.random.default_rng(seed)
    meta = dict(dataset=name, seed=seed, source="Bach submodular_2_0.zip", f_empty=0.0)
    with ZipFile(ARCHIVE) as archive:

        def read(file):
            return archive.read("submodular_online/" + file)

        if name in ("long", "wide"):
            if not 1 <= seed <= 10:
                raise ValueError(
                    "The supplied GENRMF instances have seeds 1 through 10."
                )
            lines = read(f"graph_{name}_{seed}.txt").decode().splitlines()
            n = int(next(s for s in lines if s.startswith("p ")).split()[2])
            arcs = np.array(
                [list(map(int, s.split()[1:])) for s in lines if s.startswith("a ")]
            )
            W = sparse.coo_matrix(
                (arcs[:, 2].astype(float), (arcs[:, 0] - 1, arcs[:, 1] - 1)),
                shape=(n, n),
            ).tocsr()
            U = W + W.T  # Exactly Bach's symmetrized cut, not directed max-flow.
            rp = rng.permutation(n - 2) + 1
            C = U[rp][:, rp].tocsr()
            b = U[rp, n - 1].toarray().ravel() - U[rp, 0].toarray().ravel()
            meta.update(
                total_vertices=n,
                arcs=len(arcs),
                raw_cut_offset=float(U[0].sum()),
                graph_file=f"graph_{name}_{seed}.txt",
            )
            return Instance(name, "cut", b, C, meta=meta)

        if name == "speech":
            arcs = [
                list(map(int, s.split()[1:]))
                for s in read("uniform-sent.800").decode().splitlines()
                if s.startswith("a ")
            ]
            pairs = np.array(
                [(i - 3, j - 803) for i, j, _ in arcs if i != 1 and j != 2]
            )
            W = sparse.coo_matrix(
                (np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])),
                shape=(800, 1105),
            ).tocsr()
            W.sum_duplicates()
            W = W[rng.permutation(800)]
            meta.update(groups=1105, incidences=W.nnz, lambda_=22.0, alpha=0.5)
            return Instance(name, "speech", -np.ones(800), W, meta=meta)

        if name == "moons":
            n = 200
            r = 2 + 0.4 * rng.standard_normal(n)
            theta = -np.pi / 2 + np.pi * rng.random(n)
            X1 = np.c_[-0.5 + r * np.cos(theta), 1 + r * np.sin(theta)]
            r = 2 + 0.4 * rng.standard_normal(n)
            theta = np.pi / 2 + np.pi * rng.random(n)
            X2 = np.c_[0.5 + r * np.cos(theta), -1 + r * np.sin(theta)]
            X = np.r_[X1, X2][rng.permutation(400)]
            D = cdist(X, X, "sqeuclidean")
            K = np.exp(-8 * D / np.median(D)) + 0.04 * np.eye(400)
            inv = cho_solve((cholesky(K, lower=True), True), np.eye(400))
            inv = (inv + inv.T) / 2
            singleton = np.log(np.diag(K)) + np.log(np.diag(inv))
            val = 2 * singleton.max()
            s = np.zeros(400)
            s[:8], s[-8:] = val, -val
            rp = rng.permutation(400)
            meta.update(
                labels=16,
                generator="NumPy PCG64; Bach construction, not MATLAB-identical",
            )
            return Instance(
                name, "logdet", -s[rp], K[np.ix_(rp, rp)], inv[np.ix_(rp, rp)], meta
            )

        if name == "image":
            size = (image_size, image_size)
            labels = np.asarray(
                Image.open(BytesIO(read("gray3_labels.png")))
                .convert("RGB")
                .resize(size, Image.Resampling.BICUBIC)
            )
            image = (
                np.asarray(
                    Image.open(BytesIO(read("gray3.png")))
                    .convert("RGB")
                    .resize(size, Image.Resampling.BICUBIC)
                ).mean(axis=2)
                / 255
            )
            image += rng.standard_normal(size) / 20
            # MATLAB's column-major pixel indexing.
            xx, yy = np.meshgrid(np.arange(image_size), np.arange(image_size))
            coords = np.c_[xx.ravel(order="F"), yy.ravel(order="F")]
            D1 = cdist(coords, coords)
            D2 = cdist(image.ravel(order="F")[:, None], image.ravel(order="F")[:, None])
            K = np.exp(-32 * D1 / np.median(D1) - 4 * D2 / np.median(D2))
            np.fill_diagonal(K, 0.0)
            rp = rng.permutation(len(K))
            K = K[np.ix_(rp, rp)]
            val = K.sum(axis=1).max()  # 2 * max singleton of .5 cut(K).
            s = np.zeros(len(K))
            s[labels[:, :, 0].ravel(order="F") < 100] = val
            s[labels[:, :, 2].ravel(order="F") < 200] = -val
            if image_mode == "corrected":
                s = s[rp]
            meta.update(
                image_size=image_size,
                image_mode=image_mode,
                generator="Pillow bicubic / NumPy PCG64; not MATLAB-identical",
            )
            return Instance(name, "cut", -s, 0.5 * K, meta=meta)
    raise ValueError(name)
