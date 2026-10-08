import numpy as np


def is_prime(m):
    if m < 2:
        return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small:
        if m % p == 0:
            return m == p
    d = m - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, m)
        if x == 1 or x == m - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, m)
            if x == m - 1:
                break
        else:
            return False
    return True


def prime_5mod8(bits):
    m = (1 << bits) - 3
    while True:
        if m % 8 == 5 and is_prime(m):
            return m
        m -= 1


class Ring:
    def __init__(self, n, q, max_terms=64):
        self.n = n
        self.q = q
        bits = 2 * q.bit_length() + n.bit_length() + max_terms.bit_length() + 1
        self.width = max(8, (bits + 7) // 8)
        self.two64 = pow(2, 64, q)
        self.small_q = q < (1 << 31)

    def reduce(self, a):
        return np.mod(np.asarray(a, dtype=np.int64), self.q)

    def centered(self, a):
        a = np.mod(np.asarray(a, dtype=np.int64), self.q)
        return np.where(a > self.q // 2, a - self.q, a)

    def uniform(self, rng, shape):
        return rng.integers(0, self.q, size=tuple(shape) + (self.n,), dtype=np.int64)

    def pack(self, a):
        a = np.mod(np.asarray(a, dtype=np.int64), self.q).astype("<u8")
        count = a.shape[0]
        raw = np.zeros((count, self.width), dtype=np.uint8)
        raw[:, :8] = a.view(np.uint8).reshape(count, 8)
        return int.from_bytes(raw.tobytes(), "little")

    def unpack_mod(self, x, count):
        raw = np.frombuffer(x.to_bytes(self.width * count, "little"), dtype=np.uint8).reshape(count, self.width)
        lo = np.ascontiguousarray(raw[:, :8]).view("<u8").reshape(count)
        hb = np.zeros((count, 8), dtype=np.uint8)
        hb[:, : self.width - 8] = raw[:, 8:]
        hi = hb.view("<u8").reshape(count)
        q = self.q
        lo_m = (lo % np.uint64(q)).astype(np.int64)
        hi_m = (hi % np.uint64(q)).astype(np.int64)
        if self.small_q:
            return np.mod(lo_m + np.mod(hi_m * self.two64, q), q)
        prod = np.array([int(h) * self.two64 % q for h in hi_m], dtype=np.int64)
        return np.mod(lo_m + prod, q)

    def fold(self, c):
        n = self.n
        tail = np.zeros(n, dtype=np.int64)
        tail[: n - 1] = c[n: 2 * n - 1]
        return np.mod(c[:n] - tail, self.q)

    def mul(self, a, b):
        C = self.pack(a) * self.pack(b)
        return self.fold(self.unpack_mod(C, 2 * self.n))

    def add(self, a, b):
        return np.mod(np.asarray(a, dtype=np.int64) + np.asarray(b, dtype=np.int64), self.q)

    def sub(self, a, b):
        return np.mod(np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64), self.q)

    def packvec(self, v):
        return [self.pack(v[j]) for j in range(v.shape[0])]

    def matvec_packed(self, Mp, vp):
        rows = len(Mp)
        out = np.zeros((rows, self.n), dtype=np.int64)
        for i in range(rows):
            acc = 0
            for mij, vj in zip(Mp[i], vp):
                acc += mij * vj
            out[i] = self.fold(self.unpack_mod(acc, 2 * self.n))
        return out

    def packmat(self, M):
        return [[self.pack(M[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])]

    def matvec(self, M, v):
        return self.matvec_packed(self.packmat(M), self.packvec(v))

    def scalar_vec(self, c, v):
        cp = self.pack(c)
        return np.stack([self.fold(self.unpack_mod(cp * self.pack(v[j]), 2 * self.n)) for j in range(v.shape[0])])

    def embedding(self, a):
        a = np.asarray(a, dtype=np.float64)
        n = a.shape[-1]
        tw = np.exp(1j * np.pi * np.arange(n) / n)
        return np.fft.fft(a * tw, axis=-1)

    def spectral_sq(self, a):
        return float(np.max(np.abs(self.embedding(a)) ** 2))


def schoolbook(a, b, q):
    n = len(a)
    c = np.zeros(n, dtype=object)
    for i in range(n):
        for j in range(n):
            k = i + j
            if k < n:
                c[k] += int(a[i]) * int(b[j])
            else:
                c[k - n] -= int(a[i]) * int(b[j])
    return np.array([int(x) % q for x in c], dtype=np.int64 if q < (1 << 62) else object)


class BigRing:
    def __init__(self, n, q, max_terms=64):
        self.n = n
        self.q = q
        self.width = (2 * q.bit_length() + n.bit_length() + max_terms.bit_length() + 8) // 8

    def _obj(self, a):
        return np.array([int(x) for x in np.ravel(a)], dtype=object).reshape(np.shape(a))

    def reduce(self, a):
        return self._obj(a) % self.q

    def centered(self, a):
        a = self._obj(a) % self.q
        h = self.q // 2
        return np.array([x - self.q if x > h else x for x in np.ravel(a)], dtype=object).reshape(a.shape)

    def uniform(self, rng, shape):
        shape = tuple(shape) + (self.n,)
        nb = (self.q.bit_length() + 7) // 8 + 16
        raw = rng.bytes(int(np.prod(shape)) * nb)
        vals = [int.from_bytes(raw[i * nb:(i + 1) * nb], "little") % self.q for i in range(int(np.prod(shape)))]
        return np.array(vals, dtype=object).reshape(shape)

    def pack(self, a):
        w = self.width
        return int.from_bytes(b"".join((int(x) % self.q).to_bytes(w, "little") for x in a), "little")

    def unpack_mod(self, x, count):
        w = self.width
        bs = x.to_bytes(w * count, "little")
        return np.array([int.from_bytes(bs[i * w:(i + 1) * w], "little") % self.q for i in range(count)], dtype=object)

    def fold(self, c):
        n = self.n
        out = c[:n].copy()
        out[: n - 1] = out[: n - 1] - c[n: 2 * n - 1]
        return out % self.q

    def mul(self, a, b):
        return self.fold(self.unpack_mod(self.pack(a) * self.pack(b), 2 * self.n))

    def add(self, a, b):
        return (self._obj(a) + self._obj(b)) % self.q

    def sub(self, a, b):
        return (self._obj(a) - self._obj(b)) % self.q

    def packvec(self, v):
        return [self.pack(v[j]) for j in range(v.shape[0])]

    def packmat(self, M):
        return [[self.pack(M[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])]

    def matvec_packed(self, Mp, vp):
        rows = len(Mp)
        out = np.empty((rows, self.n), dtype=object)
        for i in range(rows):
            acc = 0
            for mij, vj in zip(Mp[i], vp):
                acc += mij * vj
            out[i] = self.fold(self.unpack_mod(acc, 2 * self.n))
        return out

    def matvec(self, M, v):
        return self.matvec_packed(self.packmat(M), self.packvec(v))

    def scalar_vec(self, c, v):
        cp = self.pack(c)
        return np.stack([self.fold(self.unpack_mod(cp * self.pack(v[j]), 2 * self.n)) for j in range(v.shape[0])])


def make_ring(n, q):
    R = Ring(n, q)
    return R if R.width <= 16 else BigRing(n, q)
