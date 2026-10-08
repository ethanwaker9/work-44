import numpy as np


def rice_encode(values, k):
    v = np.asarray(values, dtype=np.int64).ravel()
    a = np.abs(v)
    hi = a >> k
    lo = a & ((1 << k) - 1)
    lens = 1 + k + hi + 1
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    total = int(lens.sum())
    bits = np.zeros(total, dtype=np.uint8)
    bits[starts] = (v < 0).astype(np.uint8)
    for j in range(k):
        bits[starts + 1 + j] = ((lo >> (k - 1 - j)) & 1).astype(np.uint8)
    bits[starts + 1 + k + hi] = 1
    return np.packbits(bits).tobytes(), total


def rice_decode(data, count, k):
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    ones = np.flatnonzero(bits)
    out = np.empty(count, dtype=np.int64)
    pos = 0
    weights = 1 << np.arange(k - 1, -1, -1)
    for i in range(count):
        sign = bits[pos]
        lo = int(np.dot(bits[pos + 1: pos + 1 + k], weights)) if k > 0 else 0
        start = pos + 1 + k
        j = np.searchsorted(ones, start)
        if j >= len(ones):
            raise ValueError("truncated")
        end = int(ones[j])
        hi = end - start
        mag = (hi << k) | lo
        out[i] = -mag if sign else mag
        pos = end + 1
    return out


def fixed_encode(values, bound):
    v = np.asarray(values, dtype=np.int64).ravel() + bound
    b = int(2 * bound).bit_length()
    bits = ((v[:, None] >> np.arange(b - 1, -1, -1)) & 1).astype(np.uint8).ravel()
    return np.packbits(bits).tobytes(), b * v.size


def fixed_decode(data, count, bound):
    b = int(2 * bound).bit_length()
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[: b * count].reshape(count, b).astype(np.int64)
    return bits @ (1 << np.arange(b - 1, -1, -1)) - bound


def big_encode(values, nbits):
    out = bytearray()
    nb = (nbits + 7) // 8
    off = 1 << (nbits - 1)
    for x in np.ravel(values):
        out += (int(x) + off).to_bytes(nb, "little")
    return bytes(out), 8 * nb * np.size(values)


def big_decode(data, count, nbits):
    nb = (nbits + 7) // 8
    off = 1 << (nbits - 1)
    return np.array([int.from_bytes(data[i * nb:(i + 1) * nb], "little") - off for i in range(count)], dtype=object)


def big_rice_encode(values, k):
    out = []
    mask = (1 << k) - 1
    for x in np.ravel(values):
        x = int(x)
        a = -x if x < 0 else x
        out.append("1" if x < 0 else "0")
        out.append(format(a & mask, "0%db" % k) if k > 0 else "")
        out.append("0" * (a >> k) + "1")
    s = "".join(out)
    nbits = len(s)
    s += "0" * ((-nbits) % 8)
    return int(s, 2).to_bytes(len(s) // 8, "big") if s else b"", nbits


def big_rice_decode(data, count, k):
    s = bin(int.from_bytes(data, "big"))[2:].zfill(8 * len(data))
    out = np.empty(count, dtype=object)
    pos = 0
    for i in range(count):
        sign = s[pos] == "1"
        lo = int(s[pos + 1: pos + 1 + k], 2) if k > 0 else 0
        start = pos + 1 + k
        end = s.index("1", start)
        mag = ((end - start) << k) | lo
        out[i] = -mag if sign else mag
        pos = end + 1
    return out
