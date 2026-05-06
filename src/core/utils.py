def fib_offsets(n):
    assert n > 1
    out = [1, 2]
    for i in range(n - 2):
        out.append(out[i] + out[i + 1])
    return out


def offsets_to_ranges(offsets: list[int]):
    prev: int = 1
    ranges: list[tuple[int, int]] = []
    for offset in offsets:
        ranges.append((prev, offset))
        prev = offset + 1
    return ranges
