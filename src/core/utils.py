def fib_offsets(n):
    assert n > 1
    out = [1, 2]
    for i in range(n - 2):
        out.append(out[i] + out[i + 1])
    return out
