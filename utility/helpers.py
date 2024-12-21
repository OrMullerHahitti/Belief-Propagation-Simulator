def num_to_letter(n: int) -> str:
    # 1 -> 'a', 2 -> 'b', ...
    return chr(ord('a') + (n - 1))

def letter_to_num(c: str) -> int:
    # 'a' -> 1, 'b' -> 2, ...
    return ord(c) - ord('a') + 1
