from pathlib import Path

from tokenizers import Tokenizer


# ============================================================
# Load tokenizer
# ============================================================

TOKENIZER_FILE = Path(
    "tokenizer/one_thousand_and_one_nights_bpe.json"
)

tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))


print("=" * 70)
print("TOKENIZER TEST")
print("=" * 70)

print(f"Vocabulary size: {tokenizer.get_vocab_size()}")


# ============================================================
# Test sentences
# ============================================================

test_texts = [
    "The king went to the palace.",
    "In the name of God, the Compassionate, the Merciful.",
    "She was beautiful and wise.",
    "The king's kingdom was vast.",
    "He proceeded day and night.",
    "Shahrazad began to tell her story.",
    "Hello!",
    "Hello, world.",
    "The king's palace.",
    "He said: 'Go away!'",
    "What is this?",
    "One-two-three."
]


# ============================================================
# Encode + Decode
# ============================================================

for i, text in enumerate(test_texts, start=1):

    encoding = tokenizer.encode(text)

    token_ids = encoding.ids
    tokens = encoding.tokens

    decoded = tokenizer.decode(token_ids)

    print("\n" + "-" * 70)
    print(f"TEST {i}")
    print("-" * 70)

    print(f"Original:")
    print(text)

    print("\nTokens:")
    print(tokens)

    print("\nToken IDs:")
    print(token_ids)

    print(f"\nNumber of tokens: {len(token_ids)}")

    print("\nDecoded:")
    print(decoded)

    print("\nExact match:")
    print(text == decoded)