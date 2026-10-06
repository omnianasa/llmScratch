from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# path of a processed corpus file (cleaned text)

CORPUS_FILE = Path("../data/processed/one_thousand_and_one_nights_clean.txt")

TOKENIZER_DIR = Path("tokenizer")
TOKENIZER_FILE = TOKENIZER_DIR / "one_thousand_and_one_nights_bpe.json"

VOCAB_SIZE = 8000

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
]

# Load corpus
print("=" * 70)
print("LOADING CORPUS")
print("=" * 70)

text = CORPUS_FILE.read_text(encoding="utf-8")

print(f"Characters: {len(text):,}")
print(f"Words:      {len(text.split()):,}")

# Train BPE tokenizer
tokenizer = Tokenizer(
    BPE(unk_token="[UNK]")
)

tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False
)

tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    min_frequency=2,
    show_progress=True,
)


print("\n" + "=" * 70)
print("TRAINING BPE TOKENIZER")
print("=" * 70)

tokenizer.train(files=[str(CORPUS_FILE)],trainer=trainer,)

TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
tokenizer.save(str(TOKENIZER_FILE))


print("\n" + "=" * 70)
print("TOKENIZER SAVED")
print("=" * 70)

print(f"File: {TOKENIZER_FILE}")
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")


# Test the tokenizer
test_text = ("In the name of God, the Compassionate, ""the Merciful.")
encoding = tokenizer.encode(test_text)

print("\n" + "=" * 70)
print("TOKENIZATION TEST")
print("=" * 70)

print("Text:")
print(test_text)

print("\nTokens:")
print(encoding.tokens)

print("\nIDs:")
print(encoding.ids)

print("\nNumber of tokens:")
print(len(encoding.ids))

# Encode the entire corpus and save token IDs to a .npy file
print("\n" + "=" * 70)
print("ENCODING FULL CORPUS")
print("=" * 70)

full_encoding = tokenizer.encode(text)

token_ids = np.array(
    full_encoding.ids,
    dtype=np.uint16
)

print(f"Total tokens: {len(token_ids):,}")
print(f"Array dtype:  {token_ids.dtype}")
print(f"Memory:       {token_ids.nbytes / 1024 / 1024:.2f} MB")

# save token IDs to a .npy file

TOKENIZED_DIR = Path("data/tokenized")
TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_IDS_FILE = TOKENIZED_DIR / "corpus.npy"
np.save(TOKEN_IDS_FILE, token_ids)
print(f"\nSaved to: {TOKEN_IDS_FILE}")

# test decoding
decoded = tokenizer.decode(encoding.ids)

print("\nDecoded:")
print(decoded)