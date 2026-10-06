from pathlib import Path
import re


INPUT_FILE = Path("one_thousand_and_one_nights_raw.txt")
OUTPUT_FILE = Path("processed/one_thousand_and_one_nights_clean.txt")


#Load the text from a file
def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")




def remove_front_matter(lines: list[str]) -> list[str]:
    """
    The actual literary text starts at:

        INTRODUCTION.

    Everything before that belongs to title pages,
    editor's preface, contents, illustrations, etc.
    """

    for i, line in enumerate(lines):

        if line.strip().upper() == "INTRODUCTION.":
            print(f"Found INTRODUCTION at line {i}")
            return lines[i:]

    raise ValueError("Could not find the beginning of the Introduction.")

def normalize_characters(text: str) -> str:

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def is_page_header(line: str) -> bool:

    line = line.strip()

    # Examples:
    # 2 INTRODUCTION.
    # 3 INTRODUCTION.
    # 4 INTRODUCTION.

    pattern = r"^\d+\s+[A-Z][A-Z\s'.-]+$"

    return bool(re.match(pattern, line))


def remove_page_headers(lines: list[str]) -> list[str]:

    cleaned = []
    removed = 0
    for line in lines:

        if is_page_header(line):
            removed += 1
            continue

        cleaned.append(line)

    print(f"Removed page headers: {removed}")
    return cleaned

def is_page_number(line: str) -> bool:

    line = line.strip()

    # Only remove lines consisting purely of digits.
    return bool(re.fullmatch(r"\d+", line))


def remove_page_numbers(lines: list[str]) -> list[str]:

    cleaned = []
    removed = 0

    for line in lines:
        if is_page_number(line):
            removed += 1
            continue

        cleaned.append(line)
    print(f"Removed page numbers: {removed}")
    return cleaned


def repair_hyphenation(text: str) -> str:

    # Example:
    #
    # Com-
    # passionate
    #
    # becomes:
    #
    # Compassionate

    text = re.sub(
        r"(\w+)-\s*\n\s*(\w+)",
        r"\1\2",
        text
    )

    return text


def reconstruct_paragraphs(lines: list[str]) -> str:

    paragraphs = []
    current = []
    for line in lines:

        line = line.strip()
        # Empty line = paragraph boundary
        if not line:

            if current:
                paragraphs.append(" ".join(current))
                current = []

            continue

        current.append(line)

    # Last paragraph
    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)


def normalize_whitespace(text: str) -> str:

    # Multiple spaces → one space
    text = re.sub(r"[ \t]+", " ", text)

    # Excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_text(raw_text: str) -> str:

    lines = raw_text.splitlines()

    print(f"Original lines: {len(lines):,}")

    # Front matter
    lines = remove_front_matter(lines)

    print(f"After front matter removal: {len(lines):,}")

    # Page headers
    lines = remove_page_headers(lines)

    # Page numbers
    lines = remove_page_numbers(lines)

    # Reconstruct text
    text = reconstruct_paragraphs(lines)

    # Repair OCR line-break hyphenation
    text = repair_hyphenation(text)

    # Normalize characters
    text = normalize_characters(text)

    # Normalize whitespace
    text = normalize_whitespace(text)

    return text



def main():

    print("=" * 70)
    print("THOUSAND AND ONE NIGHTS: PREPROCESSING")
    print("=" * 70)

    raw_text = load_text(INPUT_FILE)

    print(f"Raw characters: {len(raw_text):,}")
    print(f"Raw words:      {len(raw_text.split()):,}")

    print("\nCleaning...")

    cleaned_text = clean_text(raw_text)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"Clean characters: {len(cleaned_text):,}")
    print(f"Clean words:      {len(cleaned_text.split()):,}")
    print(f"Clean lines:      {len(cleaned_text.splitlines()):,}")

    OUTPUT_FILE.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT_FILE.write_text(cleaned_text,encoding="utf-8")
    print("\nSaved to:")
    print(OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()