import requests
from pathlib import Path

urls = [
    "https://archive.org/download/dli.ministry.23880/E08106_The_Thousand_And_One_Nights_Vol_1_djvu.txt",
    "https://archive.org/download/dli.ministry.23880/E08107_The_Thousand_And_One_Nights_Vol_2_djvu.txt",
]

output = Path("one_thousand_and_one_nights_raw.txt")
with open(output, "w", encoding="utf-8") as f:

    for i, url in enumerate(urls, 1):
        print(f"Downloading volume {i}...")
        response = requests.get(
            url,
            timeout=120
        )

        response.raise_for_status()

        f.write(response.text)
        f.write("\n\n")

        print(f"Volume {i} downloaded: {len(response.text):,} characters")

print(f"\nSaved to: {output}")