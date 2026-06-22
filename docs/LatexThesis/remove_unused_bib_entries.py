from pathlib import Path
import csv
import shutil


SCRIPT_DIR = Path(__file__).resolve().parent

BIB_FILE = SCRIPT_DIR / "tesi.bib"
UNUSED_CSV = SCRIPT_DIR / "bibliography_unused_entries.csv"

FULL_BACKUP = SCRIPT_DIR / "tesi_full.bib"
SAFETY_BACKUP = SCRIPT_DIR / "tesi_before_unused_cleanup.bib"
REMOVED_REPORT = SCRIPT_DIR / "bibliography_removed_unused_entries.txt"


def load_unused_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Unused entries CSV not found: {csv_path}")

    keys = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if "bibkey" not in reader.fieldnames:
            raise ValueError(f"'bibkey' column not found in {csv_path}")

        for row in reader:
            key = row.get("bibkey", "").strip()
            status = row.get("status", "").strip()

            if key and (not status or status == "IN_BIB_NOT_CITED"):
                keys.add(key)

    return keys


def find_bib_entries(text: str) -> list[tuple[int, int, str]]:
    """
    Return entries as tuples:
        (start_index, end_index, bibkey)

    Uses brace balancing, so it is safer than a simple regex.
    """
    entries = []
    i = 0
    n = len(text)

    while i < n:
        at = text.find("@", i)
        if at == -1:
            break

        # Skip comments such as %@article{...}
        line_start = text.rfind("\n", 0, at) + 1
        before_at = text[line_start:at].strip()
        if before_at.startswith("%"):
            i = at + 1
            continue

        brace = text.find("{", at)
        comma = text.find(",", at)

        if brace == -1 or comma == -1 or comma < brace:
            i = at + 1
            continue

        entry_type = text[at + 1:brace].strip().lower()

        # Skip @comment, @preamble, @string entries
        if entry_type in {"comment", "preamble", "string"}:
            i = at + 1
            continue

        key = text[brace + 1:comma].strip()

        depth = 0
        end = None

        for j in range(brace, n):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break

        if end is None:
            raise ValueError(f"Unbalanced BibTeX entry starting at index {at}, key={key}")

        # Include trailing blank lines after the entry.
        k = end
        while k < n and text[k] in {" ", "\t", "\r", "\n"}:
            k += 1

        entries.append((at, k, key))
        i = k

    return entries


def remove_entries(text: str, keys_to_remove: set[str]) -> tuple[str, list[str]]:
    entries = find_bib_entries(text)

    chunks = []
    last = 0
    removed = []

    for start, end, key in entries:
        chunks.append(text[last:start])

        if key in keys_to_remove:
            removed.append(key)
            # Entry skipped.
        else:
            chunks.append(text[start:end])

        last = end

    chunks.append(text[last:])

    cleaned = "".join(chunks)

    # Normalize excessive blank lines.
    while "\n\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n\n", "\n\n\n")

    return cleaned.strip() + "\n", sorted(removed)


def main() -> None:
    if not BIB_FILE.exists():
        raise FileNotFoundError(f"BibTeX file not found: {BIB_FILE}")

    unused_keys = load_unused_keys(UNUSED_CSV)

    if not unused_keys:
        print("No unused BibTeX entries found. Nothing to remove.")
        return

    # Preserve a complete archive copy if it does not already exist.
    if not FULL_BACKUP.exists():
        shutil.copy2(BIB_FILE, FULL_BACKUP)
        print(f"Full backup created: {FULL_BACKUP}")
    else:
        print(f"Full backup already exists, left unchanged: {FULL_BACKUP}")

    # Always create a safety backup of the current file.
    shutil.copy2(BIB_FILE, SAFETY_BACKUP)
    print(f"Safety backup created: {SAFETY_BACKUP}")

    original_text = BIB_FILE.read_text(encoding="utf-8", errors="replace")
    cleaned_text, removed = remove_entries(original_text, unused_keys)

    BIB_FILE.write_text(cleaned_text, encoding="utf-8")

    REMOVED_REPORT.write_text("\n".join(removed) + "\n", encoding="utf-8")

    print()
    print("Unused BibTeX cleanup completed.")
    print(f"Requested unused keys: {len(unused_keys)}")
    print(f"Removed entries: {len(removed)}")
    print(f"Updated BibTeX file: {BIB_FILE}")
    print(f"Removed entries report: {REMOVED_REPORT}")

    missing_from_bib = sorted(unused_keys - set(removed))
    if missing_from_bib:
        print()
        print("Warning: these unused keys were listed in the CSV but not found in tesi.bib:")
        for key in missing_from_bib:
            print(f"  - {key}")


if __name__ == "__main__":
    main()