from __future__ import annotations

r"""
Audit LaTeX images actually used by a thesis project.

The script scans the LaTeX source starting from main.tex, follows \input/\include
commands, extracts \includegraphics references, resolves them on disk, and writes
CSV/JSON reports for:
  - images referenced by LaTeX and successfully resolved;
  - image references that are missing or ambiguous;
  - image files present in the image folders but not referenced by the compiled thesis.

Suggested command from the repository root:
    python datasets/scripts/audit/audit_latex_images_used.py --main docs/LatexThesis/main.tex

The script has no mandatory third-party dependencies. If Pillow is installed, it also
records raster image dimensions.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = [
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".eps",
    ".svg",
    ".tif",
    ".tiff",
]

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "build",
    "out",
    "output",
    "aux",
    "_minted-main",
    "audit",
    "latex_image_audit",
}

INCLUDEGRAPHICS_RE = re.compile(
    r"\\includegraphics\s*(?:\[[^\]]*\]\s*)?\{(?P<path>[^{}]+)\}",
    re.MULTILINE,
)

# Common LaTeX file inclusion commands. This deliberately ignores commented lines.
TEX_INCLUDE_RE = re.compile(
    r"\\(?:input|include|subfile)\s*\{(?P<path>[^{}]+)\}",
    re.MULTILINE,
)

TEX_IMPORT_RE = re.compile(
    r"\\(?:import|subimport)\s*\{(?P<dir>[^{}]*)\}\s*\{(?P<path>[^{}]+)\}",
    re.MULTILINE,
)

GRAPHICSPATH_RE = re.compile(
    r"\\graphicspath\s*\{(?P<body>(?:\{[^{}]+\}\s*)+)\}",
    re.MULTILINE,
)

BRACED_PATH_RE = re.compile(r"\{([^{}]+)\}")

CAPTION_RE = re.compile(r"\\caption(?:\[[^\]]*\])?\s*\{(?P<caption>.*?)\}", re.DOTALL)
LABEL_RE = re.compile(r"\\label\s*\{(?P<label>[^{}]+)\}")


@dataclass
class ImageReference:
    raw_path: str
    source_tex: str
    line: int
    exists: bool
    resolved_path: str
    status: str
    sha256: str
    size_bytes: int | None
    width_px: int | None
    height_px: int | None
    caption: str
    label: str


@dataclass
class ImageInventoryRow:
    path: str
    used: bool
    sha256: str
    size_bytes: int | None
    width_px: int | None
    height_px: int | None


def find_repo_root(start: Path) -> Path:
    """Find a likely repository root, compatible with the FAIR-Lab thesis layout."""
    start = start.resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / "docs" / "LatexThesis").is_dir() and (candidate / "datasets" / "scripts").is_dir():
            return candidate
        if (candidate / "docs" / "LatexThesis" / "main.tex").is_file():
            return candidate
        if (candidate / "main.tex").is_file() and candidate.name.lower() in {"latexthesis", "thesis"}:
            return candidate.parent.parent if candidate.parent.name == "docs" else candidate
    raise RuntimeError(
        "Could not determine repository root. Run the script from the repository root "
        "or pass --main with an explicit main.tex path."
    )


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def strip_latex_comments(text: str) -> str:
    """Remove LaTeX comments while preserving line count."""
    cleaned_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        cut_at: int | None = None
        i = 0
        while i < len(line):
            if line[i] == "%":
                # Count consecutive backslashes before %. Odd count means escaped percent.
                slash_count = 0
                j = i - 1
                while j >= 0 and line[j] == "\\":
                    slash_count += 1
                    j -= 1
                if slash_count % 2 == 0:
                    cut_at = i
                    break
            i += 1
        if cut_at is None:
            cleaned_lines.append(line)
        else:
            # Preserve newline if present.
            newline = "\n" if line.endswith("\n") else ""
            cleaned_lines.append(line[:cut_at] + newline)
    return "".join(cleaned_lines)


def line_number_at(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace")


def normalize_tex_path(raw_path: str) -> str:
    path = raw_path.strip()
    if not path.lower().endswith(".tex"):
        path += ".tex"
    return path


def resolve_tex_include(raw_path: str, source_tex: Path, thesis_dir: Path) -> Path | None:
    norm = normalize_tex_path(raw_path)
    candidates = [
        (source_tex.parent / norm),
        (thesis_dir / norm),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def collect_tex_files(main_tex: Path, thesis_dir: Path, scan_all_tex: bool) -> list[Path]:
    if scan_all_tex:
        return sorted(
            p.resolve()
            for p in thesis_dir.rglob("*.tex")
            if not any(part in DEFAULT_EXCLUDED_DIRS for part in p.parts)
        )

    visited: set[Path] = set()
    queue: deque[Path] = deque([main_tex.resolve()])

    while queue:
        tex = queue.popleft().resolve()
        if tex in visited or not tex.is_file():
            continue
        visited.add(tex)

        cleaned = strip_latex_comments(read_text(tex))

        for match in TEX_INCLUDE_RE.finditer(cleaned):
            child = resolve_tex_include(match.group("path"), tex, thesis_dir)
            if child and child not in visited:
                queue.append(child)

        for match in TEX_IMPORT_RE.finditer(cleaned):
            base_dir = match.group("dir").strip()
            raw_file = match.group("path").strip()
            norm_file = normalize_tex_path(raw_file)
            candidates = [
                tex.parent / base_dir / norm_file,
                thesis_dir / base_dir / norm_file,
            ]
            for candidate in candidates:
                if candidate.is_file():
                    queue.append(candidate.resolve())
                    break

    return sorted(visited)


def collect_graphicspaths(tex_files: Iterable[Path], thesis_dir: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    for tex in tex_files:
        cleaned = strip_latex_comments(read_text(tex))
        for match in GRAPHICSPATH_RE.finditer(cleaned):
            for raw in BRACED_PATH_RE.findall(match.group("body")):
                raw = raw.strip()
                for candidate in (thesis_dir / raw, tex.parent / raw):
                    candidate = candidate.resolve()
                    if candidate not in seen:
                        paths.append(candidate)
                        seen.add(candidate)

    # Useful conventional folders, even when \graphicspath is not declared.
    for conventional in ("images", "figures", "assets", "img", "plots"):
        candidate = (thesis_dir / conventional).resolve()
        if candidate not in seen:
            paths.append(candidate)
            seen.add(candidate)

    return paths


def image_candidates(raw_path: str, source_tex: Path, thesis_dir: Path, graphicspaths: list[Path]) -> list[Path]:
    raw = raw_path.strip().replace("\\", "/")
    raw_path_obj = Path(raw)

    bases = [source_tex.parent, thesis_dir, *graphicspaths]
    candidate_paths: list[Path] = []

    if raw_path_obj.is_absolute():
        base_candidates = [raw_path_obj]
    else:
        base_candidates = [base / raw_path_obj for base in bases]

    for candidate in base_candidates:
        if candidate.suffix:
            candidate_paths.append(candidate)
        else:
            candidate_paths.extend(candidate.with_suffix(ext) for ext in IMAGE_EXTENSIONS)

    # De-duplicate while preserving order.
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidate_paths:
        resolved = candidate.resolve()
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return deduped


def resolve_image(raw_path: str, source_tex: Path, thesis_dir: Path, graphicspaths: list[Path]) -> tuple[str, Path | None]:
    matches = [p for p in image_candidates(raw_path, source_tex, thesis_dir, graphicspaths) if p.is_file()]
    if not matches:
        return "missing", None
    unique = []
    seen = set()
    for p in matches:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    if len(unique) > 1:
        # Usually LaTeX will use the first one according to extension/search order.
        return "ambiguous_first_match_used", unique[0]
    return "ok", unique[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_dimensions(path: Path) -> tuple[int | None, int | None]:
    if path.suffix.lower() == ".pdf":
        return None, None
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None, None


def file_metadata(path: Path | None) -> tuple[str, int | None, int | None, int | None]:
    if path is None or not path.is_file():
        return "", None, None, None
    width, height = image_dimensions(path)
    return sha256_file(path), path.stat().st_size, width, height


def compact_latex_text(text: str, max_len: int = 160) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) > max_len:
        compact = compact[: max_len - 1].rstrip() + "…"
    return compact


def extract_nearby_caption_and_label(cleaned_tex: str, match_start: int, match_end: int) -> tuple[str, str]:
    """Best-effort extraction of caption/label around an includegraphics command."""
    window_start = max(0, match_start - 3000)
    window_end = min(len(cleaned_tex), match_end + 3000)
    window = cleaned_tex[window_start:window_end]

    captions = list(CAPTION_RE.finditer(window))
    labels = list(LABEL_RE.finditer(window))

    caption = ""
    label = ""

    if captions:
        # Prefer first caption after image, otherwise nearest caption.
        local_image_end = match_end - window_start
        after = [m for m in captions if m.start() >= local_image_end]
        chosen = after[0] if after else min(captions, key=lambda m: abs(m.start() - local_image_end))
        caption = compact_latex_text(chosen.group("caption"))

    if labels:
        local_image_end = match_end - window_start
        after = [m for m in labels if m.start() >= local_image_end]
        chosen = after[0] if after else min(labels, key=lambda m: abs(m.start() - local_image_end))
        label = chosen.group("label").strip()

    return caption, label


def extract_image_references(tex_files: Iterable[Path], thesis_dir: Path, repo_root: Path, graphicspaths: list[Path]) -> list[ImageReference]:
    refs: list[ImageReference] = []

    for tex in tex_files:
        raw_text = read_text(tex)
        cleaned = strip_latex_comments(raw_text)
        for match in INCLUDEGRAPHICS_RE.finditer(cleaned):
            raw_path = match.group("path").strip()
            status, resolved = resolve_image(raw_path, tex, thesis_dir, graphicspaths)
            sha, size_bytes, width, height = file_metadata(resolved)
            caption, label = extract_nearby_caption_and_label(cleaned, match.start(), match.end())

            refs.append(
                ImageReference(
                    raw_path=raw_path,
                    source_tex=rel(tex, repo_root),
                    line=line_number_at(cleaned, match.start()),
                    exists=resolved is not None,
                    resolved_path=rel(resolved, repo_root) if resolved else "",
                    status=status,
                    sha256=sha,
                    size_bytes=size_bytes,
                    width_px=width,
                    height_px=height,
                    caption=caption,
                    label=label,
                )
            )

    return refs


def discover_image_roots(thesis_dir: Path, explicit_roots: list[str] | None) -> list[Path]:
    if explicit_roots:
        return [Path(root).expanduser().resolve() for root in explicit_roots]

    roots = []
    for name in ("images", "figures", "assets", "img", "plots"):
        candidate = thesis_dir / name
        if candidate.is_dir():
            roots.append(candidate.resolve())

    # Fallback: scan thesis dir if no conventional image folder exists.
    if not roots:
        roots.append(thesis_dir.resolve())

    return roots


def iter_image_files(image_roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in image_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in DEFAULT_EXCLUDED_DIRS for part in path.parts):
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def build_inventory(image_roots: list[Path], used_paths: set[Path], repo_root: Path) -> list[ImageInventoryRow]:
    rows: list[ImageInventoryRow] = []
    for path in sorted(iter_image_files(image_roots)):
        sha, size_bytes, width, height = file_metadata(path)
        rows.append(
            ImageInventoryRow(
                path=rel(path, repo_root),
                used=path.resolve() in used_paths,
                sha256=sha,
                size_bytes=size_bytes,
                width_px=width,
                height_px=height,
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit images effectively referenced by a LaTeX thesis."
    )
    parser.add_argument(
        "--main",
        type=str,
        default=None,
        help="Path to main.tex. Default: docs/LatexThesis/main.tex from repository root.",
    )
    parser.add_argument(
        "--thesis-dir",
        type=str,
        default=None,
        help="Path to the LaTeX thesis directory. Default: parent of --main.",
    )
    parser.add_argument(
        "--image-root",
        action="append",
        default=None,
        help="Image root to inventory. Can be passed multiple times. Default: images/figures/assets/img/plots under thesis dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for audit reports. Default: results/latex_image_audit under repo root.",
    )
    parser.add_argument(
        "--scan-all-tex",
        action="store_true",
        help="Scan every .tex under thesis dir instead of following only files included from main.tex.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if missing image references are found.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.main:
            main_tex = Path(args.main).expanduser().resolve()
            # Derive a sensible repo root from main path first, then cwd.
            try:
                repo_root = find_repo_root(main_tex.parent)
            except RuntimeError:
                repo_root = find_repo_root(Path.cwd())
        else:
            repo_root = find_repo_root(Path.cwd())
            main_tex = (repo_root / "docs" / "LatexThesis" / "main.tex").resolve()

        thesis_dir = Path(args.thesis_dir).expanduser().resolve() if args.thesis_dir else main_tex.parent.resolve()
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (repo_root / "results" / "latex_image_audit").resolve()

        if not main_tex.is_file():
            raise FileNotFoundError(f"main.tex not found: {main_tex}")
        if not thesis_dir.is_dir():
            raise FileNotFoundError(f"thesis directory not found: {thesis_dir}")

        tex_files = collect_tex_files(main_tex, thesis_dir, args.scan_all_tex)
        graphicspaths = collect_graphicspaths(tex_files, thesis_dir)
        references = extract_image_references(tex_files, thesis_dir, repo_root, graphicspaths)

        used_paths = {
            (repo_root / ref.resolved_path).resolve()
            for ref in references
            if ref.exists and ref.resolved_path
        }

        image_roots = discover_image_roots(thesis_dir, args.image_root)
        inventory = build_inventory(image_roots, used_paths, repo_root)

        missing_refs = [ref for ref in references if not ref.exists]
        ambiguous_refs = [ref for ref in references if ref.status == "ambiguous_first_match_used"]
        unused_images = [row for row in inventory if not row.used]

        duplicate_hashes: dict[str, list[str]] = defaultdict(list)
        for row in inventory:
            if row.sha256:
                duplicate_hashes[row.sha256].append(row.path)
        duplicate_images = [
            {"sha256": sha, "paths": paths}
            for sha, paths in sorted(duplicate_hashes.items())
            if len(paths) > 1
        ]

        output_dir.mkdir(parents=True, exist_ok=True)

        write_csv(output_dir / "latex_image_references.csv", [asdict(r) for r in references])
        write_csv(output_dir / "latex_image_missing.csv", [asdict(r) for r in missing_refs])
        write_csv(output_dir / "latex_image_unused.csv", [asdict(r) for r in unused_images])
        write_csv(output_dir / "latex_image_inventory.csv", [asdict(r) for r in inventory])

        summary = {
            "repo_root": rel(repo_root, repo_root),
            "main_tex": rel(main_tex, repo_root),
            "thesis_dir": rel(thesis_dir, repo_root),
            "output_dir": rel(output_dir, repo_root),
            "scan_all_tex": bool(args.scan_all_tex),
            "tex_files_scanned_count": len(tex_files),
            "tex_files_scanned": [rel(p, repo_root) for p in tex_files],
            "graphicspaths": [rel(p, repo_root) for p in graphicspaths],
            "image_roots": [rel(p, repo_root) for p in image_roots],
            "includegraphics_references_count": len(references),
            "resolved_references_count": sum(1 for r in references if r.exists),
            "missing_references_count": len(missing_refs),
            "ambiguous_references_count": len(ambiguous_refs),
            "unique_used_images_count": len(used_paths),
            "inventory_images_count": len(inventory),
            "unused_images_count": len(unused_images),
            "duplicate_image_groups_count": len(duplicate_images),
            "status": "FAIL_MISSING_IMAGES" if missing_refs else "OK",
        }

        (output_dir / "latex_image_audit_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "latex_image_duplicates.json").write_text(
            json.dumps(duplicate_images, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("\nLaTeX image audit completed")
        print(f"  main.tex:                 {rel(main_tex, repo_root)}")
        print(f"  tex files scanned:        {len(tex_files)}")
        print(f"  includegraphics found:    {len(references)}")
        print(f"  resolved references:      {summary['resolved_references_count']}")
        print(f"  missing references:       {len(missing_refs)}")
        print(f"  unique used images:       {len(used_paths)}")
        print(f"  inventory images:         {len(inventory)}")
        print(f"  unused images:            {len(unused_images)}")
        print(f"  duplicate image groups:   {len(duplicate_images)}")
        print(f"  reports:                  {rel(output_dir, repo_root)}")

        if missing_refs:
            print("\nMissing image references:")
            for ref in missing_refs[:20]:
                print(f"  - {ref.source_tex}:{ref.line} -> {ref.raw_path}")
            if len(missing_refs) > 20:
                print(f"  ... and {len(missing_refs) - 20} more")

        if args.strict and missing_refs:
            return 2
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
