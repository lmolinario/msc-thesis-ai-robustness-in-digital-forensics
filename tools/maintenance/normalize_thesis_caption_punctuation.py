from __future__ import annotations

from pathlib import Path

FILES = [
    Path('docs/LatexThesis/sections/02_background.tex'),
    Path('docs/LatexThesis/sections/03_StateoftheArt.tex'),
    Path('docs/LatexThesis/sections/08_appendix.tex'),
]
EXPECTED = {
    '02_background.tex': 6,
    '03_StateoftheArt.tex': 3,
    '08_appendix.tex': 2,
}


def normalize_captions(text: str) -> tuple[str, int]:
    out: list[str] = []
    cursor = 0
    changed = 0
    while True:
        start = text.find('\\caption', cursor)
        if start < 0:
            out.append(text[cursor:])
            break
        out.append(text[cursor:start])
        brace = text.find('{', start)
        if brace < 0:
            raise RuntimeError('Malformed caption command')
        depth = 0
        end = brace
        while end < len(text):
            char = text[end]
            escaped = end > 0 and text[end - 1] == '\\'
            if char == '{' and not escaped:
                depth += 1
            elif char == '}' and not escaped:
                depth -= 1
                if depth == 0:
                    break
            end += 1
        if depth != 0:
            raise RuntimeError('Unbalanced caption braces')
        segment = text[start:end + 1]
        inner = text[brace + 1:end]
        stripped = inner.rstrip()
        if stripped.endswith('.'):
            trailing = inner[len(stripped):]
            inner = stripped[:-1] + trailing
            segment = text[start:brace + 1] + inner + '}'
            changed += 1
        out.append(segment)
        cursor = end + 1
    return ''.join(out), changed


def main() -> None:
    total = 0
    for path in FILES:
        source = path.read_text(encoding='utf-8')
        updated, changed = normalize_captions(source)
        expected = EXPECTED[path.name]
        if changed != expected:
            raise RuntimeError(
                f'{path}: expected {expected} terminal periods, found {changed}'
            )
        path.write_text(updated, encoding='utf-8')
        total += changed
    if total != 11:
        raise RuntimeError(f'Expected 11 caption changes, found {total}')


if __name__ == '__main__':
    main()
