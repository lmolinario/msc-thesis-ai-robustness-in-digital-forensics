#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00_download_raw_datasets_bundle.py

Restore controlled-access image artifacts used by the FAIR-Lab thesis pipeline.
Image corpora are intentionally excluded from the public ``main`` branch.

Artifacts:
- ``raw`` (default): heterogeneous raw source bundle;
- ``frozen``: exact 11,500-file forensic evaluation bundle used for commercial
  black-box testing.

Recommended workflow:
1. ``--artifact <raw|frozen> --request-access``;
2. request access through the restricted storage page;
3. download the approved ZIP through the browser;
4. restore it with ``--artifact <...> --archive <path>``.

The complete ZIP is verified against the authoritative digests in
``docs/artifact/CONTROLLED_ARTIFACT_CHECKSUMS.sha256``. For the frozen bundle,
every blind input is additionally verified against the committed per-file hash
manifest.

Authorized direct-download URLs may alternatively be supplied through ``--url``
or the artifact-specific environment variable. Private or temporary URLs must
never be committed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import stat
import webbrowser
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import gdown

from datasets.scripts.utils.paths import (
    DATASETS_DIR,
    RAW_DATASETS_DIR,
    REPO_ROOT,
    repo_relative_path,
)

CHECKSUMS_PATH = REPO_ROOT / "docs" / "artifact" / "CONTROLLED_ARTIFACT_CHECKSUMS.sha256"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    label: str
    direct_url_env: str
    request_url_env: str
    default_request_url: str
    archive_path: Path
    extract_dir: Path
    expected_file_count: int | None = None
    allowed_archive_roots: tuple[str, ...] = ()
    replace_roots: tuple[str, ...] = ()
    hash_manifest_path: Path | None = None

    @property
    def archive_filename(self) -> str:
        return self.archive_path.name


RAW_ARTIFACT = ArtifactSpec(
    key="raw",
    label="raw dataset bundle",
    direct_url_env="FAIRLAB_RAW_DATASET_BUNDLE_URL",
    request_url_env="FAIRLAB_RAW_DATASET_BUNDLE_REQUEST_URL",
    default_request_url=(
        "https://drive.google.com/file/d/"
        "1yGbGZ3aFJRUZZQdSxrNlwY20Txa6KqbH/view?usp=drive_link"
    ),
    archive_path=RAW_DATASETS_DIR / "downloaded_raw_archives" / "00_raw_datasets_bundle.zip",
    extract_dir=RAW_DATASETS_DIR / "downloaded_raw_archives" / "extracted_bundle",
)

FROZEN_BUNDLE_DIR = DATASETS_DIR / "forensic_evaluation_bundle"
FROZEN_ARTIFACT = ArtifactSpec(
    key="frozen",
    label="frozen forensic evaluation bundle",
    direct_url_env="FAIRLAB_FROZEN_FORENSIC_EVALUATION_BUNDLE_URL",
    request_url_env="FAIRLAB_FROZEN_FORENSIC_EVALUATION_BUNDLE_REQUEST_URL",
    # Add the stable public request page here after the controlled upload is frozen.
    default_request_url="",
    archive_path=(
        REPO_ROOT
        / "downloads"
        / "controlled_artifacts"
        / "16_frozen_forensic_evaluation_bundle.zip"
    ),
    extract_dir=FROZEN_BUNDLE_DIR,
    expected_file_count=11_500,
    allowed_archive_roots=("blind_tool_input", "structured_audit_view", "metadata"),
    replace_roots=("blind_tool_input", "structured_audit_view"),
    hash_manifest_path=FROZEN_BUNDLE_DIR / "metadata" / "bundle_hashes_sha256.csv",
)

ARTIFACTS = {spec.key: spec for spec in (RAW_ARTIFACT, FROZEN_ARTIFACT)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore a controlled-access FAIR-Lab image artifact."
    )
    parser.add_argument(
        "--artifact",
        choices=tuple(ARTIFACTS),
        default="raw",
        help="Artifact to restore. Default: raw.",
    )
    parser.add_argument(
        "--request-access",
        action="store_true",
        help="Open the selected artifact's restricted access-request page.",
    )
    parser.add_argument(
        "--request-page",
        default="",
        help="Stable storage view-page override for the selected artifact.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Browser-downloaded ZIP to validate, hash, extract, and verify.",
    )
    parser.add_argument(
        "--url",
        default="",
        help="Authorized direct-download URL; otherwise use the artifact environment variable.",
    )
    parser.add_argument(
        "--expected-sha256",
        default="",
        help=(
            "Explicit archive SHA-256 override. When omitted, the script reads "
            "docs/artifact/CONTROLLED_ARTIFACT_CHECKSUMS.sha256."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace an existing locally downloaded archive.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help=(
            "Replace existing local output. For the frozen artifact, only the "
            "non-public image views are replaced; metadata are preserved."
        ),
    )
    parser.add_argument(
        "--skip-content-verification",
        action="store_true",
        help="Skip frozen per-file SHA-256 verification (diagnostic use only).",
    )
    return parser


def resolve_request_page(spec: ArtifactSpec, cli_page: str) -> tuple[str, str]:
    if cli_page.strip():
        return cli_page.strip(), "command line"
    env_value = os.getenv(spec.request_url_env, "").strip()
    if env_value:
        return env_value, f"environment variable {spec.request_url_env}"
    if spec.default_request_url:
        return spec.default_request_url, "repository configuration"
    raise RuntimeError(
        f"No request page is configured for the {spec.label}. Use --request-page "
        f"or {spec.request_url_env}. Do not commit signed or temporary URLs."
    )


def open_access_request_page(spec: ArtifactSpec, cli_page: str) -> None:
    request_page, source = resolve_request_page(spec, cli_page)
    print(f"[INFO] Opening access page for the {spec.label} ({source}):")
    print(f"       {request_page}")
    if not webbrowser.open(request_page, new=2):
        print("[WARN] Browser launch failed. Copy the URL above and request access.")


def resolve_bundle_url(spec: ArtifactSpec, cli_url: str) -> tuple[str, str]:
    if cli_url.strip():
        return cli_url.strip(), "command line"
    env_value = os.getenv(spec.direct_url_env, "").strip()
    if env_value:
        return env_value, f"environment variable {spec.direct_url_env}"
    raise RuntimeError(
        f"No authorized direct-download URL was provided for the {spec.label}. "
        "Use --request-access and then --archive, or provide --url or "
        f"{spec.direct_url_env}."
    )


def normalize_sha256(value: str, source: str) -> str:
    digest = value.strip().lower()
    if not SHA256_RE.fullmatch(digest):
        raise RuntimeError(f"Invalid SHA-256 digest from {source}: {value!r}")
    return digest


def load_authoritative_checksums(path: Path) -> dict[str, str]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Authoritative checksum file not found: {path}")

    checksums: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise RuntimeError(f"Malformed checksum line {line_number} in {path}: {line}")
            digest = normalize_sha256(parts[0], f"{path}:{line_number}")
            filename = parts[1].lstrip("*").strip()
            filename = PurePosixPath(filename.replace("\\", "/")).name
            if not filename:
                raise RuntimeError(f"Missing filename on checksum line {line_number} in {path}")
            if filename in checksums:
                raise RuntimeError(f"Duplicate checksum filename in {path}: {filename}")
            checksums[filename] = digest

    if not checksums:
        raise RuntimeError(f"No checksums found in: {path}")
    return checksums


def resolve_expected_archive_digest(
    spec: ArtifactSpec,
    cli_expected: str,
) -> tuple[str, str]:
    if cli_expected.strip():
        return normalize_sha256(cli_expected, "--expected-sha256"), "command line override"

    checksums = load_authoritative_checksums(CHECKSUMS_PATH)
    try:
        return checksums[spec.archive_filename], str(CHECKSUMS_PATH)
    except KeyError as exc:
        raise RuntimeError(
            f"No authoritative checksum is registered for {spec.archive_filename} in "
            f"{CHECKSUMS_PATH}."
        ) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive_digest(path: Path, expected_sha256: str) -> str:
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Archive SHA-256 mismatch: expected={expected_sha256}, actual={actual}, path={path}"
        )
    return actual


def validate_zip_archive(path: Path) -> None:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Archive is missing or empty: {path}")
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"The supplied file is not a valid ZIP archive: {path}")
    with zipfile.ZipFile(path, "r") as archive:
        corrupt_member = archive.testzip()
        if corrupt_member:
            raise RuntimeError(f"Corrupt ZIP member detected: {corrupt_member}")


def validate_not_symlink(member: zipfile.ZipInfo) -> None:
    unix_mode = member.external_attr >> 16
    if unix_mode and stat.S_ISLNK(unix_mode):
        raise RuntimeError(f"Symbolic links are not allowed: {member.filename}")


def safe_destination(member_name: str, extract_dir: Path) -> Path:
    member_path = PurePosixPath(member_name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise RuntimeError(f"Unsafe archive path: {member_name}")
    root = extract_dir.resolve()
    destination = (root / Path(*member_path.parts)).resolve()
    if destination != root and root not in destination.parents:
        raise RuntimeError(f"Unsafe archive path: {member_name}")
    return destination


def extract_raw_archive(archive_path: Path, extract_dir: Path, force: bool) -> None:
    if extract_dir.exists() and any(extract_dir.iterdir()):
        if not force:
            print(f"[SKIP] Extraction directory already populated: {extract_dir}")
            return
        print(f"[INFO] Removing existing extraction directory: {extract_dir}")
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            validate_not_symlink(member)
            safe_destination(member.filename, extract_dir)
        archive.extractall(extract_dir)
    print(f"[OK] Raw archive extracted to: {extract_dir}")


def detect_common_prefix(
    members: list[zipfile.ZipInfo],
    allowed_roots: tuple[str, ...],
) -> tuple[str, ...]:
    prefixes: set[tuple[str, ...]] = set()
    member_paths: list[tuple[str, ...]] = []

    for member in members:
        parts = PurePosixPath(member.filename).parts
        if not parts:
            continue
        member_paths.append(parts)
        indexes = [index for index, part in enumerate(parts) if part in allowed_roots]
        if indexes:
            prefixes.add(tuple(parts[: indexes[0]]))

    if not prefixes:
        raise RuntimeError(
            "Frozen archive contains none of the expected roots: "
            + ", ".join(allowed_roots)
        )
    if len(prefixes) != 1:
        rendered = ["/".join(prefix) or "<archive-root>" for prefix in prefixes]
        raise RuntimeError("Inconsistent frozen archive prefixes: " + ", ".join(sorted(rendered)))

    prefix = next(iter(prefixes))
    for parts in member_paths:
        if any(part in allowed_roots for part in parts):
            continue
        if len(parts) <= len(prefix) and tuple(parts) == prefix[: len(parts)]:
            continue
        raise RuntimeError(
            "Unexpected frozen archive member: " + PurePosixPath(*parts).as_posix()
        )
    return prefix


def normalize_frozen_name(
    member: zipfile.ZipInfo,
    prefix: tuple[str, ...],
    allowed_roots: tuple[str, ...],
) -> str:
    parts = list(PurePosixPath(member.filename).parts)
    if prefix and tuple(parts[: len(prefix)]) == prefix:
        parts = parts[len(prefix) :]
    if not parts:
        return ""
    if parts[0] not in allowed_roots:
        raise RuntimeError(f"Unexpected frozen archive path: {member.filename}")
    return PurePosixPath(*parts).as_posix()


def prepare_frozen_roots(spec: ArtifactSpec, force: bool) -> None:
    populated = [
        spec.extract_dir / root
        for root in spec.replace_roots
        if (spec.extract_dir / root).exists() and any((spec.extract_dir / root).iterdir())
    ]
    if populated and not force:
        raise FileExistsError(
            "Frozen output already exists. Use --force-extract to replace only: "
            + ", ".join(str(path) for path in populated)
        )
    if force:
        for root in spec.replace_roots:
            path = spec.extract_dir / root
            if path.exists():
                print(f"[INFO] Removing existing controlled output: {path}")
                shutil.rmtree(path)


def extract_frozen_archive(archive_path: Path, spec: ArtifactSpec, force: bool) -> None:
    prepare_frozen_roots(spec, force)
    spec.extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        members = archive.infolist()
        prefix = detect_common_prefix(members, spec.allowed_archive_roots)

        for member in members:
            validate_not_symlink(member)
            name = normalize_frozen_name(member, prefix, spec.allowed_archive_roots)
            if not name:
                continue
            parts = PurePosixPath(name).parts
            if parts[0] == "metadata":
                # Archive metadata are accepted for packaging convenience, but the
                # committed public metadata remain authoritative and are not overwritten.
                continue

            destination = safe_destination(name, spec.extract_dir)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)

    print(f"[OK] Frozen bundle restored under: {spec.extract_dir}")


def load_expected_hashes(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Frozen hash manifest not found: {path}")

    expected: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"tool_input_filename", "sha256"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Missing columns in {path}: {sorted(missing)}")
        for row in reader:
            filename = str(row.get("tool_input_filename", "")).strip()
            sha256 = normalize_sha256(
                str(row.get("sha256", "")),
                f"{path}:{filename or '<missing filename>'}",
            )
            if not filename:
                raise RuntimeError(f"Incomplete frozen hash row: {row}")
            if filename in expected:
                raise RuntimeError(f"Duplicate filename in frozen hash manifest: {filename}")
            expected[filename] = sha256
    return expected


def verify_frozen_bundle(spec: ArtifactSpec) -> None:
    if spec.hash_manifest_path is None:
        raise RuntimeError("Frozen hash manifest is not configured.")

    blind_dir = spec.extract_dir / "blind_tool_input" / "files"
    if not blind_dir.is_dir():
        raise FileNotFoundError(f"Frozen blind input directory not found: {blind_dir}")

    expected = load_expected_hashes(spec.hash_manifest_path)
    actual_paths = sorted(path for path in blind_dir.iterdir() if path.is_file())
    actual_names = {path.name for path in actual_paths}
    expected_names = set(expected)

    if spec.expected_file_count is not None:
        if len(expected) != spec.expected_file_count:
            raise RuntimeError(
                f"Unexpected frozen manifest size: expected={spec.expected_file_count}, actual={len(expected)}"
            )
        if len(actual_paths) != spec.expected_file_count:
            raise RuntimeError(
                f"Unexpected restored file count: expected={spec.expected_file_count}, actual={len(actual_paths)}"
            )

    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise RuntimeError(
            f"Frozen filename mismatch: missing={len(missing)} {missing[:10]}; "
            f"unexpected={len(unexpected)} {unexpected[:10]}"
        )

    mismatches: list[str] = []
    for index, path in enumerate(actual_paths, start=1):
        actual = sha256_file(path)
        if actual.lower() != expected[path.name]:
            mismatches.append(
                f"{path.name}: expected={expected[path.name]}, actual={actual}"
            )
        if index % 1000 == 0:
            print(f"[INFO] Verified {index}/{len(actual_paths)} frozen files...")

    if mismatches:
        raise RuntimeError(
            f"Frozen content hash mismatches: {len(mismatches)}\n"
            + "\n".join(mismatches[:20])
        )
    print(f"[OK] Frozen bundle verified: {len(actual_paths)} files match the SHA-256 manifest.")


def extract_archive(
    archive_path: Path,
    spec: ArtifactSpec,
    force: bool,
    verify_content: bool,
) -> None:
    if spec.key == "frozen":
        extract_frozen_archive(archive_path, spec, force)
        if verify_content:
            verify_frozen_bundle(spec)
        else:
            print("[WARN] Frozen per-file verification was explicitly skipped.")
        return
    extract_raw_archive(archive_path, repo_relative_path(spec.extract_dir), force)


def download_archive(url: str, archive_path: Path, force: bool) -> None:
    if archive_path.exists() and archive_path.stat().st_size > 0:
        if not force:
            validate_zip_archive(archive_path)
            print(f"[SKIP] Valid archive already exists: {archive_path}")
            return
        archive_path.unlink()

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = gdown.download(url=url, output=str(archive_path), quiet=False)
    except Exception as exc:
        raise RuntimeError(
            "Direct download failed. Complete the controlled-access browser flow, "
            "download the ZIP, and use --archive."
        ) from exc
    if result is None:
        raise RuntimeError("Direct download returned no file; use the browser and --archive.")

    validate_zip_archive(archive_path)
    print(f"[OK] Download completed: {archive_path}")


def main() -> None:
    args = build_parser().parse_args()
    spec = ARTIFACTS[args.artifact]

    if args.request_access:
        open_access_request_page(spec, args.request_page)
        no_restore_source = (
            args.archive is None
            and not args.url.strip()
            and not os.getenv(spec.direct_url_env, "").strip()
        )
        if no_restore_source:
            print("[DONE] Submit the request and rerun after authorization.")
            return

    if args.archive is not None:
        archive_path = args.archive.expanduser().resolve()
        validate_zip_archive(archive_path)
    else:
        bundle_url, source = resolve_bundle_url(spec, args.url)
        archive_path = repo_relative_path(spec.archive_path)
        print(f"[INFO] Selected artifact: {spec.label}")
        print(f"[INFO] Authorized URL source: {source}")
        download_archive(bundle_url, archive_path, args.force_download)

    expected_digest, digest_source = resolve_expected_archive_digest(
        spec,
        args.expected_sha256,
    )
    print(f"[INFO] Expected archive SHA-256 source: {digest_source}")
    digest = validate_archive_digest(archive_path, expected_digest)
    print(f"[INFO] Archive: {archive_path}")
    print(f"[OK] Archive SHA-256 verified: {digest}")

    extract_archive(
        archive_path,
        spec,
        args.force_extract,
        verify_content=not args.skip_content_verification,
    )
    print(f"[DONE] Controlled-access {spec.label} restoration completed.")


if __name__ == "__main__":
    main()
