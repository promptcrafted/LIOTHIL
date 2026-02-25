"""CLI for the Dimljus encoding pipeline.

Commands:
    info           Show cache status for a training config
    cache-latents  Encode video/image targets through VAE (requires GPU)
    cache-text     Encode captions through T5 text encoder (requires GPU)

Usage::

    # Show what would be cached (no GPU needed)
    python -m dimljus.encoding info --config path/to/train.yaml

    # Encode latents (VAE — needs GPU)
    python -m dimljus.encoding cache-latents --config path/to/train.yaml

    # Encode text (T5 — needs GPU, run separately from latents)
    python -m dimljus.encoding cache-text --config path/to/train.yaml

The two-step caching design ensures VAE and T5 never compete for VRAM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Info command
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> None:
    """Show cache status for a training config.

    Discovers samples, expands them, and reports what would be cached.
    If a cache already exists, shows how many entries are complete vs stale.
    """
    from dimljus.config.training_loader import load_training_config
    from dimljus.encoding.cache import (
        CACHE_MANIFEST_FILENAME,
        find_missing_entries,
        find_stale_entries,
        load_cache_manifest,
    )
    from dimljus.encoding.discover import discover_samples
    from dimljus.encoding.expand import expand_samples
    from dimljus.encoding.bucket import bucket_groups

    config = load_training_config(args.config)

    # Resolve dataset directory from data_config
    data_config_path = Path(config.data_config)
    if data_config_path.is_file():
        dataset_dir = data_config_path.parent
    else:
        dataset_dir = data_config_path

    print(f"Training config: {args.config}")
    print(f"Dataset: {dataset_dir}")
    print(f"Cache dir: {config.cache.cache_dir}")
    print()

    # Discover
    samples = discover_samples(
        str(dataset_dir),
        probe=True,
    )
    print(f"Discovered: {len(samples)} source samples")

    if not samples:
        print("No samples found. Check your data_config path.")
        return

    # Expand
    expanded = expand_samples(
        samples,
        target_frames=config.cache.target_frames,
        frame_extraction=config.cache.frame_extraction,
        include_head_frame=config.cache.include_head_frame,
        step=config.cache.reso_step,
    )
    print(f"Expanded: {len(expanded)} training samples")

    # Bucket distribution
    groups = bucket_groups(expanded)
    print(f"Buckets: {len(groups)}")
    for key in sorted(groups.keys()):
        print(f"  {key}: {len(groups[key])} samples")

    # Check existing cache
    cache_dir = Path(config.cache.cache_dir)
    manifest_path = cache_dir / CACHE_MANIFEST_FILENAME

    if manifest_path.is_file():
        print()
        try:
            manifest = load_cache_manifest(cache_dir)
            print(f"Cache manifest: {manifest.total_entries} entries")
            print(f"  Latents: {manifest.latent_count}")
            print(f"  Text: {manifest.text_count}")
            print(f"  References: {manifest.reference_count}")

            stale = find_stale_entries(manifest)
            if stale:
                print(f"  Stale: {len(stale)} (source files changed)")

            missing = find_missing_entries(manifest, cache_dir)
            if missing:
                print(f"  Missing: {len(missing)} (cache files not on disk)")

            if not stale and not missing:
                print("  Status: all entries up to date")
        except Exception as e:
            print(f"  Error reading cache: {e}")
    else:
        print()
        print("No cache found. Run 'cache-latents' and 'cache-text' to build it.")


# ---------------------------------------------------------------------------
# Cache-latents command (placeholder for GPU encoding)
# ---------------------------------------------------------------------------

def cmd_cache_latents(args: argparse.Namespace) -> None:
    """Encode video/image targets through VAE and cache to disk."""
    from dimljus.config.training_loader import load_training_config
    from dimljus.encoding.cache import (
        build_cache_manifest,
        ensure_cache_dirs,
        save_cache_manifest,
    )
    from dimljus.encoding.discover import discover_samples
    from dimljus.encoding.expand import expand_samples

    config = load_training_config(args.config)

    data_config_path = Path(config.data_config)
    if data_config_path.is_file():
        dataset_dir = data_config_path.parent
    else:
        dataset_dir = data_config_path

    cache_dir = Path(config.cache.cache_dir)

    print(f"Cache directory: {cache_dir}")
    print(f"Dtype: {config.cache.dtype}")
    print(f"Frame counts: {config.cache.target_frames}")
    print()

    # Discover
    samples = discover_samples(str(dataset_dir), probe=True)
    print(f"Discovered: {len(samples)} source samples")

    if not samples:
        print("No samples found. Nothing to cache.")
        return

    # Expand
    expanded = expand_samples(
        samples,
        target_frames=config.cache.target_frames,
        frame_extraction=config.cache.frame_extraction,
        include_head_frame=config.cache.include_head_frame,
        step=config.cache.reso_step,
    )
    print(f"Expanded: {len(expanded)} training samples")

    # Create cache dirs
    ensure_cache_dirs(cache_dir)

    # Build manifest (without actual encoding — GPU encoding is Phase 6 Step 10)
    manifest = build_cache_manifest(
        expanded,
        cache_dir=cache_dir,
        vae_id=config.model.path,
        dtype=config.cache.dtype,
    )

    # Save manifest
    manifest_path = save_cache_manifest(manifest, cache_dir)
    print(f"\nManifest written: {manifest_path}")
    print(f"Entries: {manifest.total_entries}")
    print()

    if args.dry_run:
        print("Dry run — no encoding performed.")
        return

    # TODO: Phase 6 Step 10 — real VAE encoding goes here
    print("VAE encoding not yet implemented (Phase 6 Steps 10-11).")
    print("Manifest and cache structure are ready for when encoders are added.")


# ---------------------------------------------------------------------------
# Cache-text command (placeholder for GPU encoding)
# ---------------------------------------------------------------------------

def cmd_cache_text(args: argparse.Namespace) -> None:
    """Encode captions through T5 and cache to disk."""
    from dimljus.config.training_loader import load_training_config
    from dimljus.encoding.cache import (
        ensure_cache_dirs,
        load_cache_manifest,
        save_cache_manifest,
    )

    config = load_training_config(args.config)
    cache_dir = Path(config.cache.cache_dir)

    print(f"Cache directory: {cache_dir}")
    print()

    # Load existing manifest (cache-latents must run first)
    try:
        manifest = load_cache_manifest(cache_dir)
    except Exception as e:
        print(f"Error: {e}")
        print("Run 'cache-latents' first to build the cache manifest.")
        sys.exit(1)

    # Count entries that need text encoding
    needs_text = [e for e in manifest.entries if e.text_file and not e.has_text]
    stems_with_text = {
        e.sample_id.rsplit("_", 1)[0]
        for e in manifest.entries
        if e.text_file
    }
    print(f"Stems with captions: {len(stems_with_text)}")

    if args.dry_run:
        print("Dry run — no encoding performed.")
        return

    # TODO: Phase 6 Step 11 — real T5 encoding goes here
    print("T5 encoding not yet implemented (Phase 6 Steps 10-11).")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the encoding CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m dimljus.encoding",
        description="Dimljus latent pre-encoding and caching pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ─── info ───
    info_parser = subparsers.add_parser(
        "info",
        help="Show cache status for a training config.",
    )
    info_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the Dimljus training config YAML.",
    )

    # ─── cache-latents ───
    latents_parser = subparsers.add_parser(
        "cache-latents",
        help="Encode video/image targets through VAE.",
    )
    latents_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the Dimljus training config YAML.",
    )
    latents_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build manifest without encoding (preview what would be cached).",
    )
    latents_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode all entries, even if cache is up to date.",
    )

    # ─── cache-text ───
    text_parser = subparsers.add_parser(
        "cache-text",
        help="Encode captions through T5 text encoder.",
    )
    text_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the Dimljus training config YAML.",
    )
    text_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be encoded without actually encoding.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the encoding CLI."""
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "info": cmd_info,
        "cache-latents": cmd_cache_latents,
        "cache-text": cmd_cache_text,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
