"""Format conversion — dimljus ↔ musubi ↔ ComfyUI checkpoint compatibility.

Dimljus uses PEFT/diffusers naming (the ai-toolkit convention):
    blocks.0.attn1.to_q.lora_A.weight

Musubi-tuner (kohya/sd-scripts lineage) uses:
    lora_unet_blocks_0_attn1_to_q.lora_down.weight

Both formats store the same tensors — just different key naming. This module
converts between them so dimljus LoRAs work in musubi/ComfyUI/A1111, and
musubi LoRAs can be loaded into dimljus for continued training.

GPU-free — operates on string keys and doesn't touch tensor values.
"""

from __future__ import annotations

import re
from typing import Any

from dimljus.training.wan.constants import (
    DIMLJUS_TO_MUSUBI_LORA_SUFFIX,
    MUSUBI_PREFIX,
    MUSUBI_TO_DIMLJUS_LORA_SUFFIX,
    T2V_LORA_TARGETS,
    I2V_EXTRA_TARGETS,
    WAN_NUM_BLOCKS,
)


# ---------------------------------------------------------------------------
# Dimljus → Musubi conversion
# ---------------------------------------------------------------------------

def dimljus_to_musubi(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a dimljus/PEFT state dict to musubi/kohya format.

    Conversion rules:
        1. Module path dots → underscores (blocks.0.attn1 → blocks_0_attn1)
        2. Add 'lora_unet_' prefix
        3. lora_A → lora_down, lora_B → lora_up
        4. Keep the separator dot between module path and LoRA suffix

    Args:
        state_dict: Dimljus format state dict.
            Keys like: 'blocks.0.attn1.to_q.lora_A.weight'

    Returns:
        New dict with musubi format keys and same tensor values.
            Keys like: 'lora_unet_blocks_0_attn1_to_q.lora_down.weight'

    Example:
        >>> dimljus_to_musubi({'blocks.0.attn1.to_q.lora_A.weight': tensor})
        {'lora_unet_blocks_0_attn1_to_q.lora_down.weight': tensor}
    """
    converted: dict[str, Any] = {}

    for key, value in state_dict.items():
        new_key = _convert_key_dimljus_to_musubi(key)
        converted[new_key] = value

    return converted


def _convert_key_dimljus_to_musubi(key: str) -> str:
    """Convert a single key from dimljus to musubi format.

    The key has two parts separated by the LoRA suffix:
        module_path.lora_A.weight → prefix_module_path.lora_down.weight

    We need to:
    1. Split at the LoRA suffix boundary
    2. Convert the module path (dots → underscores)
    3. Convert the LoRA suffix (A → down, B → up)
    4. Add the musubi prefix
    """
    # Find the LoRA suffix boundary
    for dimljus_suffix, musubi_suffix in DIMLJUS_TO_MUSUBI_LORA_SUFFIX.items():
        if key.endswith(dimljus_suffix):
            # Extract module path (everything before the LoRA suffix)
            module_path = key[: len(key) - len(dimljus_suffix) - 1]  # -1 for the dot
            # Convert dots to underscores in module path
            module_path_underscored = module_path.replace(".", "_")
            # Build musubi key
            return f"{MUSUBI_PREFIX}{module_path_underscored}.{musubi_suffix}"

    # If no LoRA suffix matched, pass through with prefix + underscore conversion
    # This handles metadata keys or non-standard entries
    return f"{MUSUBI_PREFIX}{key.replace('.', '_')}"


# ---------------------------------------------------------------------------
# Musubi → Dimljus conversion
# ---------------------------------------------------------------------------

def musubi_to_dimljus(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a musubi/kohya state dict to dimljus/PEFT format.

    Inverse of dimljus_to_musubi. Conversion rules:
        1. Remove 'lora_unet_' prefix
        2. Module path underscores → dots (blocks_0_attn1 → blocks.0.attn1)
        3. lora_down → lora_A, lora_up → lora_B

    The tricky part is the underscore-to-dot conversion in the module path.
    Musubi uses underscores both as module separators AND within module names
    (e.g. 'to_q' has an underscore that should NOT become a dot). We use
    the known Wan module structure to disambiguate.

    Args:
        state_dict: Musubi format state dict.
            Keys like: 'lora_unet_blocks_0_attn1_to_q.lora_down.weight'

    Returns:
        New dict with dimljus format keys and same tensor values.
            Keys like: 'blocks.0.attn1.to_q.lora_A.weight'
    """
    converted: dict[str, Any] = {}

    for key, value in state_dict.items():
        new_key = _convert_key_musubi_to_dimljus(key)
        converted[new_key] = value

    return converted


def _convert_key_musubi_to_dimljus(key: str) -> str:
    """Convert a single key from musubi to dimljus format.

    Strategy: parse the musubi key into known structural segments
    and reconstruct with dots.
    """
    # Split at the dot separating module path from LoRA suffix
    # 'lora_unet_blocks_0_attn1_to_q.lora_down.weight'
    # → path='lora_unet_blocks_0_attn1_to_q', suffix='lora_down.weight'
    dot_idx = key.find(".")
    if dot_idx == -1:
        return key  # No dot — unusual, pass through

    module_path = key[:dot_idx]
    lora_suffix = key[dot_idx + 1:]

    # Remove musubi prefix
    if module_path.startswith(MUSUBI_PREFIX):
        module_path = module_path[len(MUSUBI_PREFIX):]

    # Convert LoRA suffix
    for musubi_suffix, dimljus_suffix in MUSUBI_TO_DIMLJUS_LORA_SUFFIX.items():
        if lora_suffix == musubi_suffix:
            lora_suffix = dimljus_suffix
            break

    # Convert underscored module path back to dotted
    # Use regex to reconstruct: blocks_0_attn1_to_q → blocks.0.attn1.to_q
    dotted_path = _underscored_to_dotted(module_path)

    return f"{dotted_path}.{lora_suffix}"


def _underscored_to_dotted(path: str) -> str:
    """Convert an underscored module path to dotted format.

    Uses pattern matching against known Wan module structure:
        blocks_N_component_projection → blocks.N.component.projection

    Known structural segments (separated by dots in diffusers):
        blocks, N (digit), attn1, attn2, ffn
        to_q, to_k, to_v, to_out, add_k_proj, add_v_proj
        net, 0, 2, proj

    Segments that contain underscores in their original form:
        to_q, to_k, to_v, to_out, add_k_proj, add_v_proj

    Strategy: match known patterns greedily from left to right.
    """
    # The module path follows a rigid structure for Wan:
    # blocks_N_[attn1|attn2|ffn]_[projection chain]
    #
    # We reconstruct using regex that understands Wan structure.

    # Pattern: blocks_(\d+)_(.+)
    m = re.match(r"^blocks_(\d+)_(.+)$", path)
    if not m:
        # Not a standard Wan block path — best effort
        return path.replace("_", ".")

    block_num = m.group(1)
    remainder = m.group(2)

    # Parse the remainder into known segments
    segments = _parse_wan_module_path(remainder)
    return f"blocks.{block_num}.{'.'.join(segments)}"


# Known multi-word segments (contain underscores) in Wan module names.
# Order matters — longer matches first to avoid partial matching.
_KNOWN_SEGMENTS: list[str] = [
    "add_k_proj",
    "add_v_proj",
    "to_out",
    "to_q",
    "to_k",
    "to_v",
]


def _parse_wan_module_path(remainder: str) -> list[str]:
    """Parse the part after 'blocks_N_' into dot-separated segments.

    Greedy matching against known multi-word segments, then split
    remaining single-word segments on underscores.
    """
    segments: list[str] = []
    pos = 0

    while pos < len(remainder):
        # Try to match a known multi-word segment
        matched = False
        for seg in _KNOWN_SEGMENTS:
            if remainder[pos:].startswith(seg):
                # Check that the match is followed by '_' or end-of-string
                end = pos + len(seg)
                if end == len(remainder) or remainder[end] == "_":
                    segments.append(seg)
                    pos = end + 1 if end < len(remainder) else end
                    matched = True
                    break

        if not matched:
            # Take the next underscore-delimited token as a single segment
            next_underscore = remainder.find("_", pos)
            if next_underscore == -1:
                segments.append(remainder[pos:])
                break
            else:
                segments.append(remainder[pos:next_underscore])
                pos = next_underscore + 1

    return segments


# ---------------------------------------------------------------------------
# State dict validation
# ---------------------------------------------------------------------------

def validate_state_dict_keys(
    state_dict: dict[str, Any],
    variant: str = "2.2_t2v",
    rank: int | None = None,
) -> list[str]:
    """Validate LoRA state dict keys against expected Wan structure.

    Checks that:
    1. All keys follow the expected naming pattern
    2. Block numbers are within valid range (0-39)
    3. Module names match known Wan targets
    4. lora_A and lora_B pairs are complete
    5. If rank is given, verifies tensor shapes (when tensors have .shape)

    Args:
        state_dict: State dict to validate (dimljus format keys).
        variant: Wan variant for determining valid targets.
        rank: Expected LoRA rank (optional; validates shapes if given).

    Returns:
        List of issue strings. Empty list = valid.
    """
    issues: list[str] = []

    # Determine valid targets for this variant
    valid_targets = set(T2V_LORA_TARGETS)
    if "i2v" in variant:
        valid_targets.update(I2V_EXTRA_TARGETS)

    # Track A/B pairs
    a_keys: set[str] = set()
    b_keys: set[str] = set()

    for key in state_dict:
        # Check overall pattern: blocks.N.module.lora_[A|B].weight
        m = re.match(r"^blocks\.(\d+)\.(.+)\.(lora_[AB])\.weight$", key)
        if not m:
            issues.append(f"Unexpected key format: '{key}'")
            continue

        block_num = int(m.group(1))
        module_suffix = m.group(2)
        lora_part = m.group(3)

        # Validate block number
        if block_num < 0 or block_num >= WAN_NUM_BLOCKS:
            issues.append(
                f"Block {block_num} out of range (0-{WAN_NUM_BLOCKS - 1}): '{key}'"
            )

        # Validate module suffix
        if module_suffix not in valid_targets:
            issues.append(
                f"Unknown module '{module_suffix}' in key '{key}'"
            )

        # Track A/B pairs
        base_key = f"blocks.{block_num}.{module_suffix}"
        if lora_part == "lora_A":
            a_keys.add(base_key)
        else:
            b_keys.add(base_key)

        # Validate tensor shape if rank is given
        if rank is not None:
            tensor = state_dict[key]
            if hasattr(tensor, "shape"):
                shape = tensor.shape
                if lora_part == "lora_A" and len(shape) == 2:
                    if shape[0] != rank:
                        issues.append(
                            f"Rank mismatch in '{key}': expected dim 0 = {rank}, "
                            f"got {shape[0]}"
                        )
                elif lora_part == "lora_B" and len(shape) == 2:
                    if shape[1] != rank:
                        issues.append(
                            f"Rank mismatch in '{key}': expected dim 1 = {rank}, "
                            f"got {shape[1]}"
                        )

    # Check for unpaired A/B keys
    missing_b = a_keys - b_keys
    missing_a = b_keys - a_keys
    for base in sorted(missing_b):
        issues.append(f"Missing lora_B for: '{base}'")
    for base in sorted(missing_a):
        issues.append(f"Missing lora_A for: '{base}'")

    return issues
