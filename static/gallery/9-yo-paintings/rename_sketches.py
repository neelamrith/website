#!/usr/bin/env python3
"""
Sketch renaming tool using Claude's vision API.
Compresses images, analyzes them with Claude, and renames based on content.
"""

import os
import sys
import base64
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Set

import anthropic
from PIL import Image


# Configuration
GALLERY_PATH = Path(".")
LOG_FILE_PATH = Path("rename_log.txt")
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
COMPRESSION_QUALITY = 80


def get_image_files() -> list[Path]:
    """Find all supported image files in the gallery."""
    if not GALLERY_PATH.exists():
        raise FileNotFoundError(f"Gallery path not found: {GALLERY_PATH}")

    image_files = []
    for ext in SUPPORTED_FORMATS:
        # Case-insensitive search
        image_files.extend(GALLERY_PATH.glob(f"*{ext}"))
        image_files.extend(GALLERY_PATH.glob(f"*{ext.upper()}"))

    return sorted(set(image_files))  # Remove duplicates, sort for consistency


def compress_image(image_path: Path) -> bool:
    """Compress image to 80% quality. Returns True if successful."""
    try:
        with Image.open(image_path) as img:
            # Convert RGBA/LA/palette to RGB for JPEG compatibility
            if img.mode in ("RGBA", "LA", "P"):
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA":
                    rgb_img.paste(img, mask=img.split()[3])
                else:
                    rgb_img.paste(img)
                rgb_img.save(image_path, "JPEG", quality=COMPRESSION_QUALITY, optimize=True)
            else:
                # For JPEG and other formats
                img.save(image_path, quality=COMPRESSION_QUALITY, optimize=True)
        return True
    except Exception as e:
        print(f"  [ERROR] Compression failed: {e}")
        return False


def get_image_suggestion(client: anthropic.Anthropic, image_path: Path) -> Optional[str]:
    """Use Claude to suggest a filename based on image content."""
    try:
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # Determine media type
        ext = image_path.suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        media_type = media_type_map.get(ext, "image/jpeg")

        # Call Claude with vision
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Analyze this sketch and suggest a concise filename (2-3 words, lowercase, "
                                "hyphens instead of spaces). Only respond with the filename, no quotes or explanation."
                            ),
                        },
                    ],
                }
            ],
        )

        suggested_name = response.content[0].text.strip().lower()

        # Clean up the suggestion
        suggested_name = suggested_name.replace(" ", "-").replace("_", "-")
        # Remove unwanted characters, keep only alphanumeric, hyphens, and dots
        suggested_name = "".join(c for c in suggested_name if c.isalnum() or c in "-.")
        # Remove leading/trailing hyphens
        suggested_name = suggested_name.strip("-")

        return suggested_name if suggested_name else None

    except Exception as e:
        print(f"  [ERROR] Claude analysis failed: {e}")
        return None


def rename_with_duplicate_check(
    old_path: Path,
    new_name: str,
    used_names: Set[str],
    log_file
) -> bool:
    """Rename file and handle duplicates by adding numeric suffix."""
    try:
        ext = old_path.suffix
        base_name = new_name
        counter = 1

        # Check for duplicates and find available name
        final_name = new_name
        while final_name in used_names or (old_path.parent / f"{final_name}{ext}").exists():
            final_name = f"{base_name}-{counter}"
            counter += 1

        # Perform the rename
        new_path = old_path.parent / f"{final_name}{ext}"
        old_path.rename(new_path)

        # Log the operation
        log_entry = f"{old_path.name} → {new_path.name}"
        log_file.write(log_entry + "\n")
        log_file.flush()

        used_names.add(final_name)
        print(f"  [OK] Renamed to: {new_path.name}")
        return True

    except Exception as e:
        print(f"  [ERROR] Rename failed: {e}")
        return False


def main():
    """Main processing function."""
    print("=" * 60)
    print("Sketch Renaming Tool")
    print("=" * 60)

    # Get image files
    try:
        image_files = get_image_files()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not image_files:
        print("No image files found in gallery.")
        return

    print(f"Found {len(image_files)} image files\n")

    # Initialize Claude client
    try:
        client = anthropic.Anthropic()
    except Exception as e:
        print(f"[ERROR] Initializing Claude client: {e}")
        print("[ERROR] Make sure ANTHROPIC_API_KEY environment variable is set.")
        return

    # Track used names to detect duplicates
    used_names: Set[str] = set()
    processed_count = 0

    # Open log file
    with open(LOG_FILE_PATH, "w") as log_file:
        log_file.write(f"Image Rename Log - {datetime.now().isoformat()}\n")
        log_file.write("=" * 60 + "\n\n")

        for idx, image_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")

            # Step 1: Compress image
            if not compress_image(image_path):
                print("  [SKIP] Compression failed\n")
                continue
            print(f"  [OK] Compressed to {COMPRESSION_QUALITY}% quality")

            # Step 2: Get filename suggestion from Claude
            suggested_name = get_image_suggestion(client, image_path)
            if not suggested_name:
                print("  [SKIP] No suggestion generated\n")
                continue
            print(f"  [OK] Suggested name: {suggested_name}")

            # Step 3: Rename file with duplicate check
            if rename_with_duplicate_check(image_path, suggested_name, used_names, log_file):
                processed_count += 1

            print()

    # Summary
    print("=" * 60)
    print(f"[DONE] Processing complete!")
    print(f"  Processed: {processed_count}/{len(image_files)} files")
    print(f"  Log saved to: {LOG_FILE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
