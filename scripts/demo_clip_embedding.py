"""
scripts/demo_clip_embedding.py — Demo: CLIP Image Embedding & Cosine Similarity
================================================================================

This script demonstrates ReminisceCV's pretrained CLIP vision embedding engine:
1. Loads or synthetically creates two images.
2. Extracts L2-normalized 1D feature vectors using ``CLIPVisionModel``.
3. Computes cosine similarity using the canonical formula:

   .. math::
       \\text{similarity}(A, B) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}

4. Displays vector shapes, norms, and similarity metrics.

Usage
-----
Run with synthetic demonstration images:
    python scripts/demo_clip_embedding.py

Or provide custom local image paths:
    python scripts/demo_clip_embedding.py --image-a path/to/a.jpg --image-b path/to/b.jpg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from app.vision import CLIPVisionModel, cosine_similarity


def create_synthetic_images() -> tuple[Image.Image, Image.Image, Image.Image]:
    """Generate sample synthetic images for demonstration without requiring local files.

    Returns
    -------
    tuple of (image_a, image_b, image_a_variant)
        - image_a: Blue canvas with a green circle (e.g. Object 1).
        - image_b: Warm canvas with a red rectangle (e.g. Object 2, distinct).
        - image_a_variant: Similar to image_a with slight noise/variation.
    """
    # Image 1: Blue canvas + Green circle
    img_a = Image.new("RGB", (224, 224), color=(30, 60, 150))
    draw_a = ImageDraw.Draw(img_a)
    draw_a.ellipse([50, 50, 174, 174], fill=(50, 200, 100), outline=(255, 255, 255), width=3)

    # Image 2: Yellow canvas + Red square
    img_b = Image.new("RGB", (224, 224), color=(240, 200, 50))
    draw_b = ImageDraw.Draw(img_b)
    draw_b.rectangle([60, 60, 164, 164], fill=(220, 50, 50), outline=(0, 0, 0), width=3)

    # Image 3: Variant of A (same blue canvas + slightly shifted circle)
    img_a_variant = Image.new("RGB", (224, 224), color=(35, 65, 155))
    draw_av = ImageDraw.Draw(img_a_variant)
    draw_av.ellipse([55, 55, 179, 179], fill=(60, 210, 110), outline=(250, 250, 250), width=3)

    return img_a, img_b, img_a_variant


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ReminisceCV — CLIP Image Embedding & Cosine Similarity Demo"
    )
    parser.add_argument(
        "--image-a",
        type=str,
        default=None,
        help="Path to first image (optional, default: synthetic image A)",
    )
    parser.add_argument(
        "--image-b",
        type=str,
        default=None,
        help="Path to second image (optional, default: synthetic image B)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"CLIP model identifier (default: {config.clip_model_name})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on ('cpu' or 'cuda', default: auto-detect)",
    )

    args = parser.parse_args()

    print("=" * 65)
    print("  ReminisceCV — Pretrained CLIP Embedding & Similarity Demo")
    print("=" * 65)

    # 1. Initialize and load CLIP Vision Model
    print(f"\n[1] Initializing CLIP Vision Model...")
    model = CLIPVisionModel(
        model_name=args.model,
        device=args.device,
    )
    print(f"    Configured Model : {model.model_name}")
    print(f"    Target Device    : {model.device.upper()}")
    print(f"    Embedding Dim    : {model.embedding_dim}")

    print("\n[2] Loading weights and processor (cached in models/)...")
    try:
        model.load_model()
        print("    [OK] Model successfully loaded into memory.")
    except Exception as exc:
        print(f"    [FAIL] Could not load CLIP model: {exc}")
        print("    Tip: Check internet connectivity for first-time model download.")
        sys.exit(1)

    # 2. Prepare images
    if args.image_a and args.image_b:
        print(f"\n[3] Loading input images from disk:")
        print(f"    Image A: {args.image_a}")
        print(f"    Image B: {args.image_b}")
        img_a: Any = args.image_a
        img_b: Any = args.image_b
        compare_variant = False
    else:
        print(f"\n[3] Generating synthetic test images in memory...")
        img_a, img_b, img_a_var = create_synthetic_images()
        print("    Image A          : Synthetic Object 1 (Green Circle on Blue Canvas)")
        print("    Image B          : Synthetic Object 2 (Red Square on Yellow Canvas)")
        print("    Image A (Variant): Synthetic Object 1 with subtle shift")
        compare_variant = True

    # 3. Generate embeddings
    print("\n[4] Generating normalized embeddings...")
    emb_a = model.encode_image(img_a)
    emb_b = model.encode_image(img_b)

    norm_a = float(np.linalg.norm(emb_a))
    norm_b = float(np.linalg.norm(emb_b))

    print(f"    Embedding A: shape={emb_a.shape}, dtype={emb_a.dtype}, L2 norm={norm_a:.4f}")
    print(f"    Embedding B: shape={emb_b.shape}, dtype={emb_b.dtype}, L2 norm={norm_b:.4f}")

    # 4. Calculate Cosine Similarity
    # Formula: similarity(A, B) = (A · B) / (||A|| ||B||)
    sim_ab = model.similarity(emb_a, emb_b)
    sim_self_a = model.similarity(emb_a, emb_a)

    print("\n[5] Cosine Similarity Results:")
    print("    Mathematical Definition:")
    print("        similarity(A, B) = (A · B) / (||A|| ||B||)")
    print("-" * 65)
    print(f"    similarity(Image A, Image A) [Self-Identity]  : {sim_self_a:.4f} (Expected ~1.0000)")
    print(f"    similarity(Image A, Image B) [Distinct Items] : {sim_ab:.4f}")

    if compare_variant:
        emb_a_var = model.encode_image(img_a_var)
        sim_variant = model.similarity(emb_a, emb_a_var)
        print(f"    similarity(Image A, Image A Variant)         : {sim_variant:.4f} (Expected high)")

    print("-" * 65)
    threshold = config.object_similarity_threshold
    print(f"\nConfigured Retrieval Threshold : {threshold:.2f}")
    if sim_ab >= threshold:
        print(f"Decision: MATCH (similarity {sim_ab:.4f} >= threshold {threshold:.2f})")
    else:
        print(f"Decision: NO MATCH / DISTINCT (similarity {sim_ab:.4f} < threshold {threshold:.2f})")

    print("\n[OK] CLIP Vision Embedding Engine verified successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
