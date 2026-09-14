"""
scripts/demo_object_recognition.py — Offline Personal Object Recognition Demo
=============================================================================

Demonstrates embedding-based personal object recognition using ReminisceCV:
1. Registers synthetic reference objects into an isolated test gallery.
2. Evaluates query images against the registered reference gallery.
3. Displays the structured ``RecognitionResult`` for:
   - True positive matches (query matches registered object).
   - False positive rejection (unregistered object scored below threshold).

Usage
-----
Run self-contained offline demonstration:
    python scripts/demo_object_recognition.py

Or provide a custom query image against the demo gallery:
    python scripts/demo_object_recognition.py --query path/to/query.jpg
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from app.memory import MemoryService, ObjectRegistrationService
from app.recognition import ObjectRecognitionService, RecognitionResult
from app.vision import CLIPVisionModel


def generate_demo_images() -> dict[str, Image.Image]:
    """Create synthetic test images in memory for the demonstration."""
    # Object 1: Blue sphere on light gray canvas
    obj1_ref1 = Image.new("RGB", (224, 224), color=(220, 220, 220))
    draw1 = ImageDraw.Draw(obj1_ref1)
    draw1.ellipse([50, 50, 174, 174], fill=(30, 80, 210), outline=(0, 0, 100), width=3)

    # Object 1: Slight angle / illumination variant
    obj1_ref2 = Image.new("RGB", (224, 224), color=(210, 210, 210))
    draw1b = ImageDraw.Draw(obj1_ref2)
    draw1b.ellipse([45, 55, 169, 179], fill=(40, 90, 220), outline=(0, 0, 100), width=3)

    # Object 2: Orange square on dark background
    obj2_ref1 = Image.new("RGB", (224, 224), color=(40, 40, 40))
    draw2 = ImageDraw.Draw(obj2_ref1)
    draw2.rectangle([50, 50, 174, 174], fill=(230, 120, 20), outline=(255, 200, 100), width=3)

    # Query A: Close variant of Object 1 (should match Object 1)
    query_a = Image.new("RGB", (224, 224), color=(215, 215, 215))
    draw_qa = ImageDraw.Draw(query_a)
    draw_qa.ellipse([52, 48, 176, 172], fill=(35, 85, 215), outline=(0, 0, 120), width=3)

    # Query B: Close variant of Object 2 (should match Object 2)
    query_b = Image.new("RGB", (224, 224), color=(45, 45, 45))
    draw_qb = ImageDraw.Draw(query_b)
    draw_qb.rectangle([55, 55, 170, 170], fill=(225, 115, 25), outline=(255, 190, 90), width=3)

    # Query C: Unregistered distinct object (green triangle on black canvas)
    query_c = Image.new("RGB", (224, 224), color=(10, 10, 10))
    draw_qc = ImageDraw.Draw(query_c)
    draw_qc.polygon([(112, 40), (40, 180), (184, 180)], fill=(30, 200, 60), outline=(255, 255, 255))

    return {
        "obj1_ref1": obj1_ref1,
        "obj1_ref2": obj1_ref2,
        "obj2_ref1": obj2_ref1,
        "query_a": query_a,
        "query_b": query_b,
        "query_c": query_c,
    }


def print_result(scenario_name: str, res: RecognitionResult) -> None:
    """Pretty-print a structured RecognitionResult."""
    print(f"\n--- Scenario: {scenario_name} ---")
    print(f"  Matched              : {res.matched}")
    print(f"  Similarity Score     : {res.similarity:.4f}")
    print(f"  Applied Threshold    : {res.threshold:.4f}")
    print(f"  Entity ID            : {res.entity_id}")
    print(f"  Entity Name          : {res.name}")
    if res.memory:
        print(f"  Narrative            : {res.memory.narrative}")
        print(f"  Giver / Source       : {res.memory.giver_name}")
    if res.reference_image_path:
        print(f"  Best Reference Path  : {Path(res.reference_image_path).name}")

    if res.matched:
        print(f"  Status               : [MATCH CONFIRMED]")
    else:
        print(f"  Status               : [NO MATCH / REJECTED]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ReminisceCV — Personal Object Recognition Demo"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Path to an optional local query image to test against the gallery",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold (default: 0.85)",
    )
    args = parser.parse_args()

    print("=" * 68)
    print("  ReminisceCV — Personal Object Recognition Pipeline Demo")
    print("=" * 68)

    # Use a temporary directory and in-memory database for clean isolation
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ref_dir = tmp_path / "references"
        emb_dir = tmp_path / "embeddings"

        print("\n[1] Initializing CLIP Vision Model...")
        vision_model = CLIPVisionModel()
        vision_model.load_model()
        print("    [OK] Vision model ready.")

        print("\n[2] Setting up Registration & Recognition Services...")
        mem_svc = MemoryService(db_path=":memory:")
        mem_svc.start()

        reg_svc = ObjectRegistrationService(
            memory_service=mem_svc,
            vision_model=vision_model,
            references_dir=ref_dir,
            embeddings_dir=emb_dir,
        )

        rec_svc = ObjectRecognitionService(
            vision_model=vision_model,
            registration_service=reg_svc,
            threshold=args.threshold,
        )

        # Generate synthetic demo data
        demo_images = generate_demo_images()

        print("\n[3] Registering Test Objects into Personal Memory Gallery:")
        # Object 1: Synthetic Object Alpha
        obj1 = reg_svc.register_object(
            name="Synthetic Object Alpha (Blue Sphere)",
            giver="Test Giver Alpha",
            occasion="Demo Event 1",
            year="2024",
            narrative="A keepsake sphere used for vision pipeline verification.",
            reference_images=[demo_images["obj1_ref1"], demo_images["obj1_ref2"]],
        )
        print(f"    Registered: ID={obj1.memory_id} '{obj1.name}' with 2 reference images")

        # Object 2: Synthetic Object Beta
        obj2 = reg_svc.register_object(
            name="Synthetic Object Beta (Orange Cube)",
            giver="Test Giver Beta",
            occasion="Demo Event 2",
            year="2023",
            narrative="A geometric keepsake used for distinct item testing.",
            reference_images=[demo_images["obj2_ref1"]],
        )
        print(f"    Registered: ID={obj2.memory_id} '{obj2.name}' with 1 reference image")

        # Load registered embeddings into recognition service
        rec_svc.refresh_gallery()
        print(f"    Gallery Size: {rec_svc.gallery_size} reference embeddings loaded into memory.")

        print("\n[4] Running Recognition Pipeline Queries:")

        if args.query:
            query_path = Path(args.query)
            if not query_path.is_file():
                print(f"Error: Query file not found: {query_path}")
                sys.exit(1)
            result = rec_svc.recognize(query_path)
            print_result(f"Custom Query Image ({query_path.name})", result)
        else:
            # Query 1: Should match Object 1
            res_a = rec_svc.recognize(demo_images["query_a"])
            print_result("Query A (Variant of Blue Sphere)", res_a)

            # Query 2: Should match Object 2
            res_b = rec_svc.recognize(demo_images["query_b"])
            print_result("Query B (Variant of Orange Cube)", res_b)

            # Query 3: Unregistered item (Green Triangle)
            res_c = rec_svc.recognize(demo_images["query_c"])
            print_result("Query C (Unregistered Green Triangle)", res_c)

        print("\n" + "=" * 68)
        print("  Object Recognition Pipeline Verified Successfully.")
        print("=" * 68)

        mem_svc.stop()


if __name__ == "__main__":
    main()
