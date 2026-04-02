#!/usr/bin/env python3
"""
Receipt Expense Extractor

Batch-processes receipt photos using Google Gemini via LangChain to extract
expense data and output a CPA-ready CSV for IRS Form 1120 corporate tax prep.
"""

import argparse
import base64
import csv
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

IRS_1120_CATEGORIES = [
    "Advertising",
    "Compensation",
    "Depreciation",
    "Insurance",
    "Interest",
    "Legal/Professional",
    "Office Expenses",
    "Rent",
    "Repairs",
    "Software/SaaS",
    "Supplies",
    "Taxes/Licenses",
    "Travel",
    "Meals (50% deductible)",
    "Utilities",
    "Other",
]


class LineItem(BaseModel):
    description: str = Field(description="Description of the line item")
    amount: float = Field(description="Amount for this line item")


class ReceiptData(BaseModel):
    is_receipt: bool = Field(description="True if the image is a receipt/invoice, False if it is not (e.g. a random photo, screenshot, document)")
    vendor: str = Field(default="", description="Name of the vendor/merchant")
    date: str = Field(default="", description="Transaction date in YYYY-MM-DD format")
    total: float = Field(description="Total amount on the receipt")
    tax: float = Field(default=0.0, description="Tax amount, 0 if not visible")
    currency: str = Field(default="USD", description="Currency code (e.g. USD, EUR)")
    payment_method: str = Field(
        default="Unknown",
        description="Payment method: Cash, Credit Card, Debit Card, ACH, Wire, Check, or Unknown",
    )
    category: str = Field(
        description=f"IRS Form 1120 expense category. Must be one of: {', '.join(IRS_1120_CATEGORIES)}"
    )
    line_items: list[LineItem] = Field(
        default_factory=list,
        description="Individual line items if visible on the receipt",
    )
    confidence_score: float = Field(
        description="Confidence score from 0.0 to 1.0 for the overall extraction quality"
    )
    business_purpose_suggestion: str = Field(
        description="Suggested business purpose description for this expense"
    )


# ---------------------------------------------------------------------------
# HEIC conversion
# ---------------------------------------------------------------------------


def convert_heic_to_jpeg(heic_path: Path) -> Path:
    """Convert a HEIC image to JPEG and return the path to the converted file."""
    import pillow_heif

    pillow_heif.register_heif_opener()
    img = Image.open(heic_path)
    jpeg_path = heic_path.with_suffix(".jpg")
    img.save(jpeg_path, "JPEG")
    return jpeg_path


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def encode_image_to_base64(image_path: Path) -> tuple[str, str]:
    """Return (base64_data, mime_type) for an image file."""
    suffix = image_path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    mime_type = mime_map.get(suffix, "image/jpeg")
    data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return data, mime_type


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are an expert accountant and receipt data extractor. Analyze this image.

First, determine if the image is actually a receipt or invoice. If it is NOT a receipt (e.g. a random photo, screenshot, meme, document), return: {{"is_receipt": false}}

If it IS a receipt, return a JSON object with these fields:
- is_receipt: true
- vendor: Name of the vendor/merchant
- date: Transaction date in YYYY-MM-DD format (use best guess if partially visible)
- total: Total amount as a number (no currency symbols)
- currency: Three-letter currency code (default USD)
- payment_method: One of: Cash, Credit Card, Debit Card, ACH, Wire, Check, Unknown
- category: IRS Form 1120 expense category — MUST be exactly one of: {categories}
- line_items: Array of objects with "description" and "amount" for each visible line item
- confidence_score: Your confidence in the extraction accuracy from 0.0 to 1.0
- business_purpose_suggestion: A concise business purpose description for this expense

{business_context}

Be precise with numbers. If a field is not visible, use reasonable defaults. Always return valid JSON."""


def build_prompt(business_context: Optional[str] = None) -> str:
    ctx = ""
    if business_context:
        ctx = (
            f"Business context for better categorization and purpose suggestions: {business_context}\n"
            "Use this context to write a more specific and relevant business_purpose_suggestion."
        )
    return EXTRACTION_PROMPT.format(
        categories=", ".join(IRS_1120_CATEGORIES),
        business_context=ctx,
    )


def extract_receipt(
    model: ChatGoogleGenerativeAI,
    image_path: Path,
    prompt_text: str,
    logger: logging.Logger,
    max_retries: int = 3,
) -> Optional[ReceiptData]:
    """Send an image to Gemini and return structured ReceiptData, with retries."""
    b64_data, mime_type = encode_image_to_base64(image_path)

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
            },
        ]
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = model.invoke([message])
            text = response.content if isinstance(response.content, str) else str(response.content)

            # Strip markdown fences if present
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()

            data = json.loads(text)
            return ReceiptData(**data)

        except (json.JSONDecodeError, Exception) as e:
            wait = 2**attempt
            logger.warning(
                "Attempt %d/%d failed for %s: %s — retrying in %ds",
                attempt,
                max_retries,
                image_path.name,
                str(e),
                wait,
            )
            if attempt < max_retries:
                time.sleep(wait)

    logger.error("All %d attempts failed for %s", max_retries, image_path.name)
    return None


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "receipt_id",
    "receipt_key",
    "original_file",
    "vendor",
    "date",
    "total",
    "tax",
    "currency",
    "payment_method",
    "category",
    "line_items",
    "confidence_score",
    "business_purpose_suggestion",
]


def make_receipt_key(date_str: str, category: str, idx: int) -> str:
    """Build a receipt key like '2024-03-15_meals_001' used as the renamed photo filename."""
    cat_slug = category.lower().replace("/", "_").replace(" ", "_")
    cat_slug = cat_slug.replace("(", "").replace(")", "").replace("%", "pct")
    return f"{date_str}_{cat_slug}_{idx:03d}"


def write_csv(rows: list[dict], output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def write_summary(rows: list[dict], output_path: Path) -> str:
    """Generate and write summary.txt. Returns the summary string."""
    total_expenses = sum(r["total"] for r in rows)
    category_totals: dict[str, float] = {}
    for r in rows:
        cat = r["category"]
        category_totals[cat] = category_totals.get(cat, 0) + r["total"]

    flagged = [r for r in rows if r["confidence_score"] < 0.7]

    lines = [
        "=" * 60,
        "RECEIPT EXPENSE EXTRACTION SUMMARY",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        f"Total Receipts Processed: {len(rows)}",
        f"Total Expenses: ${total_expenses:,.2f}",
        "",
        "-" * 40,
        "BREAKDOWN BY CATEGORY",
        "-" * 40,
    ]

    for cat in sorted(category_totals, key=lambda c: category_totals[c], reverse=True):
        lines.append(f"  {cat:<30s} ${category_totals[cat]:>10,.2f}")

    lines += ["", "-" * 40, "FLAGGED (LOW CONFIDENCE < 0.7)", "-" * 40]
    if flagged:
        for r in flagged:
            lines.append(
                f"  {r['receipt_id']}  {r['vendor']:<25s}  confidence={r['confidence_score']:.2f}"
            )
    else:
        lines.append("  None — all extractions above confidence threshold.")

    lines.append("")
    summary = "\n".join(lines)
    output_path.write_text(summary)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}


def collect_images(input_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract expense data from receipt photos using Gemini AI."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing receipt photos")
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("output"), help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--business-context",
        type=str,
        default=None,
        help="Business context for better categorization (e.g. 'Early-stage fintech SaaS startup')",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Setup output dirs
    output_dir.mkdir(parents=True, exist_ok=True)
    receipts_dir = output_dir / "receipts"
    receipts_dir.mkdir(exist_ok=True)

    # Setup logging
    logger = logging.getLogger("receipt_extractor")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(output_dir / "processing.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # Verify API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set. Create a .env file or export the variable.", file=sys.stderr)
        sys.exit(1)

    # Collect images
    images = collect_images(input_dir)
    if not images:
        print(f"No receipt images found in {input_dir}")
        sys.exit(0)

    print(f"Found {len(images)} receipt image(s) in {input_dir}")
    logger.info("Starting extraction for %d images", len(images))

    # Init model
    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.1,
        api_key=api_key,
    )

    prompt_text = build_prompt(args.business_context)
    rows: list[dict] = []
    temp_files: list[Path] = []  # track converted HEIC files for cleanup

    for idx, image_path in enumerate(tqdm(images, desc="Processing receipts"), start=1):
        receipt_id = f"receipt_{idx:03d}"
        logger.info("Processing %s -> %s", image_path.name, receipt_id)

        process_path = image_path

        # Convert HEIC if needed
        if image_path.suffix.lower() == ".heic":
            try:
                process_path = convert_heic_to_jpeg(image_path)
                temp_files.append(process_path)
                logger.info("Converted HEIC to JPEG: %s", process_path.name)
            except Exception as e:
                logger.error("Failed to convert HEIC %s: %s", image_path.name, e)
                tqdm.write(f"  ⚠ Skipping {image_path.name} (HEIC conversion failed)")
                continue

        # Validate image is readable
        try:
            with Image.open(process_path) as img:
                img.verify()
        except Exception as e:
            logger.error("Unreadable image %s: %s", image_path.name, e)
            tqdm.write(f"  ⚠ Skipping {image_path.name} (unreadable)")
            continue

        # Extract via LLM
        result = extract_receipt(model, process_path, prompt_text, logger)
        if result is None:
            tqdm.write(f"  ⚠ Failed to extract data from {image_path.name}")
            continue

        if not result.is_receipt:
            logger.info("Skipping %s — not a receipt", image_path.name)
            tqdm.write(f"  ⏭ Skipping {image_path.name} (not a receipt)")
            continue

        # Build receipt key from extracted data and rename image
        out_ext = process_path.suffix.lower()
        receipt_key = make_receipt_key(result.date, result.category, idx)
        renamed = f"{receipt_key}{out_ext}"
        shutil.copy2(process_path, receipts_dir / renamed)

        # Build CSV row
        line_items_str = "; ".join(
            f"{li.description}: ${li.amount:.2f}" for li in result.line_items
        )

        row = {
            "receipt_id": receipt_id,
            "receipt_key": renamed,
            "original_file": image_path.name,
            "vendor": result.vendor,
            "date": result.date,
            "total": result.total,
            "tax": result.tax,
            "currency": result.currency,
            "payment_method": result.payment_method,
            "category": result.category,
            "line_items": line_items_str,
            "confidence_score": result.confidence_score,
            "business_purpose_suggestion": result.business_purpose_suggestion,
        }
        rows.append(row)

    # Cleanup temp HEIC conversions
    for tmp in temp_files:
        try:
            tmp.unlink()
        except OSError:
            pass

    if not rows:
        print("No receipts were successfully extracted.")
        logger.warning("No successful extractions")
        sys.exit(0)

    # Write outputs
    csv_path = output_dir / "expenses_receipts.csv"
    write_csv(rows, csv_path)
    logger.info("Wrote %d rows to %s", len(rows), csv_path)

    summary_path = output_dir / "summary.txt"
    summary = write_summary(rows, summary_path)

    print(f"\nDone! {len(rows)}/{len(images)} receipts extracted successfully.\n")
    print(summary)
    print(f"\nOutputs written to {output_dir}/")
    print(f"  - expenses.csv ({len(rows)} rows)")
    print(f"  - receipts/ ({len(rows)} renamed images)")
    print(f"  - summary.txt")
    print(f"  - processing.log")


if __name__ == "__main__":
    main()