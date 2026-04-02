#!/usr/bin/env python3
"""
Chase Statement Transaction Extractor

Parses Chase credit card PDF statements, extracts transactions from
ACCOUNT ACTIVITY sections, and uses Gemini to classify business expenses
with IRS Form 1120 categories.
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pdfplumber
from dotenv import load_dotenv
from icalendar import Calendar
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# IRS 1120 categories (same as receipt extractor)
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

# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------


def infer_year_from_filename(filename: str) -> int:
    """Extract statement year from Chase filename like '20260324-statements-...'."""
    match = re.match(r"(\d{4})\d{4}", filename)
    if match:
        return int(match.group(1))
    return datetime.now().year


def infer_statement_month(filename: str) -> int:
    """Extract statement month from Chase filename like '20260324-statements-...'."""
    match = re.match(r"\d{4}(\d{2})", filename)
    if match:
        return int(match.group(1))
    return datetime.now().month


def parse_chase_transactions(pdf_path: Path) -> list[dict]:
    """Extract transactions from a Chase credit card PDF statement."""
    year = infer_year_from_filename(pdf_path.name)
    statement_month = infer_statement_month(pdf_path.name)

    transactions = []
    in_activity = False

    # Pattern: MM/DD  description  amount (with optional negative for credits)
    tx_pattern = re.compile(
        r"^(\d{2}/\d{2})\s+(.+?)\s+(-?[\d,]+\.\d{2})$"
    )

    # Headers/labels to skip (not transaction lines)
    skip_lines = {
        "Date of",
        "Transaction Merchant Name or Transaction Description $ Amount",
        "PAYMENTS AND OTHER CREDITS",
        "PURCHASE",
        "PURCHASES",
    }

    # Markers that end the account activity section
    end_markers = [
        "INTEREST CHARGES",
        "Totals Year-to-Date",
        "Total fees charged",
        "Your Annual Percentage",
    ]

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = text.split("\n")

            for line in lines:
                stripped = line.strip()

                # Detect start of account activity (handles doubled chars like AACCCCOOUUNNTT)
                if "ACCOUNT ACTIVITY" in stripped or "AACCCCOOUUNNTT" in stripped:
                    in_activity = True
                    continue

                # Detect end of account activity section
                if in_activity and any(marker in stripped for marker in end_markers):
                    in_activity = False
                    continue

                if stripped in skip_lines:
                    continue

                if not in_activity:
                    continue

                match = tx_pattern.match(stripped)
                if match:
                    date_str = match.group(1)  # MM/DD
                    description = match.group(2).strip()
                    amount_str = match.group(3).replace(",", "")
                    amount = float(amount_str)

                    # Build full date — handle year boundary
                    # (e.g., statement is Jan 2026 but has Dec transactions)
                    tx_month = int(date_str.split("/")[0])
                    tx_year = year
                    if tx_month > statement_month and tx_month >= 11 and statement_month <= 2:
                        tx_year = year - 1

                    full_date = f"{tx_year}-{date_str.replace('/', '-')}"

                    transactions.append(
                        {
                            "date": full_date,
                            "description": description,
                            "amount": amount,
                            "source_file": pdf_path.name,
                        }
                    )

    return transactions


# ---------------------------------------------------------------------------
# Calendar context
# ---------------------------------------------------------------------------


def parse_calendar(ics_path: Path) -> list[dict]:
    """Parse an ICS file and return a list of events with date, summary, and location."""
    cal = Calendar.from_ical(ics_path.read_bytes())
    events = []
    for component in cal.walk():
        if component.name != "VEVENT":
            continue
        summary = str(component.get("SUMMARY", ""))
        location = str(component.get("LOCATION", ""))
        dtstart = component.get("DTSTART")
        dtend = component.get("DTEND")
        if not dtstart:
            continue
        start = dtstart.dt
        end = dtend.dt if dtend else start
        # Normalize date-only vs datetime
        if hasattr(start, "date"):
            start = start.date()
        if hasattr(end, "date"):
            end = end.date()
        events.append({
            "start": start,
            "end": end,
            "summary": summary,
            "location": location,
        })
    return events


def get_calendar_context(events: list[dict], date_str: str, window_days: int = 2) -> str:
    """Return calendar events near a transaction date as context string."""
    try:
        tx_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return ""
    nearby = []
    for ev in events:
        # Check if transaction date falls within event range (with window)
        ev_start = ev["start"] - timedelta(days=window_days)
        ev_end = ev["end"] + timedelta(days=window_days)
        if ev_start <= tx_date <= ev_end:
            loc = f" @ {ev['location']}" if ev['location'] else ""
            nearby.append(f"  - {ev['start']} to {ev['end']}: {ev['summary']}{loc}")
    if not nearby:
        return ""
    return "Calendar events around this date:\n" + "\n".join(nearby)


def get_batch_calendar_context(events: list[dict], transactions: list[dict]) -> str:
    """Return combined calendar context for a batch of transactions."""
    if not events:
        return ""
    # Collect unique date range for the batch
    dates = set()
    for tx in transactions:
        dates.add(tx["date"])
    # Get context for all dates, deduplicate events
    seen = set()
    all_context = []
    for d in sorted(dates):
        ctx = get_calendar_context(events, d)
        if ctx and ctx not in seen:
            seen.add(ctx)
            all_context.append(ctx)
    if not all_context:
        return ""
    return (
        "\nCALENDAR CONTEXT (use this to determine if expenses are business-related, "
        "e.g. travel/meals during a conference are business expenses):\n"
        + "\n".join(all_context)
        + "\n"
    )


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = """\
You are an expert accountant classifying credit card transactions for a business.

Business context: {business_context}

Below is a batch of credit card transactions. For EACH transaction, determine:
1. is_business_expense: true if this is a legitimate business expense, false if personal
2. category: IRS Form 1120 category (MUST be one of: {categories})
3. business_purpose: A concise business purpose description (empty string if personal)
4. confidence: Your confidence from 0.0 to 1.0

CLASSIFICATION RULES (apply in order):

1. ALWAYS business — classify these based on merchant nature alone, regardless of business context:
   - SaaS/dev tools (Claude, OpenAI, Vercel, Framer, Calendly, GitHub, Figma, Notion, Slack, reddit ads, google workspace, and more etc.)
   - Cloud/infra (AWS, GCP, Azure, Heroku, Supabase, etc.)
   - Domain/hosting (Cloudflare, Namecheap, GoDaddy, etc.)
   - Freelancer platforms (Fiverr, Upwork, Toptal, etc.)
   - Coworking/office (WeWork, Regus, etc.)
   - Professional services (legal, accounting, consulting)
   - Business phone/internet (AT&T, T-Mobile, Verizon prepaid, Google Workspace, etc.)
   - Advertising (Google Ads, Meta Ads, LinkedIn, etc.)

2. ALWAYS personal — these are never business expenses:
   - Groceries (Safeway, Trader Joe's, Whole Foods, etc.)
   - Personal entertainment (Netflix, Spotify, gaming, movies, etc.)
   - Personal shopping (clothing, home goods, etc.)
   - Liquor stores, laundry, personal care

3. AMBIGUOUS — use the business context AND calendar context to decide:
   - Meals/restaurants — business if during a conference, client meeting, or team event; personal otherwise
   - Rideshare (Lyft, Uber) — business if commuting to meetings, conferences, or office; personal otherwise
   - Hotels/lodging — business if during a work trip or conference; personal otherwise
   - Amazon purchases — could be office supplies or personal; use amount and description to judge

For ambiguous cases, lean toward personal and give a lower confidence score if unsure.

TRANSACTIONS:
{transactions}
{calendar_context}
Return a JSON array with one object per transaction, in the same order. Each object must have:
{{"index": 0, "is_business_expense": true/false, "category": "...", "business_purpose": "...", "confidence": 0.0-1.0}}

Return ONLY the JSON array, no other text."""


def classify_transactions(
    model: ChatGoogleGenerativeAI,
    transactions: list[dict],
    business_context: str,
    logger: logging.Logger,
    calendar_events: list[dict] | None = None,
    batch_size: int = 30,
    max_retries: int = 3,
) -> list[dict]:
    """Classify transactions in batches using Gemini. Returns enriched transaction dicts."""

    results = []

    for batch_start in range(0, len(transactions), batch_size):
        batch = transactions[batch_start : batch_start + batch_size]

        # Format transactions for the prompt
        tx_lines = []
        for i, tx in enumerate(batch):
            tx_lines.append(f"[{i}] {tx['date']}  {tx['description']}  ${tx['amount']:.2f}")
        tx_text = "\n".join(tx_lines)

        # Add calendar context for this batch
        cal_ctx = get_batch_calendar_context(calendar_events or [], batch)

        prompt = CLASSIFY_PROMPT.format(
            business_context=business_context,
            categories=", ".join(IRS_1120_CATEGORIES),
            transactions=tx_text,
            calendar_context=cal_ctx,
        )

        classified = None
        for attempt in range(1, max_retries + 1):
            try:
                response = model.invoke([HumanMessage(content=prompt)])
                text = response.content if isinstance(response.content, str) else str(response.content)

                # Strip markdown fences
                text = text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                if text.startswith("json"):
                    text = text[4:].strip()

                classified = json.loads(text)
                break

            except (json.JSONDecodeError, Exception) as e:
                wait = 2**attempt
                logger.warning(
                    "Batch %d–%d attempt %d/%d failed: %s — retrying in %ds",
                    batch_start,
                    batch_start + len(batch),
                    attempt,
                    max_retries,
                    str(e),
                    wait,
                )
                if attempt < max_retries:
                    time.sleep(wait)

        if classified is None:
            logger.error("All attempts failed for batch %d–%d", batch_start, batch_start + len(batch))
            # Mark all as unclassified
            for tx in batch:
                tx["is_business_expense"] = False
                tx["category"] = "Other"
                tx["business_purpose"] = ""
                tx["confidence"] = 0.0
                results.append(tx)
            continue

        # Merge classifications back into transactions
        for i, tx in enumerate(batch):
            if i < len(classified):
                c = classified[i]
                tx["is_business_expense"] = c.get("is_business_expense", False)
                tx["category"] = c.get("category", "Other")
                tx["business_purpose"] = c.get("business_purpose", "")
                tx["confidence"] = c.get("confidence", 0.0)
            else:
                tx["is_business_expense"] = False
                tx["category"] = "Other"
                tx["business_purpose"] = ""
                tx["confidence"] = 0.0
            results.append(tx)

    return results


# ---------------------------------------------------------------------------
# CSV + summary output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "transaction_id",
    "receipt_key",
    "date",
    "description",
    "amount",
    "category",
    "business_purpose",
    "confidence",
    "receipt_required",
    "source_file",
]


def make_receipt_key(date_str: str, category: str, idx: int) -> str:
    """Build a receipt key like '2024-03-15_meals_50pct_deductible_001'."""
    cat_slug = category.lower().replace("/", "_").replace(" ", "_")
    cat_slug = cat_slug.replace("(", "").replace(")", "").replace("%", "pct")
    return f"{date_str}_{cat_slug}_{idx:03d}"


def write_csv(rows: list[dict], output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict], all_tx_count: int, output_path: Path) -> str:
    total_expenses = sum(r["amount"] for r in rows)
    category_totals: dict[str, float] = {}
    for r in rows:
        cat = r["category"]
        category_totals[cat] = category_totals.get(cat, 0) + r["amount"]

    flagged = [r for r in rows if r["confidence"] < 0.7]

    lines = [
        "=" * 60,
        "CHASE TRANSACTION EXTRACTION SUMMARY",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        f"Total Transactions Scanned: {all_tx_count}",
        f"Business Expenses Found: {len(rows)}",
        f"Personal (excluded): {all_tx_count - len(rows)}",
        f"Total Business Expenses: ${total_expenses:,.2f}",
        "",
        "-" * 40,
        "BREAKDOWN BY CATEGORY",
        "-" * 40,
    ]

    for cat in sorted(category_totals, key=lambda c: category_totals[c], reverse=True):
        lines.append(f"  {cat:<30s} ${category_totals[cat]:>10,.2f}")

    needs_receipt = [r for r in rows if r["receipt_required"] == "YES"]
    lines += ["", "-" * 40, f"RECEIPT REQUIRED ({len(needs_receipt)} items)", "-" * 40]
    if needs_receipt:
        for r in needs_receipt:
            lines.append(
                f"  {r['transaction_id']}  {r['description'][:30]:<30s}  "
                f"${r['amount']:>8,.2f}  ({r['category']})"
            )
    else:
        lines.append("  None — all transactions under $75.")

    lines += ["", "-" * 40, "FLAGGED (LOW CONFIDENCE < 0.7)", "-" * 40]
    if flagged:
        for r in flagged:
            lines.append(
                f"  {r['transaction_id']}  {r['description'][:30]:<30s}  "
                f"${r['amount']:>8,.2f}  confidence={r['confidence']:.2f}"
            )
    else:
        lines.append("  None — all classifications above confidence threshold.")

    lines.append("")
    summary = "\n".join(lines)
    output_path.write_text(summary)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract business expenses from Chase credit card PDF statements using Gemini AI."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing Chase statement PDFs",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("output"), help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--business-context",
        type=str,
        default="Business expenses for a company",
        help="Business context for classification (e.g. 'Early-stage fintech SaaS startup')",
    )
    parser.add_argument(
        "--calendar",
        type=Path,
        default=None,
        help="Path to .ics calendar file for context (helps classify travel/meal expenses)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of PDFs to process (e.g. --limit 1 for testing)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = logging.getLogger("transaction_extractor")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(output_dir / "processing_transactions.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # Verify API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set. Create a .env file or export the variable.", file=sys.stderr)
        sys.exit(1)

    # Collect PDFs
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {input_dir}")
        sys.exit(0)

    if args.limit:
        pdfs = pdfs[: args.limit]

    print(f"Processing {len(pdfs)} Chase statement(s)")

    # Phase 1: Parse all PDFs
    all_transactions: list[dict] = []
    for pdf_path in tqdm(pdfs, desc="Parsing PDFs"):
        try:
            txs = parse_chase_transactions(pdf_path)
            all_transactions.extend(txs)
            logger.info("Parsed %d transactions from %s", len(txs), pdf_path.name)
        except Exception as e:
            logger.error("Failed to parse %s: %s", pdf_path.name, e)
            tqdm.write(f"  ⚠ Skipping {pdf_path.name}: {e}")

    if not all_transactions:
        print("No transactions found in any statement.")
        sys.exit(0)

    # Filter out payments/credits (negative amounts)
    purchases = [tx for tx in all_transactions if tx["amount"] > 0]
    credits = [tx for tx in all_transactions if tx["amount"] <= 0]

    print(f"Extracted {len(all_transactions)} transactions ({len(purchases)} purchases, {len(credits)} payments/credits)")
    logger.info("Total: %d, Purchases: %d, Credits: %d", len(all_transactions), len(purchases), len(credits))

    # Load calendar if provided
    calendar_events = None
    if args.calendar:
        if args.calendar.is_file():
            calendar_events = parse_calendar(args.calendar)
            print(f"Loaded {len(calendar_events)} calendar events for context")
            logger.info("Loaded %d calendar events from %s", len(calendar_events), args.calendar)
        else:
            print(f"Warning: calendar file not found: {args.calendar}", file=sys.stderr)

    # Phase 2: Classify with Gemini
    print(f"Classifying {len(purchases)} transactions with Gemini...")
    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.1,
        api_key=api_key,
    )

    classified = classify_transactions(model, purchases, args.business_context, logger, calendar_events)

    # Filter to business expenses only
    business_expenses = [tx for tx in classified if tx.get("is_business_expense")]
    print(f"Identified {len(business_expenses)} business expenses out of {len(purchases)} purchases")

    if not business_expenses:
        print("No business expenses identified.")
        sys.exit(0)

    # Build CSV rows
    rows = []
    for idx, tx in enumerate(sorted(business_expenses, key=lambda t: t["date"]), start=1):
        amount = tx["amount"]
        category = tx.get("category", "Other")
        # IRS requires receipts for $75+ and always for lodging
        needs_receipt = amount >= 75 or category == "Travel"
        receipt_key = make_receipt_key(tx["date"], category, idx)
        rows.append(
            {
                "transaction_id": f"tx_{idx:03d}",
                "receipt_key": receipt_key,
                "date": tx["date"],
                "description": tx["description"],
                "amount": amount,
                "category": category,
                "business_purpose": tx["business_purpose"],
                "confidence": tx["confidence"],
                "receipt_required": "YES" if needs_receipt else "no",
                "source_file": tx["source_file"],
            }
        )

    # Write outputs
    csv_path = output_dir / "expenses_transactions.csv"
    write_csv(rows, csv_path)
    logger.info("Wrote %d rows to %s", len(rows), csv_path)

    summary_path = output_dir / "summary_transactions.txt"
    summary = write_summary(rows, len(purchases), summary_path)

    print(f"\nDone! {len(rows)} business expenses extracted.\n")
    print(summary)
    print(f"\nOutputs written to {output_dir}/")
    print(f"  - expenses_transactions.csv ({len(rows)} rows)")
    print(f"  - summary_transactions.txt")
    print(f"  - processing_transactions.log")


if __name__ == "__main__":
    main()
