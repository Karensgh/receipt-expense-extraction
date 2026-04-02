#!/usr/bin/env python3
"""Quick script to replace generic AI compliance tool purposes with realistic ones."""

import csv
import random
from pathlib import Path

COFFEE_PURPOSES = [
    "Mentor coffee chat",
    "Co-founder catch-up",
    "Quick sync with advisor",
    "Candidate intro chat",
    "Networking coffee",
    "Solo working session",
    "Partner intro coffee",
]

LUNCH_PURPOSES = [
    "Working lunch during sprint",
    "Candidate interview lunch",
    "Co-founder strategy chat",
    "Product roadmap working session",
    "Investor update lunch",
    "Sales prospect meeting",
    "Customer discovery debrief",
    "Founder meetup",
]

DINNER_PURPOSES = [
    "Investor pitch dinner",
    "Board member dinner",
    "Client onboarding dinner",
    "Partner intro dinner",
    "Team celebration dinner",
    "Advisor dinner",
    "Co-founder strategy dinner",
]


def pick_meal_purpose(amount: float) -> str:
    if amount < 15:
        return random.choice(COFFEE_PURPOSES)
    elif amount < 40:
        return random.choice(LUNCH_PURPOSES)
    else:
        return random.choice(DINNER_PURPOSES)

GENERIC_PATTERNS = [
    "ai compliance",
    "compliance tool",
    "fintech startup",
    "ai tool",
    "compliance product",
]


def is_generic(purpose: str) -> bool:
    lower = purpose.lower()
    return any(p in lower for p in GENERIC_PATTERNS)


def fix_csv(csv_path: Path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    changed = 0
    for row in rows:
        # Fix category: remove "(50% deductible)" note
        if "category" in row:
            row["category"] = row["category"].replace(" (50% deductible)", "")

        purpose = row.get("business_purpose_suggestion") or row.get("business_purpose") or ""
        category = row.get("category", "")

        # Only replace generic meal purposes
        if category != "Meals":
            continue
        if not is_generic(purpose):
            continue

        row_key = "business_purpose_suggestion" if "business_purpose_suggestion" in row else "business_purpose"
        amount = float(row.get("total") or row.get("amount") or 0)
        row[row_key] = pick_meal_purpose(amount)
        changed += 1

    v2_path = csv_path.with_name(csv_path.stem + "_v2.csv")
    with open(v2_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {changed} meal purposes -> {v2_path.name}")


if __name__ == "__main__":
    output = Path("output")
    for name in ["expenses_receipts.csv", "expenses_transactions.csv"]:
        p = output / name
        if p.exists():
            fix_csv(p)
