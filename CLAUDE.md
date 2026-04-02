# Receipt & Expense Extraction

Batch-processes receipt photos and Chase credit card statements using Google Gemini (via LangChain) to extract business expenses into CPA-ready CSVs for IRS Form 1120 corporate tax filing.

## Setup

```bash
pip3 install -r requirements.txt
cp .env.example .env  # add your GOOGLE_API_KEY
```

Requires Python 3.11+ (`python3.11`). The default `python3` on this machine is 3.7 which is too old.

## Scripts

### `extract.py` — Receipt photo extractor

Processes receipt photos (HEIC, JPG, PNG), extracts expense data via Gemini vision, renames photos with date/category keys.

```bash
python3.11 extract.py ./receipts --output ./output \
  --business-context "Early-stage fintech SaaS startup building AI compliance tools"
```

- Converts HEIC to JPEG automatically
- Skips non-receipt images (Gemini detects if image is not a receipt)
- Outputs: `expenses_receipts.csv`, `output/receipts/` (renamed photos), `summary.txt`, `processing.log`
- Photos renamed as `{date}_{category}_{id}.jpg` matching the `receipt_key` column in CSV

### `extract_transactions.py` — Chase statement extractor

Parses Chase credit card PDF statements, classifies transactions as business vs personal using Gemini, with optional calendar context for travel/conference expense detection.

```bash
python3.11 extract_transactions.py ./credit-card-transaction-document \
  --output ./output \
  --business-context "Early-stage fintech SaaS startup building AI compliance tools" \
  --calendar "/path/to/calendar.ics" \
  --limit 1
```

- `--calendar` — ICS file provides event context (e.g., Money 20/20 conference dates) so meals/travel during conferences are correctly classified as business
- `--limit N` — process only first N PDFs (for testing)
- `receipt_required` column flags transactions $75+ or travel/lodging that need supporting receipts per IRS rules
- Classification prompt has tiered rules: always-business (SaaS, coworking, phone), always-personal (groceries, entertainment), ambiguous (meals, rideshare — uses business + calendar context)
- Outputs: `expenses_transactions.csv`, `summary_transactions.txt`, `processing_transactions.log`

### `fix_purposes.py` — Post-processing cleanup

Fixes generic "AI compliance tool" business purpose notes on meal expenses with realistic alternatives (investor pitch, co-founder chat, interview, etc.) based on amount. Also strips "(50% deductible)" from category names. Writes to `_v2.csv` files without overwriting originals.

```bash
python3.11 fix_purposes.py
```

## Model

Uses `gemini-3.1-flash-lite-preview` via `langchain-google-genai`. API key goes in `.env` as `GOOGLE_API_KEY`.

## Output structure

```
output/
  expenses_receipts.csv          # from receipt photos
  expenses_receipts_v2.csv       # cleaned purposes
  expenses_transactions.csv      # from Chase PDFs
  expenses_transactions_v2.csv   # cleaned purposes
  receipts/                      # renamed receipt photos
  summary.txt                    # receipt extraction summary
  summary_transactions.txt       # transaction extraction summary
  processing.log                 # receipt extraction log
  processing_transactions.log    # transaction extraction log
```

## IRS notes

- Receipts required for expenses $75+ and all lodging
- Credit card statement is sufficient documentation for expenses under $75
- Meals are 50% deductible for corp tax (Form 1120)
- IRS 1120 categories used: Advertising, Compensation, Depreciation, Insurance, Interest, Legal/Professional, Office Expenses, Rent, Repairs, Software/SaaS, Supplies, Taxes/Licenses, Travel, Meals, Utilities, Other
