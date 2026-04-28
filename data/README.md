# Data Directory

## Required file: `apple_2025_10k.txt`

Place Apple's FY2025 Form 10-K as plain text at:

```
data/apple_2025_10k.txt
```

### How to obtain the 10-K text

**Option 1 — SEC EDGAR (official source)**

1. Go to https://www.sec.gov/cgi-bin/browse-edgar and search for Apple Inc. (ticker: AAPL).
2. Select "10-K" filing type.
3. Open the FY2025 annual report (filed for the fiscal year ending September 2025).
4. Download the main HTML document and save the text content.

**Option 2 — Direct EDGAR full-text search**

Search for Apple's 10-K at: https://efts.sec.gov/LATEST/search-index?q=%22apple+inc%22&dateRange=custom&startdt=2025-10-01&enddt=2025-11-30&forms=10-K

**Option 3 — Copy from the SEC filing**

Open the filing in a browser, select all text (Ctrl+A), paste into a text editor, and save as `apple_2025_10k.txt` with UTF-8 encoding.

### Tips for best retrieval quality

- Remove or minimize HTML/XML markup if copying from an HTML filing.
- Keep all financial tables intact — the retriever needs the raw numbers.
- The file can be large (several hundred KB to a few MB); this is expected.

### Files generated automatically

| File | Description |
|------|-------------|
| `embeddings_cache.pkl` | Cached chunk embeddings (generated on first run, reused after) |

`embeddings_cache.pkl` is excluded from git (see `.gitignore`). Delete it to force a rebuild if you update the 10-K text.
