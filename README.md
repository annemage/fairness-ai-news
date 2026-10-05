# The Fairness Dispatch

A newspaper-style front page of new research on **algorithmic fairness** and
**computational social choice / fair division**, updated automatically by GitHub Actions
and served by GitHub Pages from `docs/`.

## How it works

`scripts/update.py` runs on a schedule (`.github/workflows/update.yml`):

1. **arXiv API**: new preprints in cs.LG, cs.AI, cs.CY, cs.GT, cs.MA, stat.ML, econ.TH, cs.HC, cs.CL
   that match fairness / social-choice keywords.
2. **Semantic Scholar API**: newly indexed papers from top venues (FAccT, AIES, AAAI, IJCAI,
   NeurIPS, ICML, ICLR, AAMAS, EC, WINE, SAGT, JAIR, ...), plus a venue lookup for every
   preprint, so stories show "AAAI 2026" etc. with a link to the DBLP record once a paper is
   published. Preprints are re-checked for a published version for 18 months.
3. A free keyword prefilter drops obvious noise, then **Claude** decides relevance, picks the
   section, and writes a headline + short blurb. Without an API key the script falls back to
   the paper title and the first sentences of the abstract.
4. Results go to `docs/papers.json` (the site) and `data/seen.json` (so no paper is ever
   paid for twice).

Google Scholar has no API and blocks automated access, and DBLP's API currently blocks
bots too, so DBLP links come through Semantic Scholar.

## Settings (GitHub → Settings → Secrets and variables → Actions)

| Name | Kind | Default | Meaning |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | secret | - | Claude API key; without it the free fallback is used |
| `MODEL` | variable | `claude-opus-5` | e.g. `claude-sonnet-5` or `claude-haiku-4-5` for lower cost |
| `MAX_PAPERS` | variable | `60` | max papers sent to Claude per run (hard cost cap) |
| `S2_API_KEY` | secret | - | optional free Semantic Scholar key (semanticscholar.org/product/api); makes venue lookups more reliable |

Change the schedule in `.github/workflows/update.yml` (`cron`). Weekly: `"0 5 * * 1"`.
You can also trigger a run by hand: Actions → Update papers → Run workflow.

## Local run

    pip install anthropic
    python scripts/update.py
    python -m http.server -d docs     # then open http://localhost:8000
