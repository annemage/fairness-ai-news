# The Fairness Dispatch

A weekly, newspaper-style digest of new research on **algorithmic fairness** and
**computational social choice / fair division**, built by GitHub Actions and served by
GitHub Pages from `docs/`.

## How it works

`scripts/update.py` runs every Monday (`.github/workflows/update.yml`):

1. **Candidates.** Semantic Scholar: papers from top venues (FAccT, AIES, AAAI, IJCAI,
   NeurIPS, ICML, ICLR, AAMAS, EC, WINE, SAGT, JAIR, ...) from the last 26 weeks.
   arXiv: preprints from the last two weeks in cs.LG, cs.AI, cs.CY, cs.GT, cs.MA, stat.ML,
   econ.TH, cs.HC, cs.CL. Every preprint gets a venue lookup, so a published preprint
   counts as peer-reviewed and shows "AAAI 2026" with a link to its DBLP record.
2. **Free prefilter.** A fairness term must appear in the title, or at least two strong
   fairness terms in the abstract.
3. **Hard weekly cap.** At most `MAX_PER_WEEK` (20) papers per ISO week go to Claude,
   however often the workflow runs (tracked in `data/budget.json`). The slots are filled
   with peer-reviewed papers first, newest week first, going back week by week, and then
   with arXiv preprints.
4. **Claude** (Sonnet 5) decides relevance, picks one of 11 categories, and writes the
   headline, blurb and an importance score that orders the front page. Without an API key
   (or if the API fails) the script falls back to the paper title and the start of the
   abstract.
5. Output goes to `docs/papers.json` (the site). `data/seen.json` makes sure no paper
   is processed twice.

Google Scholar has no API and blocks bots, and DBLP's API currently blocks automated
clients, so DBLP links come through Semantic Scholar.

### Categories

* Algorithmic fairness: fairness notions & theory, bias mitigation, LLMs & generative AI,
  audits & benchmarks, applications, policy/law & society
* Social choice & fair division: voting & elections, participatory budgeting, fair division,
  matching & markets, mechanism design & games

## Settings (Settings → Secrets and variables → Actions)

| Name | Kind | Default | Meaning |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | secret | - | Claude API key; without it the free fallback is used |
| `MODEL` | variable | `claude-sonnet-5` | e.g. `claude-haiku-4-5` (cheaper) or `claude-opus-5` |
| `MAX_PER_WEEK` | variable | `20` | hard cap on papers processed per week |
| `PEER_LOOKBACK_WEEKS` | variable | `26` | how far back to look for peer-reviewed papers |
| `S2_API_KEY` | secret | - | optional free Semantic Scholar key; more reliable venue lookups |

To run an extra update by hand, go to Actions → Update papers → Run workflow. The weekly
cap still applies.

## Local run

    pip install anthropic
    python scripts/update.py
    python -m http.server -d docs     # then open http://localhost:8000
