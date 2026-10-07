# The Fairness Dispatch

A weekly, newspaper-style digest of new research on **algorithmic fairness** and
**computational social choice / fair division**, built by GitHub Actions and served by
GitHub Pages from `docs/`.

## How it works

Three sections: **Algorithmic fairness**, **Social choice**, and **Alignment & preference
learning**. `scripts/update.py` runs every Monday (`.github/workflows/update.yml`):

1. **Candidates.**
   - Crossref: the full IJCAI (+KR) and AAAI/AIES/ICWSM/HCOMP proceedings, with abstracts.
   - Semantic Scholar: other top venues (FAccT, NeurIPS, ICML, ICLR, AAMAS, EC, WINE, JAIR,
     ACL, EMNLP, ...).
   - arXiv: preprints from the last two weeks.
   - Peer-reviewed papers are taken from the last 52 weeks. Every preprint gets a venue
     lookup, so a published preprint counts as peer-reviewed.
2. **Free prefilter.** A topic term must appear in the title, or at least two strong topic
   terms in the abstract.
3. **Triage.** Claude reads the title and the opening of the abstract, then decides whether
   the paper is relevant and which section it belongs to. Each paper is triaged only once
   (`data/triage.json`), with at most `MAX_TRIAGE` papers per run.
4. **Selection.** At most `MAX_PER_WEEK` (20) write-ups per ISO week, and every section is
   guaranteed `MIN_PER_SECTION` (5) of them. Within each section, peer-reviewed papers come
   first (newest first), then preprints. Remaining slots go to the best papers overall.
5. **Write-up.** Claude (Sonnet 5) writes the headline, blurb and category for each chosen
   paper. Without an API key, the script falls back to keyword rules plus the paper's own
   title and abstract.

Why Crossref: Semantic Scholar lists IJCAI 2026 under the raw name "Proceedings of the
Thirty-Fifth International Joint Conference ...", which its venue filter does not match,
and some IJCAI papers are missing from Semantic Scholar entirely. Google Scholar has no API,
and DBLP's API blocks automated clients.

## Settings (Settings → Secrets and variables → Actions)

| Name | Kind | Default | Meaning |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | secret | - | Claude API key; without it the free fallback is used |
| `MODEL` | variable | `claude-sonnet-5` | e.g. `claude-haiku-4-5` (cheaper) or `claude-opus-5` |
| `MAX_PER_WEEK` | variable | `20` | hard cap on papers processed per week |
| `PEER_LOOKBACK_WEEKS` | variable | `52` | how far back to look for peer-reviewed papers |
| `MIN_PER_SECTION` | variable | `5` | guaranteed write-ups per section per week |
| `MAX_TRIAGE` | variable | `200` | max papers triaged (title + abstract opening) per run |
| `S2_API_KEY` | secret | - | optional free Semantic Scholar key; more reliable venue lookups |

To run an extra update by hand, go to Actions → Update papers → Run workflow. The weekly
cap still applies.

## Local run

    pip install anthropic
    python scripts/update.py
    python -m http.server -d docs     # then open http://localhost:8000
