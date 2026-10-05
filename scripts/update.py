"""Fetch new fairness-in-AI papers, write news-style blurbs, update docs/papers.json.

Sources
  * arXiv API            - new preprints (cs.LG, cs.AI, cs.CY, cs.GT, cs.MA, stat.ML, econ.TH)
  * Semantic Scholar API - (a) venue lookup for arXiv papers ("published at AAAI 2026"),
                           incl. the DBLP record key; (b) newly indexed papers from top venues.
  (Google Scholar has no API and blocks bots; DBLP's own API currently blocks automated
   clients, so DBLP links come via Semantic Scholar's externalIds.)

Blurbs: Claude (if ANTHROPIC_API_KEY is set), otherwise a free fallback that uses the
paper's own title and the first sentences of the abstract.

Usage:  python scripts/update.py   (env: ANTHROPIC_API_KEY, MODEL, MAX_PAPERS, optional S2_API_KEY)
"""

import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS_FILE = ROOT / "docs" / "papers.json"
SEEN_FILE = ROOT / "data" / "seen.json"

MODEL = os.environ.get("MODEL") or "claude-opus-5"
MAX_PAPERS = int(os.environ.get("MAX_PAPERS") or 60)  # cap on papers sent to Claude per run
LOOKBACK_DAYS = 10            # how far back to look for arXiv papers (dedup via seen.json)
VENUE_LOOKBACK_DAYS = 365     # venue papers: only consider recently published ones
KEEP_DAYS = 365               # drop stories older than this from the site
RECHECK_DAYS = 540            # keep checking preprints for a venue for this long
UA = "fairness-ai-news/1.0 (+https://github.com/)"

ARXIV_CATS = ["cs.LG", "cs.AI", "cs.CY", "cs.GT", "cs.MA", "stat.ML", "econ.TH", "cs.HC", "cs.CL"]
ARXIV_TERMS = [
    "fairness", "fair", "unfair", "unfairness", "bias", "discrimination", "equity",
    "envy-free", "envy-freeness", "proportionality", "proportional representation",
    "justified representation", "fair division", "social choice", "participatory budgeting",
    "apportionment", "committee voting", "approval voting", "core stability", "maximin share",
]

# Top venues for the "newly published" feed (Semantic Scholar venue names / abbreviations).
S2_VENUES = [
    "FAccT", "Conference on Fairness, Accountability and Transparency", "AIES",
    "AAAI", "IJCAI", "NeurIPS", "ICML", "ICLR", "AAMAS", "EC",
    "ACM Conference on Economics and Computation", "WINE", "SAGT", "COMSOC",
    "EAAMO", "Journal of Artificial Intelligence Research", "Artificial Intelligence",
    "Social Choice and Welfare", "Games and Economic Behavior", "ECAI", "KDD", "WWW", "TMLR",
]
S2_QUERY = ("fair | fairness | unfair | envy | proportional | proportionality | "
            "\"social choice\" | voting | discrimination | bias | equity | apportionment")

# Prefilter (free, before any tokens are spent): a fairness term in the title,
# or at least two strong fairness terms in the abstract.
PREFILTER = re.compile(
    r"\b(fair\w*|unfair\w*|bias(es|ed)?|debias\w*|discriminat\w*|equit\w*|disparit\w*|"
    r"envy\w*|proportional\w*|justified representation|social choice|voting|voter\w*|"
    r"participatory budgeting|apportionment|maximin share|demographic parity|"
    r"equali[sz]ed odds|protected (attribute|group)s?|allocation of indivisible)\b",
    re.I,
)
STRONG_TERMS = re.compile(
    r"(fair\w*|unfair\w*|envy\w*|proportional representation|justified representation|"
    r"social choice|participatory budgeting|apportionment|maximin share|discriminat\w*|"
    r"debias\w*|demographic parity|equali[sz]ed odds|disparate)", re.I)
SOCIAL_CHOICE_HINT = re.compile(
    r"\b(envy\w*|proportional\w*|justified representation|social choice|voting|voter\w*|"
    r"participatory budgeting|apportionment|maximin share|indivisible|fair division|"
    r"committee|allocation|matching|mechanism)\b", re.I)
STRONG_ML_HINT = re.compile(
    r"\b(algorithmic fairness|group fairness|individual fairness|fair(ness)?[- ]aware|"
    r"demographic parity|equali[sz]ed odds|debias\w*|disparate impact|protected attributes?|"
    r"fair (classification|ranking|representation|clustering|regression|machine learning))\b",
    re.I)

VENUE_SHORT = {
    "conference on fairness, accountability and transparency": "FAccT",
    "aaai/acm conference on ai, ethics, and society": "AIES",
    "aaai conference on artificial intelligence": "AAAI",
    "international joint conference on artificial intelligence": "IJCAI",
    "neural information processing systems": "NeurIPS",
    "international conference on machine learning": "ICML",
    "international conference on learning representations": "ICLR",
    "adaptive agents and multi-agent systems": "AAMAS",
    "acm conference on economics and computation": "EC",
    "conference on web and internet economics": "WINE",
    "european conference on artificial intelligence": "ECAI",
    "knowledge discovery and data mining": "KDD",
    "the web conference": "WWW",
    "journal of artificial intelligence research": "JAIR",
    "trans. mach. learn. res.": "TMLR",
    "transactions on machine learning research": "TMLR",
    "equity and access in algorithms, mechanisms, and optimization": "EAAMO",
}


# --------------------------------------------------------------------------- utilities

def log(*a):
    print(*a, file=sys.stderr, flush=True)


def http(url, data=None, headers=None, tries=7):
    h = {"User-Agent": UA, **(headers or {})}
    if "semanticscholar.org" in url and os.environ.get("S2_API_KEY"):
        h["x-api-key"] = os.environ["S2_API_KEY"]
    body = json.dumps(data).encode() if data is not None else None
    if body is not None:
        h["Content-Type"] = "application/json"
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=body, headers=h)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                wait = min(5 * 2 ** attempt, 120)
                log(f"  HTTP {e.code} from {url[:70]}... retrying in {wait}s")
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError:
            if attempt < tries - 1:
                time.sleep(5 * 2 ** attempt)
                continue
            raise


def load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def squash(s):
    return re.sub(r"\s+", " ", s or "").strip()


def short_venue(name):
    if not name:
        return ""
    low = name.lower()
    for k, v in VENUE_SHORT.items():
        if k in low:
            return v
    return name


def authors_str(names):
    names = [n for n in names if n]
    if len(names) > 3:
        return ", ".join(names[:3]) + " et al."
    return ", ".join(names)


def passes_prefilter(p):
    return bool(PREFILTER.search(p["title"])) or len(STRONG_TERMS.findall(p["abstract"])) >= 2


# --------------------------------------------------------------------------- sources

def fetch_arxiv(since):
    cats = " OR ".join(f"cat:{c}" for c in ARXIV_CATS)
    terms = " OR ".join(f'abs:"{t}"' if " " in t else f"abs:{t}" for t in ARXIV_TERMS)
    query = f"({cats}) AND ({terms})"
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    out, start, page = [], 0, 200
    while start < 2000:
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
            "search_query": query, "sortBy": "submittedDate", "sortOrder": "descending",
            "start": start, "max_results": page})
        root = ET.fromstring(http(url))
        entries = root.findall("a:entry", ns)
        if not entries:
            break
        done = False
        for e in entries:
            published = e.findtext("a:published", "", ns)
            if datetime.fromisoformat(published.replace("Z", "+00:00")) < since:
                done = True
                break
            aid = e.findtext("a:id", "", ns).rsplit("/abs/", 1)[-1]
            aid = re.sub(r"v\d+$", "", aid)
            out.append({
                "id": f"arxiv:{aid}",
                "arxiv": aid,
                "title": squash(e.findtext("a:title", "", ns)),
                "abstract": squash(e.findtext("a:summary", "", ns)),
                "authors": [squash(a.findtext("a:name", "", ns)) for a in e.findall("a:author", ns)],
                "url": f"https://arxiv.org/abs/{aid}",
                "date": published[:10],
                "venue": "", "venue_year": None, "dblp": None,
            })
        if done or len(entries) < page:
            break
        start += page
        time.sleep(3)  # arXiv asks for >= 3 s between calls
    log(f"arXiv: {len(out)} papers since {since.date()}")
    return out


def fetch_s2_venue_papers(since):
    fields = "title,abstract,venue,year,externalIds,publicationDate,authors,url"
    params = {"query": S2_QUERY, "fields": fields, "venue": ",".join(S2_VENUES),
              "publicationDateOrYear": f"{since.date()}:", "sort": "publicationDate:desc"}
    out, token = [], None
    for _ in range(5):  # up to 5000 results
        if token:
            params["token"] = token
        res = json.loads(http("https://api.semanticscholar.org/graph/v1/paper/search/bulk?"
                              + urllib.parse.urlencode(params)))
        for p in res.get("data", []):
            ext = p.get("externalIds") or {}
            if not p.get("abstract"):
                continue
            pid = f"doi:{ext['DOI'].lower()}" if ext.get("DOI") else f"s2:{p['paperId']}"
            out.append({
                "id": pid,
                "arxiv": ext.get("ArXiv"),
                "title": squash(p.get("title")),
                "abstract": squash(p.get("abstract")),
                "authors": [a.get("name") for a in p.get("authors") or []],
                "url": f"https://doi.org/{ext['DOI']}" if ext.get("DOI") else p.get("url"),
                "date": p.get("publicationDate") or str(p.get("year") or ""),
                "venue": short_venue(p.get("venue")),
                "venue_year": p.get("year"),
                "dblp": ext.get("DBLP"),
            })
        token = res.get("token")
        if not token:
            break
        time.sleep(2)
    log(f"Semantic Scholar venues: {len(out)} papers since {since.date()}")
    return out


def s2_venue_lookup(arxiv_ids):
    """arXiv id -> {venue, venue_year, dblp}, only for papers published somewhere other than arXiv."""
    found = {}
    ids = list(dict.fromkeys(arxiv_ids))
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        res = json.loads(http(
            "https://api.semanticscholar.org/graph/v1/paper/batch?fields=venue,year,externalIds,journal",
            data={"ids": [f"arXiv:{a}" for a in chunk]}))
        for aid, p in zip(chunk, res):
            if not p:
                continue
            venue = p.get("venue") or ""
            ext = p.get("externalIds") or {}
            dblp = ext.get("DBLP")
            if venue.lower() in ("", "arxiv.org", "arxiv") or (dblp or "").startswith("journals/corr"):
                continue
            found[aid] = {"venue": short_venue(venue), "venue_year": p.get("year"), "dblp": dblp}
        time.sleep(2)
    return found


# --------------------------------------------------------------------------- blurbs

SYSTEM = """You are the editor of a newspaper-style digest of new research on fairness in AI, \
read by researchers. It has two sections:
  "ml"           - algorithmic fairness in machine learning and AI systems: bias measurement and \
mitigation, fair classification/ranking/recommendation, fairness of LLMs, auditing, \
fairness-related policy/regulation of AI, fairness in the deployment of AI.
  "socialchoice" - computational social choice and fair division: fair allocation (envy-freeness, \
proportionality, MMS), voting and committee elections (justified representation, proportional \
representation), participatory budgeting, apportionment, fair matching, fairness in mechanism design.

For each paper decide:
- relevant: true only if fairness/equity (or social choice) is a central topic of the paper. \
False if "fair" is incidental (e.g. "a fair comparison"), if "bias" means statistical/inductive bias, \
or if the paper is about something else entirely.
- section: "ml" or "socialchoice" (for irrelevant papers pick whichever is closer).
- headline: BBC-style news headline, at most 12 words, plain English, accurate, no hype, no question \
marks, no clickbait. Say what was found or built, not "Researchers study...".
- blurb: one or two sentences, at most 40 words, explaining what the paper does and why it matters, \
for a researcher skimming the page. Stay strictly faithful to the abstract; do not invent results.
- score: 1-5, how interesting/significant it looks to a fairness researcher (5 = likely to be widely read).
For irrelevant papers, headline and blurb may be empty strings."""

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "relevant": {"type": "boolean"},
            "section": {"type": "string", "enum": ["ml", "socialchoice"]},
            "headline": {"type": "string"},
            "blurb": {"type": "string"},
            "score": {"type": "integer"},
        },
        "required": ["id", "relevant", "section", "headline", "blurb", "score"],
        "additionalProperties": False,
    }}},
    "required": ["items"],
    "additionalProperties": False,
}


def claude_blurbs(papers):
    import anthropic

    client = anthropic.Anthropic()
    use_fallbacks = MODEL.startswith(("claude-opus-5", "claude-fable-5"))
    results, usage = {}, {"input": 0, "output": 0}
    for i in range(0, len(papers), 15):
        chunk = papers[i:i + 15]
        listing = "\n\n".join(
            f"<paper id=\"{p['id']}\">\nTitle: {p['title']}\nAbstract: {p['abstract']}\n</paper>"
            for p in chunk)
        kwargs = dict(
            model=MODEL, max_tokens=16000, system=SYSTEM,
            messages=[{"role": "user", "content": f"Papers:\n\n{listing}\n\nReturn one item per paper id."}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        if not MODEL.startswith("claude-haiku"):
            kwargs["output_config"]["effort"] = "low"
        try:
            if use_fallbacks:
                resp = client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                resp = client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            log(f"  Claude API error ({e.status_code}): {e.message} - using free fallback for this chunk")
            continue
        usage["input"] += resp.usage.input_tokens
        usage["output"] += resp.usage.output_tokens
        if resp.stop_reason in ("refusal", "max_tokens"):
            log(f"  chunk stopped with {resp.stop_reason} - using free fallback for this chunk")
            continue
        text = next(b.text for b in resp.content if b.type == "text")
        for item in json.loads(text)["items"]:
            results[item["id"]] = item
    log(f"Claude ({MODEL}): {usage['input']} input / {usage['output']} output tokens")
    return results


def free_blurb(p):
    text = f"{p['title']} {p['abstract']}"
    sentences = re.split(r"(?<=[.!?])\s+", p["abstract"])
    blurb = ""
    for s in sentences:
        if len((blurb + " " + s).split()) > 45 and blurb:
            break
        blurb = (blurb + " " + s).strip()
    sc = len(SOCIAL_CHOICE_HINT.findall(text))
    ml = len(STRONG_ML_HINT.findall(text))
    return {
        "id": p["id"],
        "relevant": bool(re.search(r"\bfair(ness)?\b|\bunfair|\bbias|envy|proportional|social choice|voting",
                                   p["title"], re.I) or STRONG_ML_HINT.search(text)),
        "section": "socialchoice" if sc > ml else "ml",
        "headline": p["title"],
        "blurb": blurb,
        "score": 2,
    }


# --------------------------------------------------------------------------- main

def main():
    now = datetime.now(timezone.utc)
    site = load_json(PAPERS_FILE, {"updated": None, "papers": []})
    seen = set(load_json(SEEN_FILE, []))
    first_run = not seen
    stories = {p["id"]: p for p in site["papers"]}
    known_arxiv = {p.get("arxiv") for p in site["papers"] if p.get("arxiv")}

    # 1. gather candidates
    candidates = fetch_arxiv(now - timedelta(days=LOOKBACK_DAYS))
    try:
        venue_papers = fetch_s2_venue_papers(now - timedelta(days=VENUE_LOOKBACK_DAYS))
    except Exception as e:  # never let one source kill the run
        log(f"Semantic Scholar venue search failed: {e}")
        venue_papers = []
    if first_run:
        # Don't flood the first edition with a year of backlog: keep only the last 3 weeks.
        cutoff = str((now - timedelta(days=21)).date())
        for p in venue_papers:
            if p["date"] < cutoff:
                seen.add(p["id"])
    arxiv_ids_in_batch = {c["arxiv"] for c in candidates}
    for p in venue_papers:  # skip venue versions of preprints we already cover
        if p["arxiv"] and (p["arxiv"] in known_arxiv or p["arxiv"] in arxiv_ids_in_batch):
            seen.add(p["id"])
    candidates += venue_papers

    fresh, dup = [], set()
    for c in candidates:
        if c["id"] in seen or c["id"] in dup:
            continue
        dup.add(c["id"])
        if not passes_prefilter(c):
            seen.add(c["id"])
            continue
        fresh.append(c)
    # newest first; cap what we send to the model, the rest waits for the next run
    fresh.sort(key=lambda c: c["date"], reverse=True)
    batch = fresh[:MAX_PAPERS]
    log(f"{len(fresh)} new candidates after prefilter; processing {len(batch)}")

    # 2. venue lookup for new preprints + older preprints still without a venue
    recheck_cutoff = str((now - timedelta(days=RECHECK_DAYS)).date())
    to_check = [c["arxiv"] for c in batch if c.get("arxiv") and not c["venue"]]
    time.sleep(5)  # Semantic Scholar's anonymous pool is ~1 request/second, shared
    to_check += [p["arxiv"] for p in stories.values()
                 if p.get("arxiv") and not p.get("venue") and p["date"] >= recheck_cutoff]
    try:
        venues = s2_venue_lookup(to_check) if to_check else {}
    except Exception as e:
        log(f"Venue lookup failed: {e}")
        venues = {}
    log(f"Venue lookup: {len(venues)} of {len(to_check)} preprints have a published version")
    for p in list(batch) + list(stories.values()):
        v = venues.get(p.get("arxiv"))
        if v and not p.get("venue"):
            p.update(v)

    # 3. headlines & blurbs
    judged = {}
    if batch and os.environ.get("ANTHROPIC_API_KEY"):
        judged = claude_blurbs(batch)
    for p in batch:
        j = judged.get(p["id"]) or free_blurb(p)
        seen.add(p["id"])
        if not j["relevant"] or not j["headline"]:
            continue
        stories[p["id"]] = {
            "id": p["id"], "arxiv": p.get("arxiv"), "title": p["title"],
            "headline": j["headline"], "blurb": j["blurb"], "section": j["section"],
            "score": max(1, min(5, j["score"])), "authors": authors_str(p["authors"]),
            "url": p["url"], "date": p["date"], "added": str(now.date()),
            "venue": p.get("venue") or "", "venue_year": p.get("venue_year"), "dblp": p.get("dblp"),
            "ai": p["id"] in judged,
        }

    # 4. write
    keep_cutoff = str((now - timedelta(days=KEEP_DAYS)).date())
    papers = sorted((p for p in stories.values() if p["added"] >= keep_cutoff),
                    key=lambda p: (p["added"], p["score"], p["date"]), reverse=True)
    save_json(PAPERS_FILE, {"updated": now.isoformat(timespec="minutes"), "papers": papers})
    save_json(SEEN_FILE, sorted(seen))
    log(f"Site now has {len(papers)} stories ({sum(p['added'] == str(now.date()) for p in papers)} added today)")


if __name__ == "__main__":
    main()
