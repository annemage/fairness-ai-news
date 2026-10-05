"""Fetch new fairness-in-AI papers, write news-style blurbs, update docs/papers.json.

Runs once a week. At most MAX_PER_WEEK papers (default 20) are processed per ISO week,
however often the script is started - this is the hard cost cap.

Priority for those slots:
  1. peer-reviewed papers (top venues via Semantic Scholar, or arXiv papers that have
     since been published), newest week first, going back up to PEER_LOOKBACK_WEEKS;
  2. then arXiv preprints, newest first, if slots remain.

Sources
  * arXiv API            - preprints (cs.LG, cs.AI, cs.CY, cs.GT, cs.MA, stat.ML, econ.TH, ...)
  * Semantic Scholar API - papers from top venues, and venue lookup for arXiv papers
                           ("published at AAAI 2026"), incl. the DBLP record key.
  (Google Scholar has no API and blocks bots; DBLP's own API currently blocks automated
   clients, so DBLP links come via Semantic Scholar's externalIds.)

Blurbs: Claude (if ANTHROPIC_API_KEY is set), otherwise a free fallback that uses the
paper's own title and the first sentences of the abstract.

Usage:  python scripts/update.py
        env: ANTHROPIC_API_KEY, MODEL, MAX_PER_WEEK, PEER_LOOKBACK_WEEKS, optional S2_API_KEY
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
BUDGET_FILE = ROOT / "data" / "budget.json"

MODEL = os.environ.get("MODEL") or "claude-sonnet-5"
MAX_PER_WEEK = int(os.environ.get("MAX_PER_WEEK") or 20)  # hard cap: papers processed per ISO week
ARXIV_LOOKBACK_DAYS = 14      # preprints: last two weeks (dedup via seen.json)
PEER_LOOKBACK_WEEKS = int(os.environ.get("PEER_LOOKBACK_WEEKS") or 26)  # peer-reviewed: fill up from earlier weeks
KEEP_DAYS = 730               # drop stories older than this from the site
RECHECK_DAYS = 540            # keep checking preprints for a published version this long
UA = "fairness-ai-news/1.0 (+https://github.com/annemage/fairness-ai-news)"

# Categories: key -> (section, label, guidance for the editor model)
CATEGORIES = {
    "notions":    ("ml", "Fairness notions & theory",
                   "definitions and metrics of fairness, impossibility results, causal/counterfactual "
                   "fairness, theoretical analysis of fairness criteria"),
    "methods":    ("ml", "Bias mitigation",
                   "methods to train or post-process fair models: fair classification, regression, "
                   "representation learning, clustering, ranking/recommendation algorithms"),
    "genai":      ("ml", "LLMs & generative AI",
                   "bias, stereotypes and fairness in large language models, image/video generators "
                   "and other foundation models"),
    "audits":     ("ml", "Audits & benchmarks",
                   "measuring bias in deployed or existing systems, evaluation protocols, datasets "
                   "and benchmarks for fairness"),
    "apps":       ("ml", "Applications",
                   "fairness in a specific application domain: health, lending, hiring, education, "
                   "criminal justice, recommender systems, public sector"),
    "policy":     ("ml", "Policy, law & society",
                   "regulation, law, ethics, governance, participatory design, sociotechnical and "
                   "philosophical perspectives on AI fairness"),
    "voting":     ("sc", "Voting & elections",
                   "single-winner and committee voting, proportional representation, justified "
                   "representation, apportionment, liquid democracy, AI alignment via social choice"),
    "pb":         ("sc", "Participatory budgeting",
                   "participatory budgeting and other collective budget decisions"),
    "division":   ("sc", "Fair division",
                   "allocation of goods, chores or cake: envy-freeness, proportionality, maximin "
                   "share, equitability, rent division"),
    "matching":   ("sc", "Matching & markets",
                   "stable matching, school choice, two-sided markets, kidney exchange, fairness in "
                   "matching platforms"),
    "mechanisms": ("sc", "Mechanism design & games",
                   "fairness in mechanism design, cooperative games and cost sharing, fair "
                   "scheduling, online and dynamic fair allocation, fairness in multi-agent systems"),
}
SECTION_LABEL = {"ml": "Algorithmic fairness", "sc": "Social choice & fair division"}

ARXIV_CATS = ["cs.LG", "cs.AI", "cs.CY", "cs.GT", "cs.MA", "stat.ML", "econ.TH", "cs.HC", "cs.CL"]
ARXIV_TERMS = [
    "fairness", "fair", "unfair", "unfairness", "bias", "discrimination", "equity",
    "envy-free", "envy-freeness", "proportionality", "proportional representation",
    "justified representation", "fair division", "social choice", "participatory budgeting",
    "apportionment", "committee voting", "approval voting", "core stability", "maximin share",
    "stable matching",
]

# Top venues for the peer-reviewed feed (Semantic Scholar venue names / abbreviations).
S2_VENUES = [
    "FAccT", "Conference on Fairness, Accountability and Transparency", "AIES",
    "AAAI", "IJCAI", "NeurIPS", "ICML", "ICLR", "AAMAS", "EC",
    "ACM Conference on Economics and Computation", "WINE", "SAGT", "COMSOC",
    "EAAMO", "Journal of Artificial Intelligence Research", "Artificial Intelligence",
    "Social Choice and Welfare", "Games and Economic Behavior", "ECAI", "KDD", "WWW", "TMLR",
]
S2_QUERY = ("fair | fairness | unfair | envy | proportional | proportionality | "
            "\"social choice\" | voting | discrimination | bias | equity | apportionment | "
            "\"participatory budgeting\" | \"stable matching\"")

# Prefilter (free, before any tokens are spent): a fairness term in the title,
# or at least two strong fairness terms in the abstract.
PREFILTER = re.compile(
    r"\b(fair\w*|unfair\w*|bias(es|ed)?|debias\w*|discriminat\w*|equit\w*|disparit\w*|"
    r"envy\w*|proportional\w*|justified representation|social choice|voting|voter\w*|"
    r"participatory budgeting|apportionment|maximin share|demographic parity|stable matching|"
    r"equali[sz]ed odds|protected (attribute|group)s?|allocation of indivisible)\b",
    re.I,
)
STRONG_TERMS = re.compile(
    r"\b(fair\w*|unfair\w*|envy\w*|proportional representation|justified representation|"
    r"social choice|participatory budgeting|apportionment|maximin share|discriminat\w*|"
    r"debias\w*|demographic parity|equali[sz]ed odds|disparate|stable matching)\b", re.I)

# Free fallback categoriser (only used without an API key / if a Claude call fails)
FALLBACK_RULES = [
    ("pb", r"participatory budget"),
    ("voting", r"\b(voting|voter|election|committee|apportionment|justified representation|"
               r"proportional representation|social choice)"),
    ("division", r"\b(envy|indivisible|fair division|fair allocation|proportionality|cake|chores|maximin share|rent division)"),
    ("matching", r"\b(matching|school choice|two-sided market|kidney)"),
    ("mechanisms", r"\b(mechanism|cooperative game|cost sharing|scheduling|online allocation)"),
    ("genai", r"\b(llm|language model|generative|diffusion|text-to-image|foundation model|chatbot)"),
    ("policy", r"\b(regulat|law|legal|polic|governance|ethic|sociotechnical)"),
    ("audits", r"\b(audit|benchmark|dataset|evaluat|measur)"),
    ("apps", r"\b(health|clinical|medical|lending|credit|hiring|recruit|education|criminal|recommend)"),
    ("notions", r"\b(definition|notion|impossib|counterfactual fairness|causal fairness|metric)"),
]

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


def strength(p):
    """Cheap relevance score used to order candidates within a priority tier."""
    return 3 * len(STRONG_TERMS.findall(p["title"])) + len(STRONG_TERMS.findall(p["abstract"]))


# --------------------------------------------------------------------------- sources

def fetch_arxiv(since):
    cats = " OR ".join(f"cat:{c}" for c in ARXIV_CATS)
    terms = " OR ".join(f'abs:"{t}"' if " " in t else f"abs:{t}" for t in ARXIV_TERMS)
    query = f"({cats}) AND ({terms})"
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out, start, page = [], 0, 200
    while start < 3000:
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
            aid = re.sub(r"v\d+$", "", e.findtext("a:id", "", ns).rsplit("/abs/", 1)[-1])
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
            if not p.get("abstract") or not p.get("publicationDate"):
                continue
            pid = f"doi:{ext['DOI'].lower()}" if ext.get("DOI") else f"s2:{p['paperId']}"
            out.append({
                "id": pid,
                "arxiv": ext.get("ArXiv"),
                "title": squash(p.get("title")),
                "abstract": squash(p.get("abstract")),
                "authors": [a.get("name") for a in p.get("authors") or []],
                "url": f"https://doi.org/{ext['DOI']}" if ext.get("DOI") else p.get("url"),
                "date": p["publicationDate"],
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
            "https://api.semanticscholar.org/graph/v1/paper/batch?fields=venue,year,externalIds",
            data={"ids": [f"arXiv:{a}" for a in chunk]}))
        for aid, p in zip(chunk, res):
            if not p:
                continue
            venue = p.get("venue") or ""
            dblp = (p.get("externalIds") or {}).get("DBLP")
            if venue.lower() in ("", "arxiv.org", "arxiv") or (dblp or "").startswith("journals/corr"):
                continue
            found[aid] = {"venue": short_venue(venue), "venue_year": p.get("year"), "dblp": dblp}
        time.sleep(2)
    return found


# --------------------------------------------------------------------------- blurbs

def system_prompt():
    cats = "\n".join(f'  "{k}" ({SECTION_LABEL[s]} / {label}): {desc}'
                     for k, (s, label, desc) in CATEGORIES.items())
    return f"""You are the editor of a newspaper-style weekly digest of new research on fairness in AI, \
read by researchers. Papers fall into two sections - algorithmic fairness in ML/AI, and \
computational social choice & fair division - each split into categories:
{cats}

For each paper decide:
- relevant: true only if fairness/equity or social choice is a central topic of the paper. \
False if "fair" is incidental (e.g. "a fair comparison"), if "bias" means statistical/inductive bias, \
or if the paper is about something else entirely.
- category: the single best-fitting category key from the list above.
- headline: newspaper headline, at most 12 words, plain English, accurate, no hype, no question \
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
            "category": {"type": "string", "enum": list(CATEGORIES)},
            "headline": {"type": "string"},
            "blurb": {"type": "string"},
            "score": {"type": "integer"},
        },
        "required": ["id", "relevant", "category", "headline", "blurb", "score"],
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
    for i in range(0, len(papers), 20):
        chunk = papers[i:i + 20]
        listing = "\n\n".join(
            f"<paper id=\"{p['id']}\">\nTitle: {p['title']}\nAbstract: {p['abstract']}\n</paper>"
            for p in chunk)
        kwargs = dict(
            model=MODEL, max_tokens=16000, system=system_prompt(),
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
        except anthropic.APIConnectionError as e:
            log(f"  Claude API unreachable: {e} - using free fallback for this chunk")
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
    blurb = ""
    for s in re.split(r"(?<=[.!?])\s+", p["abstract"]):
        if len((blurb + " " + s).split()) > 45 and blurb:
            break
        blurb = (blurb + " " + s).strip()
    category = next((k for k, rx in FALLBACK_RULES if re.search(rx, p["title"], re.I)), None) \
        or next((k for k, rx in FALLBACK_RULES if re.search(rx, text, re.I)), "methods")
    return {
        "id": p["id"],
        "relevant": bool(re.search(r"\bfair(ness)?\b|\bunfair|\bbias|envy|proportional|social choice|"
                                   r"voting|participatory|matching", p["title"], re.I)),
        "category": category, "headline": p["title"], "blurb": blurb, "score": 2,
    }


# --------------------------------------------------------------------------- main

def main():
    now = datetime.now(timezone.utc)
    week = "%d-W%02d" % now.isocalendar()[:2]
    site = load_json(PAPERS_FILE, {"updated": None, "edition": 0, "papers": []})
    seen = set(load_json(SEEN_FILE, []))
    budget = load_json(BUDGET_FILE, {})
    used = budget.get("used", 0) if budget.get("week") == week else 0
    slots = max(0, MAX_PER_WEEK - used)
    stories = {p["id"]: p for p in site["papers"]}
    known_arxiv = {p.get("arxiv") for p in site["papers"] if p.get("arxiv")}
    log(f"Week {week}: {used} of {MAX_PER_WEEK} papers already used, {slots} slots left")

    # 1. gather candidates
    try:
        venue_papers = fetch_s2_venue_papers(now - timedelta(weeks=PEER_LOOKBACK_WEEKS))
    except Exception as e:  # never let one source kill the run
        log(f"Semantic Scholar venue search failed: {e}")
        venue_papers = []
    try:
        preprints = fetch_arxiv(now - timedelta(days=ARXIV_LOOKBACK_DAYS))
    except Exception as e:
        log(f"arXiv search failed: {e}")
        preprints = []

    venue_by_arxiv = {p["arxiv"]: p for p in venue_papers if p["arxiv"]}
    candidates, ids = [], set()
    for c in venue_papers + preprints:
        if c["id"] in seen or c["id"] in ids or c.get("arxiv") in known_arxiv:
            continue
        if c["id"].startswith("arxiv:") and c["arxiv"] in venue_by_arxiv:
            continue  # the peer-reviewed version is already a candidate
        ids.add(c["id"])
        if not passes_prefilter(c):
            seen.add(c["id"])
            continue
        candidates.append(c)

    # 2. venue lookup: new preprints (might already be published) + older preprint stories
    recheck_cutoff = str((now - timedelta(days=RECHECK_DAYS)).date())
    to_check = [c["arxiv"] for c in candidates if c.get("arxiv") and not c["venue"]]
    to_check += [p["arxiv"] for p in stories.values()
                 if p.get("arxiv") and not p.get("venue") and p["date"] >= recheck_cutoff]
    time.sleep(5)  # Semantic Scholar's anonymous pool is ~1 request/second, shared
    try:
        venues = s2_venue_lookup(to_check) if to_check else {}
    except Exception as e:
        log(f"Venue lookup failed: {e}")
        venues = {}
    log(f"Venue lookup: {len(venues)} of {len(to_check)} preprints have a published version")
    for p in candidates + list(stories.values()):
        v = venues.get(p.get("arxiv"))
        if v and not p.get("venue"):
            p.update(v)

    # 3. choose this week's papers: peer-reviewed first (newest week first, then strongest
    #    keyword match), then preprints
    def priority(c):
        weeks_ago = (now.date() - datetime.fromisoformat(c["date"][:10]).date()).days // 7
        return (0 if c["venue"] else 1, max(weeks_ago, 0), -strength(c))
    candidates.sort(key=priority)
    batch = candidates[:slots]
    n_peer = sum(1 for c in batch if c["venue"])
    log(f"{len(candidates)} candidates ({sum(1 for c in candidates if c['venue'])} peer-reviewed); "
        f"processing {len(batch)} ({n_peer} peer-reviewed, {len(batch) - n_peer} preprints)")

    # 4. headlines & blurbs
    judged = {}
    if batch and os.environ.get("ANTHROPIC_API_KEY"):
        judged = claude_blurbs(batch)
    edition = site.get("edition", 0) + (1 if batch else 0)
    for p in batch:
        j = judged.get(p["id"]) or free_blurb(p)
        seen.add(p["id"])
        if not j["relevant"] or not j["headline"]:
            continue
        cat = j["category"] if j["category"] in CATEGORIES else "methods"
        stories[p["id"]] = {
            "id": p["id"], "arxiv": p.get("arxiv"), "title": p["title"],
            "headline": j["headline"], "blurb": j["blurb"],
            "section": CATEGORIES[cat][0], "category": cat,
            "score": max(1, min(5, j["score"])), "authors": authors_str(p["authors"]),
            "url": p["url"], "date": p["date"], "added": str(now.date()), "edition": edition,
            "venue": p.get("venue") or "", "venue_year": p.get("venue_year"), "dblp": p.get("dblp"),
            "ai": p["id"] in judged,
        }

    # 5. write
    keep_cutoff = str((now - timedelta(days=KEEP_DAYS)).date())
    papers = sorted((p for p in stories.values() if p["added"] >= keep_cutoff),
                    key=lambda p: (p["added"], p["score"], p["date"]), reverse=True)
    save_json(PAPERS_FILE, {
        "updated": now.isoformat(timespec="minutes"), "edition": edition,
        "categories": {k: {"section": s, "label": label} for k, (s, label, _) in CATEGORIES.items()},
        "sections": SECTION_LABEL, "papers": papers})
    save_json(SEEN_FILE, sorted(seen))
    save_json(BUDGET_FILE, {"week": week, "used": used + len(batch)})
    log(f"Edition {edition}: site has {len(papers)} stories "
        f"({sum(p['added'] == str(now.date()) for p in papers)} added today)")


if __name__ == "__main__":
    main()
