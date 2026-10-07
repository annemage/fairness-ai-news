"""Fetch new research papers, write news-style blurbs, update docs/papers.json.

Three sections: algorithmic fairness, social choice, and alignment & preference learning.

Runs once a week. At most MAX_PER_WEEK papers (default 20) get a full write-up per ISO
week, however often the script is started - the hard cost cap. Each section is guaranteed
MIN_PER_SECTION (default 5) of those slots; the rest go to the best remaining candidates.

Priority within a section: peer-reviewed papers first (newest first, going back up to
PEER_LOOKBACK_WEEKS), then arXiv preprints.

Pipeline
  1. Candidates from
     * Crossref         - full proceedings of IJCAI (10.24963) and AAAI/AIES/ICWSM/... (10.1609),
                          with abstracts
     * Semantic Scholar - other top venues (FAccT, NeurIPS, ICML, ICLR, AAMAS, EC, ACL, ...),
                          and venue lookup for arXiv papers, incl. the DBLP record key
     * arXiv            - preprints of the last two weeks
  2. Free keyword prefilter.
  3. Triage: Claude reads title + opening of the abstract and decides relevance and section
     (cached in data/triage.json, so every paper is triaged once; at most MAX_TRIAGE per run).
  4. Selection with per-section minimums, then full headline + blurb for the chosen papers.

Without ANTHROPIC_API_KEY (or if a Claude call fails) keyword rules and the paper's own
title/abstract are used instead.

Usage:  python scripts/update.py
        env: ANTHROPIC_API_KEY, MODEL, MAX_PER_WEEK, MIN_PER_SECTION, MAX_TRIAGE,
             PEER_LOOKBACK_WEEKS, optional S2_API_KEY
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
TRIAGE_FILE = ROOT / "data" / "triage.json"
BUDGET_FILE = ROOT / "data" / "budget.json"
AWARDS_FILE = ROOT / "data" / "awards.json"   # hand-edited: [{"match": doi|arXiv id|title, "award": "..."}]


def env_int(name, default):
    return int(os.environ.get(name) or default)


MODEL = os.environ.get("MODEL") or "claude-sonnet-5"
MAX_PER_WEEK = env_int("MAX_PER_WEEK", 20)        # hard cap: full write-ups per ISO week
MIN_PER_SECTION = env_int("MIN_PER_SECTION", 5)   # guaranteed write-ups per section per week
MAX_TRIAGE = env_int("MAX_TRIAGE", 200)           # cap on papers triaged per run
PEER_LOOKBACK_WEEKS = env_int("PEER_LOOKBACK_WEEKS", 52)
ARXIV_LOOKBACK_DAYS = 14
KEEP_DAYS = 730               # drop stories older than this from the site
RECHECK_DAYS = 540            # keep checking preprints for a published version this long
UA = "fairness-ai-news/1.0 (+https://github.com/annemage/fairness-ai-news)"

SECTIONS = {"ml": "Algorithmic fairness", "sc": "Social choice", "align": "Alignment & preference learning"}

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
                   "representation, apportionment, liquid democracy, judgment aggregation"),
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
    "rlhf":       ("align", "RLHF, DPO & reward models",
                   "reinforcement learning from human feedback, direct preference optimisation and "
                   "variants, reward modelling, reward hacking"),
    "pluralistic": ("align", "Pluralistic alignment",
                   "aligning AI with diverse or conflicting preferences, social choice and voting "
                   "methods for alignment, preference aggregation across annotators or groups"),
    "preflearn":  ("align", "Preference models & ranking",
                   "learning from pairwise comparisons or rankings: Bradley-Terry, Plackett-Luce, "
                   "Mallows, preference elicitation, theory of preference learning"),
    "values":     ("align", "Values, norms & oversight",
                   "value alignment, moral and normative reasoning of AI systems, human oversight, "
                   "evaluation of alignment"),
}
SECTION_DEFAULT = {"ml": "methods", "sc": "voting", "align": "rlhf"}

ARXIV_CATS = ["cs.LG", "cs.AI", "cs.CY", "cs.GT", "cs.MA", "stat.ML", "econ.TH", "cs.HC", "cs.CL"]
ARXIV_TERMS = [
    "fairness", "fair", "unfair", "unfairness", "bias", "discrimination", "equity",
    "envy-free", "envy-freeness", "proportionality", "proportional representation",
    "justified representation", "fair division", "social choice", "participatory budgeting",
    "apportionment", "committee voting", "approval voting", "core stability", "maximin share",
    "stable matching",
    # alignment & preference learning
    "RLHF", "human feedback", "preference optimization", "reward model", "reward modeling",
    "preference learning", "pluralistic alignment", "value alignment", "AI alignment",
    "LLM alignment", "Bradley-Terry", "Plackett-Luce", "Mallows", "preference aggregation",
    "pairwise comparisons",
]

# Venues for the Semantic Scholar feed (IJCAI/AAAI/AIES also come from Crossref, see below).
S2_VENUES = [
    "FAccT", "Conference on Fairness, Accountability and Transparency", "AIES",
    "AAAI", "IJCAI", "NeurIPS", "ICML", "ICLR", "AAMAS", "EC",
    "ACM Conference on Economics and Computation", "WINE", "SAGT", "COMSOC",
    "EAAMO", "Journal of Artificial Intelligence Research", "Artificial Intelligence",
    "Social Choice and Welfare", "Games and Economic Behavior", "ECAI", "KDD", "WWW", "TMLR",
    "ACL", "EMNLP", "NAACL", "COLM", "Journal of Machine Learning Research",
]
S2_QUERY = ("fair | fairness | unfair | envy | proportional | proportionality | "
            "\"social choice\" | voting | discrimination | bias | equity | apportionment | "
            "\"participatory budgeting\" | \"stable matching\" | RLHF | \"human feedback\" | "
            "\"preference optimization\" | \"reward model\" | \"preference learning\" | alignment")

# Crossref DOI prefixes whose proceedings carry abstracts: IJCAI (+KR), AAAI's OJS
# (AAAI, AIES, ICWSM, HCOMP, ICAPS, ...).
CROSSREF_PREFIXES = ["10.24963", "10.1609"]

# Prefilter (free, before any tokens are spent): a topic term in the title,
# or at least two strong topic terms in the abstract.
PREFILTER = re.compile(
    r"\b(fair\w*|unfair\w*|bias(es|ed)?|debias\w*|discriminat\w*|equit\w*|disparit\w*|"
    r"envy\w*|proportional\w*|justified representation|social choice|voting|voter\w*|"
    r"participatory budgeting|apportionment|maximin share|demographic parity|stable matching|"
    r"equali[sz]ed odds|protected (attribute|group)s?|allocation of indivisible|"
    r"rlhf|human feedback|preference (optimi[sz]ation|learning|aggregation|model\w*|data)|"
    r"reward model\w*|dpo|align\w*|bradley-terry|plackett-luce|mallows|pairwise comparisons?)\b",
    re.I,
)
STRONG_TERMS = re.compile(
    r"\b(fair\w*|unfair\w*|envy\w*|proportional representation|justified representation|"
    r"social choice|participatory budgeting|apportionment|maximin share|discriminat\w*|"
    r"debias\w*|demographic parity|equali[sz]ed odds|disparate|stable matching|"
    r"rlhf|human feedback|preference optimi[sz]ation|reward model\w*|preference learning|"
    r"(llm|ai|value|pluralistic) alignment|bradley-terry|plackett-luce|mallows|human preferences?)\b",
    re.I)

# Free fallback rules (only used without an API key / if a Claude call fails)
SECTION_RULES = [
    ("align", r"\b(rlhf|human feedback|preference optimi|reward model|dpo\b|alignment|bradley-terry|"
              r"plackett-luce|mallows|preference learning|pairwise comparison)"),
    ("sc", r"\b(voting|voter|election|committee|apportionment|justified representation|social choice|"
           r"participatory budget|envy|indivisible|fair division|fair allocation|cake|chores|"
           r"maximin share|matching|mechanism|cooperative game)"),
]
CATEGORY_RULES = [
    ("pb", r"participatory budget"),
    ("pluralistic", r"\b(pluralistic|diverse preferences|social choice.*alignment|alignment.*social choice)"),
    ("preflearn", r"\b(bradley-terry|plackett-luce|mallows|pairwise comparison|preference learning)"),
    ("rlhf", r"\b(rlhf|human feedback|preference optimi|reward model|dpo\b)"),
    ("values", r"\b(value alignment|moral|norm|oversight|alignment)"),
    ("voting", r"\b(voting|voter|election|committee|apportionment|justified representation|"
               r"proportional representation|social choice)"),
    ("division", r"\b(envy|indivisible|fair division|fair allocation|proportionality|cake|chores|"
                 r"maximin share|rent division)"),
    ("matching", r"\b(matching|school choice|two-sided market|kidney)"),
    ("mechanisms", r"\b(mechanism|cooperative game|cost sharing|scheduling|online allocation)"),
    ("genai", r"\b(llm|language model|generative|diffusion|text-to-image|foundation model|chatbot)"),
    ("policy", r"\b(regulat|law|legal|polic|governance|ethic|sociotechnical)"),
    ("audits", r"\b(audit|benchmark|dataset|evaluat|measur)"),
    ("apps", r"\b(health|clinical|medical|lending|credit|hiring|recruit|education|criminal|recommend)"),
    ("notions", r"\b(definition|notion|impossib|counterfactual fairness|causal fairness|metric)"),
]

VENUE_SHORT = {
    "conference on fairness accountability and transparency": "FAccT",
    "conference on ai ethics and society": "AIES",
    "aaai conference on artificial intelligence": "AAAI",
    "international joint conference on artificial intelligence": "IJCAI",
    "principles of knowledge representation": "KR",
    "conference on web and social media": "ICWSM",
    "conference on human computation": "HCOMP",
    "conference on automated planning and scheduling": "ICAPS",
    "neural information processing systems": "NeurIPS",
    "international conference on machine learning": "ICML",
    "international conference on learning representations": "ICLR",
    "adaptive agents and multiagent systems": "AAMAS",
    "acm conference on economics and computation": "EC",
    "conference on web and internet economics": "WINE",
    "european conference on artificial intelligence": "ECAI",
    "knowledge discovery and data mining": "KDD",
    "the web conference": "WWW",
    "journal of artificial intelligence research": "JAIR",
    "trans mach learn res": "TMLR",
    "transactions on machine learning research": "TMLR",
    "equity and access in algorithms mechanisms and optimization": "EAAMO",
    "annual meeting of the association for computational linguistics": "ACL",
    "empirical methods in natural language processing": "EMNLP",
    "north american chapter of the association for computational linguistics": "NAACL",
}


# --------------------------------------------------------------------------- utilities

def log(*a):
    print(*a, file=sys.stderr, flush=True)


def http(url, data=None, headers=None, tries=7):
    h = {"User-Agent": UA, **(headers or {})}
    if "semanticscholar.org" in url and os.environ.get("S2_API_KEY"):
        h["x-api-key"] = os.environ["S2_API_KEY"]
    if "semanticscholar.org" in url:
        tries = min(tries, 5)  # its shared anonymous quota can stay busy; don't stall the run
    body = json.dumps(data).encode() if data is not None else None
    if body is not None:
        h["Content-Type"] = "application/json"
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=body, headers=h)
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                wait = min(5 * 2 ** attempt, 120)
                log(f"  HTTP {e.code} from {url[:70]}... retrying in {wait}s")
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
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


def title_key(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:120]


def short_venue(name):
    if not name:
        return ""
    low = re.sub(r"[^a-z ]", "", name.lower())
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
    """Cheap relevance score used to order candidates."""
    return 3 * len(STRONG_TERMS.findall(p["title"])) + len(STRONG_TERMS.findall(p["abstract"]))


def guess_section(p):
    text = p["title"] + " " + p["abstract"]
    for s, rx in SECTION_RULES:
        if re.search(rx, p["title"], re.I):
            return s
    for s, rx in SECTION_RULES:
        if len(re.findall(rx, text, re.I)) >= 2:
            return s
    return "ml"


AWARD_RX = re.compile(
    r"\b((best|outstanding|distinguished)( student| short| theory| application| demo)? paper"
    r"|paper award|test of time award|honou?rable mention)\b", re.I)


def award_from(comment):
    """'Accepted at IJCAI 2026; Best Paper Award' -> 'Best Paper Award' (the matching clause)."""
    for part in re.split(r"[;.\n]|\s-\s", comment or ""):
        if AWARD_RX.search(part):
            return squash(part).strip(" ,()")[:120]
    return ""


def opening(abstract, words=60):
    w = abstract.split()
    return " ".join(w[:words]) + (" ..." if len(w) > words else "")


def days_old(c, now):
    try:
        return max(0, (now.date() - datetime.fromisoformat(c["date"][:10]).date()).days)
    except ValueError:
        return 9999


# --------------------------------------------------------------------------- sources

def fetch_arxiv(since):
    cats = " OR ".join(f"cat:{c}" for c in ARXIV_CATS)
    terms = " OR ".join(f'abs:"{t}"' if (" " in t or "-" in t) else f"abs:{t}" for t in ARXIV_TERMS)
    query = f"({cats}) AND ({terms})"
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    out, start, page = [], 0, 200
    while start < 5000:
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
                "award": award_from(e.findtext("arxiv:comment", "", ns)),
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
    for _ in range(8):  # up to 8000 results
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
                "date": p.get("publicationDate") or f"{p.get('year')}-01-01",
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


def jats_to_text(s):
    s = re.sub(r"<jats:title>.*?</jats:title>", " ", s or "", flags=re.S)
    return squash(re.sub(r"<[^>]+>", " ", s))


def fetch_crossref(since):
    out = []
    for prefix in CROSSREF_PREFIXES:
        cursor, n = "*", 0
        while True:
            url = f"https://api.crossref.org/prefixes/{prefix}/works?" + urllib.parse.urlencode({
                "filter": f"from-pub-date:{since.date()}",
                "select": "DOI,title,abstract,author,published,container-title",
                "rows": 1000, "cursor": cursor})
            msg = json.loads(http(url))["message"]
            items = msg.get("items", [])
            for it in items:
                abstract = jats_to_text(it.get("abstract"))
                title = squash((it.get("title") or [""])[0])
                if not abstract or not title:
                    continue
                parts = [x for x in (it.get("published") or {}).get("date-parts", [[None]])[0] if x]
                if not parts:
                    continue
                date = "-".join([str(parts[0])] + [f"{x:02d}" for x in parts[1:]])
                date += "-01-01"[len(date) - 4:] if len(date) < 10 else ""
                container = (it.get("container-title") or [""])[0]
                out.append({
                    "id": f"doi:{it['DOI'].lower()}", "arxiv": None, "title": title,
                    "abstract": abstract,
                    "authors": [squash(f"{a.get('given', '')} {a.get('family', '')}")
                                for a in it.get("author") or []],
                    "url": f"https://doi.org/{it['DOI']}", "date": date[:10],
                    "venue": short_venue(container), "venue_year": parts[0], "dblp": None,
                })
            n += len(items)
            cursor = msg.get("next-cursor")
            if not items or not cursor or n >= msg.get("total-results", 0):
                break
            time.sleep(1)
    log(f"Crossref (IJCAI, AAAI, AIES, ...): {len(out)} papers since {since.date()}")
    return out


def arxiv_awards(arxiv_ids):
    """arXiv id -> award text found in the paper's arXiv comment (only ids with an award)."""
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    found, ids = {}, list(dict.fromkeys(arxiv_ids))
    for i in range(0, len(ids), 100):
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(ids[i:i + 100]), "max_results": 100})
        for e in ET.fromstring(http(url)).findall("a:entry", ns):
            aid = re.sub(r"v\d+$", "", e.findtext("a:id", "", ns).rsplit("/abs/", 1)[-1])
            award = award_from(e.findtext("arxiv:comment", "", ns))
            if award:
                found[aid] = award
        time.sleep(3)
    return found


def apply_manual_awards(papers):
    for entry in load_json(AWARDS_FILE, []):
        m = (entry.get("match") or "").strip().lower()
        if not m or not entry.get("award"):
            continue
        for p in papers:
            if m in (p["id"].lower(), (p.get("arxiv") or "").lower(), "doi:" + m) or title_key(m) == title_key(p["title"]):
                p["award"] = entry["award"]


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


# --------------------------------------------------------------------------- Claude

def describe_sections():
    lines = []
    for s, slabel in SECTIONS.items():
        lines.append(f'Section "{s}" - {slabel}:')
        lines += [f'  "{k}" ({label}): {desc}' for k, (sec, label, desc) in CATEGORIES.items() if sec == s]
    return "\n".join(lines)


TRIAGE_SYSTEM = f"""You are the editor of a weekly newspaper-style digest of new research, read by \
researchers. It has three sections, each split into categories:
{describe_sections()}

You see each paper's title and the opening of its abstract. For each paper decide:
- relevant: true only if one of the three section topics is a central topic of the paper. \
False if "fair" is incidental (e.g. "a fair comparison"), if "bias" means statistical/inductive bias, \
if "alignment" means something unrelated to aligning AI with human preferences or values \
(e.g. image-text alignment, sequence alignment), or if the paper is about something else.
- section: "ml", "sc" or "align" - the best fit (for irrelevant papers, the closest).
- score: 1-5, how interesting/significant it looks to a researcher in that section."""

WRITE_SYSTEM = f"""You are the editor of a weekly newspaper-style digest of new research, read by \
researchers. It has three sections, each split into categories:
{describe_sections()}

Each paper below has already been assigned to a section. For each paper write:
- category: the best-fitting category key within the paper's section.
- headline: newspaper headline, at most 12 words, plain English, accurate, no hype, no question \
marks, no clickbait. Say what was found or built, not "Researchers study...".
- blurb: one or two sentences, at most 40 words, explaining what the paper does and why it matters, \
for a researcher skimming the page. Stay strictly faithful to the abstract; do not invent results.
- score: 1-5, how interesting/significant it looks to a researcher in that section \
(5 = likely to be widely read)."""


def items_schema(props):
    return {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {
            "type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}}},
        "required": ["items"], "additionalProperties": False,
    }


TRIAGE_SCHEMA = items_schema({
    "id": {"type": "string"}, "relevant": {"type": "boolean"},
    "section": {"type": "string", "enum": list(SECTIONS)}, "score": {"type": "integer"}})
WRITE_SCHEMA = items_schema({
    "id": {"type": "string"}, "category": {"type": "string", "enum": list(CATEGORIES)},
    "headline": {"type": "string"}, "blurb": {"type": "string"}, "score": {"type": "integer"}})

USAGE = {"input": 0, "output": 0}


def ask_claude(system, schema, papers, render, chunk_size):
    """Send papers in chunks; return {id: item}. Chunks that fail are simply missing."""
    import anthropic

    client = anthropic.Anthropic()
    use_fallbacks = MODEL.startswith(("claude-opus-5", "claude-fable-5"))
    results = {}
    for i in range(0, len(papers), chunk_size):
        chunk = papers[i:i + chunk_size]
        listing = "\n\n".join(render(p) for p in chunk)
        kwargs = dict(
            model=MODEL, max_tokens=16000, system=system,
            messages=[{"role": "user", "content": f"Papers:\n\n{listing}\n\nReturn one item per paper id."}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
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
            log(f"  Claude API error ({e.status_code}): {e.message} - falling back for this chunk")
            continue
        except anthropic.APIConnectionError as e:
            log(f"  Claude API unreachable: {e} - falling back for this chunk")
            continue
        USAGE["input"] += resp.usage.input_tokens
        USAGE["output"] += resp.usage.output_tokens
        if resp.stop_reason in ("refusal", "max_tokens"):
            log(f"  chunk stopped with {resp.stop_reason} - falling back for this chunk")
            continue
        text = next(b.text for b in resp.content if b.type == "text")
        for item in json.loads(text)["items"]:
            results[item["id"]] = item
    return results


def triage(papers, use_claude):
    judged = {}
    if papers and use_claude:
        judged = ask_claude(
            TRIAGE_SYSTEM, TRIAGE_SCHEMA, papers, chunk_size=50,
            render=lambda p: f"<paper id=\"{p['id']}\">\nTitle: {p['title']}\n"
                             f"Abstract (opening): {opening(p['abstract'])}\n</paper>")
    out = {}
    for p in papers:
        j = judged.get(p["id"])
        if j:
            out[p["id"]] = {"r": j["relevant"], "s": j["section"], "q": max(1, min(5, j["score"]))}
        else:  # free fallback
            relevant = bool(PREFILTER.search(p["title"])) and strength(p) >= 3
            out[p["id"]] = {"r": relevant, "s": guess_section(p), "q": 2, "free": True}
    return out


def write_stories(papers, sections, use_claude):
    judged = {}
    if papers and use_claude:
        judged = ask_claude(
            WRITE_SYSTEM, WRITE_SCHEMA, papers, chunk_size=20,
            render=lambda p: f"<paper id=\"{p['id']}\" section=\"{sections[p['id']]}\">\n"
                             f"Title: {p['title']}\nAbstract: {p['abstract']}\n</paper>")
    out = {}
    for p in papers:
        sec = sections[p["id"]]
        j = judged.get(p["id"])
        if j:
            if CATEGORIES.get(j["category"], ("",))[0] != sec:  # keep text, fix category
                j = {**j, "category": SECTION_DEFAULT[sec]}
            out[p["id"]] = {**j, "ai": True}
            continue
        blurb = ""
        for s in re.split(r"(?<=[.!?])\s+", p["abstract"]):
            if len((blurb + " " + s).split()) > 45 and blurb:
                break
            blurb = (blurb + " " + s).strip()
        text = p["title"] + " " + p["abstract"]
        cat = next((k for k, rx in CATEGORY_RULES
                    if CATEGORIES[k][0] == sec and re.search(rx, text, re.I)), SECTION_DEFAULT[sec])
        out[p["id"]] = {"id": p["id"], "category": cat, "headline": p["title"], "blurb": blurb,
                        "score": 2, "ai": False}
    return out


# --------------------------------------------------------------------------- selection

def select(pool, verdicts, slots, now):
    """Peer-reviewed first (newest week first), then preprints; MIN_PER_SECTION per section."""
    def priority(c):
        return (0 if c.get("award") else 1, 0 if c["venue"] else 1, days_old(c, now) // 7,
                -verdicts[c["id"]]["q"], -strength(c))

    by_section = {s: sorted((c for c in pool if verdicts[c["id"]]["s"] == s), key=priority)
                  for s in SECTIONS}
    chosen, taken = [], set()
    # guaranteed minimum per section (round-robin, so a reduced budget is shared fairly)
    for k in range(MIN_PER_SECTION):
        for s in SECTIONS:
            if len(chosen) < slots and k < len(by_section[s]):
                chosen.append(by_section[s][k])
                taken.add(by_section[s][k]["id"])
    # remaining slots: best of the rest
    for c in sorted(pool, key=priority):
        if len(chosen) >= slots:
            break
        if c["id"] not in taken:
            chosen.append(c)
            taken.add(c["id"])
    return chosen


# --------------------------------------------------------------------------- main

def main():
    now = datetime.now(timezone.utc)
    use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    week = "%d-W%02d" % now.isocalendar()[:2]
    site = load_json(PAPERS_FILE, {"updated": None, "edition": 0, "papers": []})
    seen = set(load_json(SEEN_FILE, []))
    verdicts = load_json(TRIAGE_FILE, {})
    budget = load_json(BUDGET_FILE, {})
    used = budget.get("used", 0) if budget.get("week") == week else 0
    slots = max(0, MAX_PER_WEEK - used)
    stories = {p["id"]: p for p in site["papers"]}
    known_titles = {title_key(p["title"]) for p in site["papers"]}
    known_arxiv = {p.get("arxiv") for p in site["papers"] if p.get("arxiv")}
    log(f"Week {week}: {used} of {MAX_PER_WEEK} write-ups already used, {slots} slots left")
    if not slots:
        log("Weekly cap reached - nothing to do until next week.")
        return

    # 1. gather candidates (peer-reviewed sources first, so their version wins duplicates)
    peer_since = now - timedelta(weeks=PEER_LOOKBACK_WEEKS)
    raw = []
    for name, fetch, since in [("Crossref", fetch_crossref, peer_since),
                               ("Semantic Scholar", fetch_s2_venue_papers, peer_since),
                               ("arXiv", fetch_arxiv, now - timedelta(days=ARXIV_LOOKBACK_DAYS))]:
        try:
            raw += fetch(since)
        except Exception as e:  # never let one source kill the run
            log(f"{name} failed: {e}")

    candidates, ids, titles = [], set(), set()
    for c in raw:
        tk = title_key(c["title"])
        if (c["id"] in seen or c["id"] in ids or tk in titles or tk in known_titles
                or (c.get("arxiv") and c["arxiv"] in known_arxiv)):
            continue
        ids.add(c["id"])
        titles.add(tk)
        if not passes_prefilter(c):
            seen.add(c["id"])
            continue
        candidates.append(c)

    # 2. venue lookup: preprints that may since have been published + older preprint stories
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
    n_peer = sum(1 for c in candidates if c["venue"])
    log(f"{len(candidates)} candidates after prefilter ({n_peer} peer-reviewed, "
        f"{len(candidates) - n_peer} preprints)")

    # 3. triage untriaged candidates: round-robin over guessed sections, peer-reviewed and
    #    newest first, so every section gets triaged even with a small MAX_TRIAGE
    queues = {s: [] for s in SECTIONS}
    for c in sorted((c for c in candidates if c["id"] not in verdicts),
                    key=lambda c: (0 if c["venue"] else 1, days_old(c, now), -strength(c))):
        queues[guess_section(c)].append(c)
    order = []
    while len(order) < MAX_TRIAGE and any(queues.values()):
        for s in SECTIONS:
            if queues[s] and len(order) < MAX_TRIAGE:
                order.append(queues[s].pop(0))
    new_verdicts = triage(order, use_claude)
    verdicts.update(new_verdicts)
    log(f"Triage: {len(order)} papers, {sum(v['r'] for v in new_verdicts.values())} relevant")

    for c in candidates:  # irrelevant papers are never looked at again
        v = verdicts.get(c["id"])
        if v and not v["r"]:
            seen.add(c["id"])
    pool = [c for c in candidates if verdicts.get(c["id"], {}).get("r")]
    log("Relevant pool: " + ", ".join(
        f"{SECTIONS[s]} {sum(1 for c in pool if verdicts[c['id']]['s'] == s)}" for s in SECTIONS))

    # award notes in arXiv comments: for published candidates with an arXiv version (arXiv
    # candidates already carry theirs) and for this year's stories (awards come later)
    apply_manual_awards(candidates)
    recent = str((now - timedelta(days=365)).date())
    ids = [c["arxiv"] for c in pool if c.get("arxiv") and c["venue"] and not c.get("award")]
    ids += [p["arxiv"] for p in stories.values() if p.get("arxiv") and p["added"] >= recent]
    try:
        awards = arxiv_awards(ids) if ids else {}
    except Exception as e:
        log(f"arXiv award check failed: {e}")
        awards = {}
    for p in pool + list(stories.values()):
        if awards.get(p.get("arxiv")):
            p["award"] = awards[p["arxiv"]]
    log(f"Awards: {len(awards)} found in arXiv comments")

    # 4. choose and write this week's stories
    batch = select(pool, verdicts, slots, now)
    log("This edition: " + ", ".join(
        f"{SECTIONS[s]} {sum(1 for c in batch if verdicts[c['id']]['s'] == s)}" for s in SECTIONS)
        + f" ({sum(1 for c in batch if c['venue'])} peer-reviewed)")
    written = write_stories(batch, {c["id"]: verdicts[c["id"]]["s"] for c in batch}, use_claude)
    edition = site.get("edition", 0) + (1 if batch else 0)
    for p in batch:
        j = written[p["id"]]
        seen.add(p["id"])
        cat = j["category"]
        stories[p["id"]] = {
            "id": p["id"], "arxiv": p.get("arxiv"), "title": p["title"],
            "headline": j["headline"] or p["title"], "blurb": j["blurb"],
            "section": CATEGORIES[cat][0], "category": cat,
            "score": max(1, min(5, j["score"])), "authors": authors_str(p["authors"]),
            "url": p["url"], "date": p["date"], "added": str(now.date()), "edition": edition,
            "venue": p.get("venue") or "", "venue_year": p.get("venue_year"), "dblp": p.get("dblp"),
            "ai": j["ai"], "award": p.get("award") or "",
        }
    if use_claude:
        log(f"Claude ({MODEL}): {USAGE['input']} input / {USAGE['output']} output tokens")

    apply_manual_awards(list(stories.values()))

    # 5. write
    keep_cutoff = str((now - timedelta(days=KEEP_DAYS)).date())
    papers = sorted((p for p in stories.values() if p["added"] >= keep_cutoff),
                    key=lambda p: (p["added"], p["score"], p["date"]), reverse=True)
    save_json(PAPERS_FILE, {
        "updated": now.isoformat(timespec="minutes"), "edition": edition,
        "categories": {k: {"section": s, "label": label} for k, (s, label, _) in CATEGORIES.items()},
        "sections": SECTIONS, "papers": papers})
    save_json(SEEN_FILE, sorted(seen))
    save_json(TRIAGE_FILE, {k: v for k, v in verdicts.items() if k not in seen})
    save_json(BUDGET_FILE, {"week": week, "used": used + len(batch)})
    log(f"Edition {edition}: site has {len(papers)} stories ({len(batch)} added this run)")


if __name__ == "__main__":
    main()
