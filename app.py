import os, re, json, time, hashlib, random, io
from typing import List, Dict, Tuple
import requests
import pandas as pd
import numpy as np
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz
import nltk
from nltk.corpus import stopwords

# Gradio + OpenAI
import gradio as gr
from openai import OpenAI

# Community detection
import community as community_louvain  # package name: python-louvain

# PDF reading (for page-count heuristic)
from PyPDF2 import PdfReader

# KeyBERT + SBERT
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# -------------------------
# Config / Globals
# -------------------------
OPENALEX_BASE = "https://api.openalex.org/works"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "you@example.com")

DEFAULT_TOPIC = "retrieval-augmented generation for graphs"
DEFAULT_N_PAPERS = 25
DEFAULT_TIME_BUDGET_MIN = 360
DEFAULT_TARGET_COVERAGE = 0.8

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# Ensure NLTK data exists (image already contains it, but safe at runtime too)
try:
    _ = stopwords.words("english")
except Exception:
    nltk.download("punkt"); nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))

# SBERT + KeyBERT
_sbert = SentenceTransformer("all-MiniLM-L6-v2")
_kw_model = KeyBERT(_sbert)
USE_SBERT = True

# OpenAI client (will error at QA tab if key missing)
_client = None
if os.getenv("OPENAI_API_KEY"):
    _client = OpenAI()

# Shared planner result for QA tab
res = None

# -------------------------
# Utils
# -------------------------
def sha256_str(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()[:12]

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _get_nested(d, path, default=""):
    cur = d or {}
    for k in path:
        if not isinstance(cur, dict): return default
        cur = cur.get(k)
        if cur is None: return default
    return cur

def reconstruct_openalex_abstract(abstract_inverted_index: dict) -> str:
    if not abstract_inverted_index:
        return ""
    try:
        max_pos = max(pos for positions in abstract_inverted_index.values() for pos in positions)
        words = [""] * (max_pos + 1)
        for word, positions in abstract_inverted_index.items():
            for p in positions:
                if 0 <= p < len(words):
                    words[p] = word
        return re.sub(r"\s+", " ", " ".join(words)).strip()
    except Exception:
        return ""

# -------------------------
# Fetch corpus
# -------------------------
def fetch_openalex_works(topic: str, n: int = 25, prefer_recent: bool = True) -> List[dict]:
    params = {"search": topic, "per_page": min(n, 50), "page": 1,
              "sort": "publication_year:desc" if prefer_recent else "relevance_score:desc"}
    works = []
    while len(works) < n and params["page"] < 6:
        r = requests.get(OPENALEX_BASE, params=params, timeout=30)
        r.raise_for_status()
        data = r.json() or {}
        batch = data.get("results", []) or []
        works.extend(batch)
        if len(batch) < params["per_page"]:
            break
        params["page"] += 1
        time.sleep(0.2)

    out = []
    for w in works[:n]:
        wid = (w.get("id") or "").strip()
        title = normalize_text(w.get("title") or "")
        year = w.get("publication_year")
        doi = (w.get("doi") or "").replace("https://doi.org/", "").strip()

        url = ""
        best_oa = w.get("best_oa_location") or {}
        url = best_oa.get("url_for_pdf") or best_oa.get("url") or ""
        if not url:
            url = _get_nested(w, ["primary_location", "source", "homepage_url"], "")
        if not url:
            url = _get_nested(w, ["primary_location", "landing_page_url"], "")
        if not url:
            url = _get_nested(w, ["host_venue", "url"], "")
        if not url and wid:
            url = wid

        abstract = reconstruct_openalex_abstract(w.get("abstract_inverted_index") or {})
        referenced = w.get("referenced_works") or []
        cited_by_count = int(w.get("cited_by_count") or 0)

        out.append({
            "openalex_id": wid,
            "title": title,
            "year": year if (isinstance(year, int) or year is None) else None,
            "doi": doi,
            "oa_url": url,
            "abstract": abstract,
            "referenced_works": referenced,
            "cited_by_count": cited_by_count,
        })
    return out

# -------------------------
# Keyphrases
# -------------------------
def extract_keyphrases(text: str, max_phrases: int = 15) -> List[Tuple[str, float]]:
    if not text: return []
    t = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text.lower())
    tokens = [w for w in t.split() if w and w not in STOPWORDS and not w.isdigit()]
    if not tokens: return []
    chunks = re.split(r"\b(?:{})\b".format("|".join(map(re.escape, STOPWORDS))), t)
    phrases = []
    for ch in chunks:
        ch = ch.strip()
        if not ch: continue
        words = [w for w in ch.split() if w and w not in STOPWORDS]
        if 1 <= len(words) <= 5:
            phrases.append(" ".join(words))
    freq, degree = {}, {}
    for ph in phrases:
        ws = ph.split(); uniq = set(ws)
        for w in uniq:
            freq[w] = freq.get(w, 0) + 1
            degree[w] = degree.get(w, 0) + (len(ws) - 1)
    for w in list(freq.keys()):
        degree[w] += freq[w]
    scores = []
    for ph in phrases:
        ws = ph.split()
        score = sum(degree[w] / freq[w] for w in ws if w in freq)
        scores.append((ph, float(score)))
    best = {}
    for ph, sc in sorted(scores, key=lambda x: x[1], reverse=True):
        if any(fuzz.token_set_ratio(ph, k) >= 90 for k in best.keys()):
            continue
        best[ph] = sc
        if len(best) >= max_phrases: break
    return list(best.items())

def extract_keyphrases_sbert(text: str, top_k: int = 20):
    if not text: return []
    pairs = _kw_model.extract_keywords(
        text, top_n=top_k, keyphrase_ngram_range=(1,3),
        stop_words="english", use_mmr=True, diversity=0.6
    )
    return [(p, float(s)) for p, s in pairs]

# -------------------------
# Build corpus & graph
# -------------------------
def build_corpus(works: List[dict]):
    rows = []
    for w in works:
        text = normalize_text(f"{w['title']}. {w['abstract']}")
        words = len(text.split())
        read_min = 20 + 0.5 * (words / 100.0)
        rows.append({**w, "text": text, "word_count": words, "est_read_min": round(read_min, 1)})
    df = pd.DataFrame(rows)

    vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1,2), stop_words="english")
    tfidf = vectorizer.fit_transform(df["text"].fillna(""))
    vocab = np.array(vectorizer.get_feature_names_out())
    importance = np.asarray(tfidf.mean(axis=0)).ravel()
    concept_scores = (pd.DataFrame({"concept": vocab, "importance": importance})
                        .sort_values("importance", ascending=False).head(400))

    per_paper_phrases = []
    for _, row in df.iterrows():
        local = extract_keyphrases_sbert(row["text"], top_k=20) if USE_SBERT \
                else extract_keyphrases(row["text"], max_phrases=20)
        per_paper_phrases.append(local[:12])
    df["keyphrases"] = per_paper_phrases

    openalex_ids = set(df["openalex_id"].tolist())
    in_graph_refs = []
    for refs in df["referenced_works"]:
        in_graph_refs.append([r for r in refs if r in openalex_ids])
    df["in_graph_refs"] = in_graph_refs

    return df, concept_scores

def build_graph(df: pd.DataFrame, concept_scores: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    concept_imp = {r.concept: float(r.importance) for r in concept_scores.itertuples()}
    for c, imp in concept_imp.items():
        G.add_node(("concept", c), kind="concept", label=c, importance=imp)

    for r in df.itertuples():
        pid = ("paper", r.openalex_id)
        G.add_node(pid, kind="paper", title=r.title, year=int(r.year) if r.year else None,
                   doi=r.doi, oa_url=r.oa_url, read_min=float(r.est_read_min),
                   cited_by=int(r.cited_by_count))
        concepts = set()
        for ph, _ in r.keyphrases:
            for tok in ph.split():
                if tok in concept_imp:
                    concepts.add(tok)
        for c in list(concepts):
            G.add_edge(pid, ("concept", c), kind="covers", weight=1.0 + 2.0 * concept_imp[c])

    idset = set(df["openalex_id"].tolist())
    ref_map = {r.openalex_id: set(r.in_graph_refs) for r in df.itertuples()}
    for src, refs in ref_map.items():
        for tgt in refs:
            if tgt in idset:
                G.add_edge(("paper", src), ("paper", tgt), kind="cites", weight=1.0)
    return G

def graph_snapshot_hash(G: nx.DiGraph) -> str:
    data = nx.readwrite.json_graph.node_link_data(G)
    return sha256_str(json.dumps(data, sort_keys=True))

def collect_concepts_for_paper(G: nx.DiGraph, paper_node) -> set:
    out = set()
    for u, v in G.out_edges(paper_node):
        if G[u][v].get("kind") == "covers" and isinstance(v, tuple) and v[0] == "concept":
            out.add(v[1])
    return out

def concept_importance(G: nx.DiGraph) -> Dict[str, float]:
    return {n[1]: G.nodes[n].get("importance", 0.0)
            for n in G.nodes if G.nodes[n]["kind"] == "concept"}

def greedy_set_cover_multi(
    G: nx.DiGraph,
    time_budget_min: float,
    target_cov: float = 0.8,
    known_concepts: set | None = None,
    w_recency: float = 0.25,
    w_citations: float = 0.20,
    redundancy_penalty: float = 0.5
):
    papers = [n for n in G.nodes if G.nodes[n]["kind"] == "paper"]
    imp = concept_importance(G)
    total_imp = sum(imp.values()) or 1e-9

    def concept_set(p): return {v[1] for u, v in G.out_edges(p)
                                if G[u][v].get("kind")=="covers" and v[0]=="concept"}
    paper_concepts = {p: concept_set(p) for p in papers}
    years = [G.nodes[p].get("year") or 0 for p in papers]
    cites = [G.nodes[p].get("cited_by", 0) for p in papers]
    y_max = max(years) if years else 0
    y_min = min(years) if years else 0
    c_max = max(cites) if cites else 0

    def recency_bonus(p):
        y = G.nodes[p].get("year") or y_min
        if y_max == y_min: return 1.0
        return 1.0 + w_recency * ((y - y_min) / (y_max - y_min))

    def citation_bonus(p):
        c = G.nodes[p].get("cited_by", 0)
        if c_max == 0: return 1.0
        return 1.0 + w_citations * (c / c_max)

    selected, used_time = [], 0.0
    covered = set(known_concepts or set())

    def coverage_score(concepts: set) -> float:
        return sum(imp.get(c,0.0) for c in concepts) / total_imp

    def marginal_gain(p):
        new = paper_concepts[p] - covered
        return sum(imp.get(c,0.0) for c in new)

    while True:
        remaining = [p for p in papers if p not in selected]
        if not remaining: break
        best, best_val = None, -1.0
        for p in remaining:
            rt = float(G.nodes[p].get("read_min", 30.0))
            gain = marginal_gain(p)
            if selected:
                union_sel = set().union(*[paper_concepts[s] for s in selected])
                overlap = len(paper_concepts[p] & union_sel) / len(paper_concepts[p]) if paper_concepts[p] else 0.0
            else:
                overlap = 0.0
            anti_dup = 1.0 - redundancy_penalty * overlap
            score = (gain / (rt + 1e-9)) * recency_bonus(p) * citation_bonus(p) * max(anti_dup, 0.2)
            if score > best_val:
                best, best_val = p, score
        if best is None or best_val <= 0: break
        selected.append(best)
        used_time += float(G.nodes[best].get("read_min", 30.0))
        covered |= paper_concepts[best]
        if coverage_score(covered) >= target_cov or used_time > time_budget_min:
            break

    # Irredundancy pass
    changed = True
    while changed:
        changed = False
        for p in list(selected):
            with_p = set(known_concepts or set())
            for q in selected:
                if q != p: with_p |= paper_concepts[q]
            if coverage_score(with_p) >= coverage_score(covered):
                selected.remove(p)
                covered = with_p
                changed = True
                break

    cov = coverage_score(covered)
    return selected, used_time, cov, {c: imp.get(c,0.0) for c in covered}

def verify_metadata(df: pd.DataFrame) -> float:
    checks = 0; ok = 0
    for r in df.itertuples():
        checks += 3
        if r.title: ok += 1
        if r.openalex_id: ok += 1
        if r.year is None or isinstance(r.year, (int, np.integer)): ok += 1
    return ok / max(checks,1)

def check_links(df: pd.DataFrame, try_unpaywall: bool = False) -> float:
    ok = 0; total = 0
    for r in df.itertuples():
        total += 1
        url = r.oa_url
        if url and isinstance(url, str):
            try:
                h = requests.head(url, allow_redirects=True, timeout=10)
                if 200 <= h.status_code < 400:
                    ok += 1
            except Exception:
                pass
            time.sleep(0.05)
    return ok / max(total,1)

def edge_support_ratio(G: nx.DiGraph, df: pd.DataFrame) -> float:
    text_map = {("paper", r.openalex_id): r.text.lower() for r in df.itertuples()}
    covered_edges = 0; supported = 0
    for u, v, data in G.edges(data=True):
        if data.get("kind") != "covers": continue
        covered_edges += 1
        concept = v[1].lower(); text = text_map.get(u, "")
        if concept in text or fuzz.partial_ratio(concept, text) >= 90:
            supported += 1
    return supported / max(covered_edges,1)

def necessity_check(G: nx.DiGraph, selected: List[Tuple[str,str]], target_cov: float) -> bool:
    imp = concept_importance(G); total = sum(imp.values()) or 1e-9
    def cov_of(sel):
        covset = set()
        for p in sel: covset |= collect_concepts_for_paper(G, p)
        return sum(imp.get(c,0.0) for c in covset)/total
    full_cov = cov_of(selected)
    if full_cov < target_cov: return False
    for i in range(len(selected)):
        subset = selected[:i] + selected[i+1:]
        if cov_of(subset) >= target_cov: return False
    return True

def ais_score_explanations(selected, G, df):
    text_map = {("paper", r.openalex_id): r.text for r in df.itertuples()}
    claims = []; ok = 0; total = 0
    for p in selected:
        text = text_map.get(p, ""); title = G.nodes[p].get("title","")
        concepts = list(collect_concepts_for_paper(G, p))[:3]
        for c in concepts:
            total += 1
            idx = text.lower().find(c.lower())
            if idx >= 0:
                span = [idx, idx+len(c)]; ok += 1
            else:
                span = [0, min(80, len(text))]
            claims.append({"paper_id": p[1], "title": title, "concept": c,
                           "evidence_span": span, "text_snippet": text[span[0]:span[1]]})
    score = ok / max(total,1)
    return score, claims

def truth_report_card(df, G, selected, target_cov, used_time, cov, claims_score):
    meta_ok = verify_metadata(df)
    links_ok = check_links(df.copy(), try_unpaywall=False)
    supports = edge_support_ratio(G, df)
    necessity_ok = necessity_check(G, selected, target_cov)
    return {
        "source_integrity_metadata": round(meta_ok,3),
        "link_validity_ratio": round(links_ok,3),
        "graph_edge_support_ratio": round(supports,3),
        "plan_coverage": round(cov,3),
        "plan_time_used_min": round(used_time,1),
        "plan_irredundant": bool(necessity_ok),
        "explanation_groundedness": round(claims_score,3),
        "graph_hash": graph_snapshot_hash(G),
        "seed": SEED
    }

def build_plan_for_topic(
    topic=DEFAULT_TOPIC,
    n_papers=DEFAULT_N_PAPERS,
    time_budget_min=DEFAULT_TIME_BUDGET_MIN,
    target_coverage=DEFAULT_TARGET_COVERAGE,
    prefer_recent=True,
    prior_notes="",
    w_recency=0.25,
    w_citations=0.20,
    redundancy_penalty=0.5
):
    works = fetch_openalex_works(topic, n=n_papers, prefer_recent=prefer_recent)
    if not works: raise RuntimeError("No works from OpenAlex.")
    df, concept_scores = build_corpus(works)
    G = build_graph(df, concept_scores)

    concept_vocab = set(concept_scores["concept"].tolist())
    known = set()
    if prior_notes:
        # quick overlap with concept vocab using SBERT extractor
        kws = extract_keyphrases_sbert(prior_notes, top_k=15) if USE_SBERT \
              else extract_keyphrases(prior_notes, max_phrases=15)
        toks = {t for p,_ in kws for t in p.lower().split()}
        known = {t for t in toks if t in concept_vocab}

    selected, used_time, cov, covered_weights = greedy_set_cover_multi(
        G, time_budget_min, target_coverage, known_concepts=known,
        w_recency=w_recency, w_citations=w_citations, redundancy_penalty=redundancy_penalty
    )

    claim_score, claims = ais_score_explanations(selected, G, df)
    card = truth_report_card(df, G, selected, target_coverage, used_time, cov, claim_score)

    plan_rows = []
    for p in selected:
        node = G.nodes[p]
        plan_rows.append({"title": node.get("title",""), "year": node.get("year"),
                          "est_read_min": round(node.get("read_min", 30.0),1),
                          "oa_url": node.get("oa_url") or ""})
    plan_df = pd.DataFrame(plan_rows)
    plan_df["order"] = range(1, len(plan_df)+1)
    plan_df = plan_df[["order","title","year","est_read_min","oa_url"]]

    top_concepts = sorted(covered_weights.items(), key=lambda x: x[1], reverse=True)[:15]
    top_concepts_df = pd.DataFrame([{"concept":k, "weight":round(v,5)} for k,v in top_concepts])

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/claims.json","w") as f: json.dump(claims, f, indent=2)
    nx.write_gexf(G, "artifacts/readpath_graph.gexf")
    plan_df.to_csv("artifacts/reading_plan.csv", index=False)
    top_concepts_df.to_csv("artifacts/covered_concepts.csv", index=False)
    with open("artifacts/report_card.json","w") as f: json.dump(card, f, indent=2)

    return {"df": df, "concepts": concept_scores, "G": G, "selected": selected,
            "plan_df": plan_df, "covered_df": top_concepts_df, "claims": claims,
            "report_card": card}

# -------------------------
# Graph HTML (vis-network)
# -------------------------
def _simple_visnetwork_html(G, selected=None, height_px=700):
    selected = set(selected or [])
    def nid(n): return f"{n[0]}::{n[1]}"
    nodes, edges = [], []
    # optional communities for color
    P = nx.Graph()
    for n,d in G.nodes(data=True):
        if d.get("kind")=="paper": P.add_node(n)
    for u,v,data in G.edges(data=True):
        if data.get("kind")=="cites" and G.nodes[u]["kind"]=="paper" and G.nodes[v]["kind"]=="paper":
            P.add_edge(u,v)
    comm = community_louvain.best_partition(P) if P.number_of_edges() > 0 else {}
    palette = ["#4c78a8","#f58518","#e45756","#72b7b2","#54a24b","#b279a2","#ff9da6","#9d755d","#bab0ab"]

    for n, data in G.nodes(data=True):
        if data["kind"] == "paper":
            label = (data.get("title","")[:60] + ("…" if len(data.get("title",""))>60 else ""))
            base = palette[comm.get(n,0) % len(palette)]
            color = "#f58518" if n in selected else base
            nodes.append({"id": nid(n), "label": label,
                          "title": f"{data.get('title','')} ({data.get('year')})",
                          "color": color, "shape":"dot", "size":12})
        else:
            imp = float(data.get("importance", 0.0)); size = 5 + 20*imp
            nodes.append({"id": nid(n), "label": data.get("label",""),
                          "title": data.get("label",""),
                          "color":"#54a24b", "shape":"dot", "size":size})
    for u,v,ed in G.edges(data=True):
        if ed.get("kind")=="covers":
            edges.append({"from": nid(u), "to": nid(v), "color":"#54a24b"})
        elif ed.get("kind")=="cites":
            edges.append({"from": nid(u), "to": nid(v), "color":"#b279a2","arrows":"to"})
    options = {"physics":{"solver":"forceAtlas2Based","stabilization":{"iterations":200}},
               "interaction":{"hover":True},"edges":{"smooth":False}}
    return f"""
<div id="readpath-graph" style="width:100%; height:{int(height_px)}px; border:none;"></div>
<link rel="stylesheet" href="https://unpkg.com/vis-network/styles/vis-network.min.css">
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<script>
  (function(){{
    const nodes = new vis.DataSet({json.dumps(nodes)});
    const edges = new vis.DataSet({json.dumps(edges)});
    const options = {json.dumps(options)};
    new vis.Network(document.getElementById('readpath-graph'), {{nodes, edges}}, options);
  }})();
</script>
"""

def _iframe_from_html_snippet(snippet: str, h: int = 700) -> str:
    import base64
    b64 = base64.b64encode(snippet.encode("utf-8")).decode("ascii")
    return f'<iframe src="data:text/html;base64,{b64}" style="width:100%;height:{int(h)}px;border:none;"></iframe>'

def _render_graph(G, selected, height_px=700):
    return _iframe_from_html_snippet(_simple_visnetwork_html(G, selected, height_px), h=height_px)

# -------------------------
# PDF page-count heuristic
# -------------------------
def try_pdf_pages(url: str, timeout=12):
    if not url or (not url.lower().endswith(".pdf")): return None
    try:
        r = requests.get(url, timeout=timeout); r.raise_for_status()
        return len(PdfReader(io.BytesIO(r.content)).pages)
    except Exception:
        return None

def refine_read_time_with_pdf(G: nx.DiGraph, selected: list, top_k: int = 3):
    for p in selected[:top_k]:
        url = G.nodes[p].get("oa_url") or ""
        pages = try_pdf_pages(url)
        if pages:
            G.nodes[p]["read_min"] = round(pages * 2.5, 1)

# -------------------------
# QA (retrieval + LLM)
# -------------------------
OPENAI_MODEL = "gpt-4o-mini"
MAX_CHARS_PER_CHUNK = 1200
TOP_K_DEFAULT = 5
USE_SELECTED_ONLY_DEFAULT = True

def _doc_text(row: Dict) -> str:
    t = (row.get("title","") + ". " + (row.get("abstract","") or "")).strip()
    if (not t) and isinstance(row.get("text",""), str): t = row["text"]
    return t

def _registry_lines(docs: List[Dict]) -> str:
    out = []
    for i, d in enumerate(docs, 1):
        yr = f" ({d['year']})" if d.get("year") else ""
        out.append(f"[R{i}] {d['title']}{yr} — {d['url'] or d['id']}")
    return "\n".join(out)

def _safe_graph_hash(r) -> str:
    try:
        h = r.get("report_card", {}).get("graph_hash")
        if h: return h
    except Exception:
        pass
    try:
        ids = sorted(r["df"]["openalex_id"].tolist())
        return hashlib.sha256(",".join(ids).encode()).hexdigest()[:12]
    except Exception:
        return "unknown"

_qa_docs = None
_qa_X = None
_qa_registry = ""
_qa_res_hash = None
_qa_scope_selected = None

def build_qa_index(res_obj, restrict_to_selected: bool):
    if res_obj is None or "df" not in res_obj:
        raise RuntimeError("`res` not found. Build a plan first.")
    df = res_obj["df"]
    sel_ids = {p[1] for p in res_obj["selected"]} if restrict_to_selected else None

    docs = []
    for r in df.to_dict(orient="records"):
        if sel_ids and r["openalex_id"] not in sel_ids:
            continue
        txt = _doc_text(r)
        if not txt.strip(): continue
        docs.append({"id": r["openalex_id"], "title": r["title"],
                     "year": r["year"], "url": r["oa_url"], "text": txt})
    if not docs:
        raise RuntimeError("No documents to index (reading plan may be empty).")

    X = _sbert.encode([d["text"] for d in docs],
                      convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    reg_text = _registry_lines(docs)
    os.makedirs("artifacts", exist_ok=True)
    np.savez_compressed("artifacts/qa_index_sbert.npz", X=X)
    with open("artifacts/qa_sources.json","w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    with open("artifacts/qa_registry.txt","w", encoding="utf-8") as f:
        f.write(reg_text)
    return docs, X, reg_text

def ensure_index_up_to_date(restrict_to_selected: bool):
    global _qa_docs, _qa_X, _qa_registry, _qa_res_hash, _qa_scope_selected, res
    if res is None: raise RuntimeError("Build a plan first.")
    current_hash = _safe_graph_hash(res)
    if (_qa_docs is None or current_hash != _qa_res_hash or restrict_to_selected != _qa_scope_selected):
        _qa_docs, _qa_X, _qa_registry = build_qa_index(res, restrict_to_selected)
        _qa_res_hash = current_hash
        _qa_scope_selected = restrict_to_selected

def _search(docs: List[Dict], X: np.ndarray, query: str, top_k: int = TOP_K_DEFAULT):
    q = _sbert.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
    sims = (X @ q).tolist()
    idx = np.argsort(sims)[::-1][:top_k]
    out = []
    for rank, i in enumerate(idx, 1):
        d = docs[i]
        out.append({"rank": rank, "score": float(sims[i]), "id": d["id"], "title": d["title"],
                    "year": d["year"], "url": d["url"], "snippet": d["text"][:MAX_CHARS_PER_CHUNK]})
    return out

SYS_INSTRUCTIONS = (
    "You are a careful research assistant. Use ONLY the provided materials.\n"
    "Allowed sources:\n"
    "1) CORPUS REGISTRY — list/enumerate papers.\n"
    "2) EVIDENCE SNIPPETS — support factual claims with inline citations [1], [2], etc.\n"
    "If a claim is not supported by snippets, reply exactly: 'I don't know.' Do not invent sources or URLs."
)

def _build_prompt(question: str, evidence: List[Dict], registry_text: str) -> str:
    lines = [f"QUESTION:\n{question}\n", "CORPUS REGISTRY (all allowed papers):", registry_text, "",
             "SOURCES FOR THIS ANSWER (top retrieved):"]
    for e in evidence:
        yr = f" ({e['year']})" if e.get("year") else ""
        lines.append(f"[{e['rank']}] {e['title']}{yr} — {e['url'] or e['id']}")
    lines.append("\nEVIDENCE SNIPPETS:")
    for e in evidence:
        lines.append(f"--- [{e['rank']}] {e['title']}")
        lines.append(e["snippet"])
    lines.append("\nINSTRUCTIONS:\n- For listing/identifying corpus papers, use the CORPUS REGISTRY."
                 "\n- For factual answers, rely on EVIDENCE SNIPPETS and include [#] citations."
                 "\n- If not supported, answer: I don't know.")
    return "\n".join(lines)

def qa_answer(question: str, docs: List[Dict], X: np.ndarray, registry_text: str, top_k: int = TOP_K_DEFAULT):
    if _client is None:
        return "Missing OPENAI_API_KEY on the server. Set it and retry.", []
    ev = _search(docs, X, question, top_k=top_k)
    prompt = _build_prompt(question, ev, registry_text)
    resp = _client.responses.create(
        model=OPENAI_MODEL,
        instructions=SYS_INSTRUCTIONS,
        input=prompt,
        max_output_tokens=600,
        temperature=0.2,
    )
    answer = resp.output_text
    refs = [{"num": e["rank"], "title": e["title"], "year": e["year"], "url": e["url"] or e["id"]} for e in ev]
    return answer, refs

def _is_meta_list_q(q: str) -> bool:
    q = (q or "").lower()
    keys = ["what are the articles", "what are the papers", "list the papers",
            "list the articles", "which papers do i have", "which articles do i have",
            "what are the articles that i have from the planner"]
    return any(k in q for k in keys)

def _render_registry_md(docs: List[Dict]) -> str:
    lines = ["Here are the articles in your current corpus:\n"]
    for i, d in enumerate(docs, 1):
        yr = f" ({d['year']})" if d.get("year") else ""
        url = d["url"] or d["id"]
        lines.append(f"{i}. [{d['title']}{yr}]({url})")
    return "\n".join(lines)

def _render_answer_md(answer: str, refs: List[Dict]) -> str:
    out = [answer.strip(), "\n\n**References:**"]
    for r in refs:
        yr = f" ({r['year']})" if r.get("year") else ""
        out.append(f"[{r['num']}] [{r['title']}{yr}]({r['url']})")
    return "\n".join(out)

def _has_citation(text: str) -> bool:
    return bool(re.search(r"\[\d+\]", text or ""))

# -------------------------
# Gradio UI (two tabs)
# -------------------------
def _safe_json(obj) -> str:
    def _conv(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        return str(x)
    try:
        return json.dumps(obj, indent=2, default=_conv)
    except Exception:
        return json.dumps({"error": "failed to serialize"}, indent=2)

def app_infer(topic, n_papers, time_budget, target_cov, prefer_recent, notes="", w_rec=0.25, w_cit=0.20, red_pen=0.50):
    global res
    try:
        result = build_plan_for_topic(
            topic=topic.strip() or DEFAULT_TOPIC,
            n_papers=int(n_papers),
            time_budget_min=float(time_budget),
            target_coverage=float(target_cov),
            prefer_recent=bool(prefer_recent),
            prior_notes=notes or "",
            w_recency=float(w_rec),
            w_citations=float(w_cit),
            redundancy_penalty=float(red_pen)
        )
        try: refine_read_time_with_pdf(result["G"], result["selected"], top_k=3)
        except Exception: pass

        res = result
        plan_md = result["plan_df"].to_markdown(index=False)
        covered_md = result["covered_df"].to_markdown(index=False)
        report_md = "### Truth Report Card\n```json\n" + _safe_json(result["report_card"]) + "\n```"
        graph_iframe = _render_graph(result["G"], result["selected"], height_px=700)
        combo_md = f"## Reading Plan\n{plan_md}\n\n## Top Covered Concepts\n{covered_md}\n\n{report_md}\n\nArtifacts saved in ./artifacts/"
        plan_path = "artifacts/reading_plan.csv" if os.path.exists("artifacts/reading_plan.csv") else None
        report_path = "artifacts/report_card.json" if os.path.exists("artifacts/report_card.json") else None
        return gr.update(value=combo_md), graph_iframe, plan_path, report_path
    except Exception as e:
        return gr.update(value=f"❌ Error while building plan: {e}"), "<p style='color:red'>Failed.</p>", None, None

def _qa_handler(question, top_k, restrict_to_selected):
    if res is None:
        return "Please build a plan first (Planner tab)."
    restrict = bool(restrict_to_selected)
    ensure_index_up_to_date(restrict)
    if _is_meta_list_q(question):
        return _render_registry_md(_qa_docs)
    answer, refs = qa_answer(question, _qa_docs, _qa_X, _qa_registry, top_k=int(top_k))
    if not _has_citation(answer):
        return "I don't know."
    return _render_answer_md(answer, refs)

with gr.Blocks(title="ReadPath — Coverage Planner + Corpus-Strict QA") as demo:
    gr.Markdown("## ReadPath — Coverage-first reading plan + ask-the-papers (source-grounded)")

    with gr.Tab("Planner"):
        with gr.Row():
            topic = gr.Textbox(label="Topic", value=DEFAULT_TOPIC, lines=2)
            notes = gr.Textbox(label="What I already know", lines=2, placeholder="e.g., basics of DP, GANs, CTGAN")
        with gr.Row():
            n_papers = gr.Slider(10, 80, value=DEFAULT_N_PAPERS, step=1, label="Candidate papers")
            time_budget = gr.Slider(30, 720, value=DEFAULT_TIME_BUDGET_MIN, step=10, label="Time budget (min)")
            target_cov = gr.Slider(0.3, 0.95, value=DEFAULT_TARGET_COVERAGE, step=0.05, label="Target coverage")
            prefer_recent = gr.Checkbox(value=True, label="Prefer recent")
        gr.Markdown("**Planner weights** (optional)")
        with gr.Row():
            w_rec = gr.Slider(0.0, 0.6, value=0.25, step=0.05, label="Recency bonus")
            w_cit = gr.Slider(0.0, 0.6, value=0.20, step=0.05, label="Citation bonus")
            red_pen = gr.Slider(0.0, 0.9, value=0.5, step=0.05, label="Redundancy penalty")

        run = gr.Button("Build Plan", variant="primary")
        out_md = gr.Markdown()
        out_graph = gr.HTML()
        dl_plan = gr.File(label="Download plan (CSV)")
        dl_report = gr.File(label="Download report (JSON)")
        run.click(app_infer,
                  inputs=[topic, n_papers, time_budget, target_cov, prefer_recent, notes, w_rec, w_cit, red_pen],
                  outputs=[out_md, out_graph, dl_plan, dl_report])

    with gr.Tab("Ask the Papers"):
        gr.Markdown("Answers come **only** from your current corpus (default: selected reading plan).")
        with gr.Row():
            qa_q = gr.Textbox(label="Your question", placeholder="e.g., What evaluation metrics are common?", lines=2)
        with gr.Row():
            qa_k = gr.Slider(1, 10, value=TOP_K_DEFAULT, step=1, label="Top-K sources")
            qa_sel = gr.Checkbox(value=USE_SELECTED_ONLY_DEFAULT, label="Use only selected (reading plan)")
        qa_btn = gr.Button("Ask", variant="primary")
        qa_out = gr.Markdown()
        qa_btn.click(_qa_handler, inputs=[qa_q, qa_k, qa_sel], outputs=[qa_out])

# Launch server (0.0.0.0 for Docker)
PORT = int(os.getenv("PORT", "7860"))
demo.queue().launch(server_name="0.0.0.0", server_port=PORT, show_api=False, share=False)
