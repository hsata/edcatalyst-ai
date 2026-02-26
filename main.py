from http import client
from random import seed
from weakref import ref
from fastapi import FastAPI
from pydantic import BaseModel, Field
import arxiv
from typing import List
import boto3
import os
import json


app = FastAPI(title="EdCatalyst AI")

from fastapi import HTTPException
import traceback

@app.get("/")
def read_root():
    return {"message": "EdCatalyst AI backend is running 🚀"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

class AnalyzeRequest(BaseModel):
    topic: str = Field(..., min_length=3)
    max_papers: int = Field(8, ge=1, le=15)

class Paper(BaseModel):
    title: str
    authors: List[str]
    published: str
    summary: str
    pdf_url: str
    arxiv_url: str
    relevance_reason: str

class Plan(BaseModel):
    refined_queries: List[str]
    screening_criteria: List[str]

class SelectedPaper(BaseModel):
    title: str
    arxiv_url: str
    why_selected: str

class ActionPlan(BaseModel):
    for_teachers: List[str]
    for_schools: List[str]
    next_7_days: List[str]

class GroundedItem(BaseModel):
            item: str
            evidence: str
    
class AnalyzeResponse(BaseModel):
        topic: str
        plan: Plan
        papers: List[Paper]
        selected_papers: List[SelectedPaper]
        access_gaps: List[GroundedItem]
        research_ideas: List[GroundedItem]
        action_plan: ActionPlan
        impact_score: int
        impact_reasons: List[str]

EDU_KEYWORDS = [
    "education", "student", "learning", "tutor", "classroom", "school",
    "mooc", "curriculum", "pedagogy", "edtech", "assessment"
]
ACCESS_KEYWORDS = [
    "low-bandwidth", "low bandwidth", "offline", "rural", "low resource",
    "low-income", "underserved", "multilingual", "access", "equity",
    "fairness", "digital divide"
]

def make_plan(topic: str) -> Plan:
    refined = [
        f'({topic}) AND (education OR learning OR students OR tutoring)',
        f'({topic}) AND (low-bandwidth OR offline OR rural OR low-resource)',
        f'(machine learning) AND (education access OR education equity OR digital divide)'
    ]
    criteria = [
        "ML/AI applied to education (not generic ML).",
        "Mentions constraints: low bandwidth/offline/rural/low-resource.",
        "Discusses datasets/populations or deployment feasibility.",
        "Evaluation beyond accuracy (cost/latency/usability/fairness).",
    ]
    return Plan(refined_queries=refined, screening_criteria=criteria)

def score_relevance(title: str, summary: str) -> tuple[int, str]:
    text = f"{title} {summary}".lower()
    edu_hits = sum(1 for k in EDU_KEYWORDS if k in text)
    access_hits = sum(1 for k in ACCESS_KEYWORDS if k in text)
    score = edu_hits * 2 + access_hits * 3

    reasons = []
    if edu_hits > 0:
        reasons.append("education terms found")
    if access_hits > 0:
        reasons.append("access/constraint terms found")
    if not reasons:
        reasons.append("weak match (likely generic ML)")
    return score, ", ".join(reasons)

def generate_gaps(topic: str, papers: List[Paper]) -> List[str]:
    gaps = [
        "Few papers test models in low-connectivity or offline classroom settings.",
        "Limited discussion of multilingual / local-language learning data.",
        "Evaluation often focuses on accuracy but not cost, latency, and device constraints.",
        "Not enough evidence of deployment in under-resourced schools (real-world pilots).",
        "Dataset representation gaps: rural learners and low-income regions are under-sampled."
    ]
    if len(papers) <= 1:
        gaps.insert(0, f"Most results were not education-access focused for '{topic}'. Better sources/queries needed.")
    return gaps

def generate_ideas(topic: str) -> List[str]:
    return [
        "Offline-first tutoring model for low-end Android; evaluate latency, memory, learning gains.",
        "Multilingual education dataset (local languages + code-switching) + benchmarks for small models.",
        "Low-bandwidth adaptive learning that syncs periodically; compare outcomes with/without connectivity.",
        "Fairness study: rural vs urban learners; report subgroup gaps + mitigation.",
        "Teacher-in-the-loop AI: lightweight feedback + explainable recommendations without constant internet."
    ]

def impact_score_and_reasons(papers: List[Paper]) -> tuple[int, List[str]]:
    if not papers:
        return 30, ["No strongly relevant education-access papers found in current retrieval."]
    strong = sum(1 for p in papers if "weak" not in p.relevance_reason)
    ratio = strong / len(papers)
    score = int(50 + 40 * ratio)
    score = max(0, min(100, score))
    reasons = [
        f"{len(papers)} papers passed screening.",
        "Higher when papers mention offline/low-bandwidth constraints and equity factors."
    ]
    return score, reasons

def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in Nova output.")
    return text[start:end+1]

def nova_reasoning(topic: str, papers: List[Paper]):
    REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    MODEL_ID = os.environ.get("NOVA_MODEL_ID")

    client = boto3.client("bedrock-runtime", region_name=REGION)

    paper_summaries = "\n\n".join(
    [f"Title: {p.title}\nArxiv: {p.arxiv_url}\nSummary: {p.summary[:800]}" for p in papers[:6]]
)

    prompt = f"""
Return ONLY valid JSON. No extra text.

Topic: {topic}

Candidate papers (choose the most useful):
{paper_summaries}

Output EXACTLY this JSON schema:
{{
  "selected_papers": [
    {{"title": "paper title", "arxiv_url": "url", "why_selected": "1-2 sentences"}}
  ],
  "access_gaps": [
    {{"item": "gap text", "evidence": "must reference at least 1 paper title"}}
  ],
  "research_ideas": [
    {{"item": "idea text", "evidence": "must reference at least 1 paper title"}}
  ],
  "action_plan": {{
    "for_teachers": ["...", "...", "..."],
    "for_schools": ["...", "...", "..."],
    "next_7_days": ["...", "...", "..."]
  }},
  "impact_score": 0,
  "impact_reasons": ["reason1", "reason2"]
}}

Rules:
- selected_papers must be exactly 3 items chosen from candidate papers.
- access_gaps must have exactly 5 objects.
- research_ideas must have exactly 5 objects.
- action_plan lists must each have exactly 3 items.
- evidence must mention at least one provided paper title.
- impact_score is integer 0-100.
- impact_reasons exactly 2 strings.
"""


    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {"role": "user", "content": [{"text": prompt}]}
        ],
        inferenceConfig={"maxTokens": 900, "temperature": 0.0},
    )

    text_output = response["output"]["message"]["content"][0]["text"]

    return text_output


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    plan = make_plan(req.topic)

    # 1) Retrieve
    client = arxiv.Client()
    seen = set()
    raw_results = []

    for q in plan.refined_queries:
        search = arxiv.Search(
            query=q,
            max_results=req.max_papers,
            sort_by=arxiv.SortCriterion.Relevance
        )

        for r in client.results(search):
            if r.entry_id in seen:
                continue
            seen.add(r.entry_id)
            raw_results.append(r)

    # 2) Score + sort
    scored = []
    for r in raw_results:
        s, reason = score_relevance(r.title or "", r.summary or "")
        scored.append((s, reason, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    # 3) Build top papers list (NO Nova calls here)
    papers: List[Paper] = []
    for s, reason, r in scored:
        if len(papers) >= 6:
            break
        if s < 4:
            continue
        papers.append(
            Paper(
                title=(r.title or "").strip(),
                authors=[a.name for a in (r.authors or [])],
                published=r.published.isoformat() if r.published else "",
                summary=(r.summary or "").strip(),
                pdf_url=r.pdf_url or "",
                arxiv_url=r.entry_id or "",
                relevance_reason=reason,
            )
        )

    # Helpers to normalize Nova output shapes
    def ensure_grounded_list(x):
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict) and "item" in x[0]:
            return x
        if isinstance(x, list) and (len(x) == 0 or isinstance(x[0], str)):
            return [{"item": s, "evidence": "N/A"} for s in x]
        return []

    def ensure_selected_list(x):
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict) and "title" in x[0]:
            return x
        return []

    def ensure_action_plan(x):
        if isinstance(x, dict):
            return {
                "for_teachers": x.get("for_teachers", [])[:3],
                "for_schools": x.get("for_schools", [])[:3],
                "next_7_days": x.get("next_7_days", [])[:3],
            }
        return {"for_teachers": [], "for_schools": [], "next_7_days": []}

    # 4) Call Nova ONCE (after papers are ready)
    try:
        nova_text = nova_reasoning(req.topic, papers)
        # If extract_json_object is not defined, replace next line with: nova_json = json.loads(nova_text)
        nova_json = json.loads(extract_json_object(nova_text))
    except Exception as e:
        fallback_selected = [
            {"title": p.title, "arxiv_url": p.arxiv_url, "why_selected": "Fallback selection (no Nova)."}
            for p in papers[:3]
        ]
        fallback_action = {
            "for_teachers": [
                "Use offline-friendly resources (downloadable PDFs/videos) and schedule sync times.",
                "Collect quick weekly feedback to adapt lessons for low-connectivity students.",
                "Pair students for peer learning when connectivity is limited."
            ],
            "for_schools": [
                "Create a shared local content library on one device/hotspot for classroom use.",
                "Set a weekly connectivity window to sync updates and upload progress.",
                "Prioritize low-cost devices and local-language materials where possible."
            ],
            "next_7_days": [
                "Pick 1 topic + create a small offline lesson pack (PDF + 5 questions).",
                "Run a 20-minute pilot with 3 students; note friction points.",
                "Refine and rerun; document before/after outcomes."
            ]
        }
        nova_json = {
            "selected_papers": fallback_selected,
            "access_gaps": [{"item": g, "evidence": "Fallback (no Nova)."} for g in generate_gaps(req.topic, papers)],
            "research_ideas": [{"item": i, "evidence": "Fallback (no Nova)."} for i in generate_ideas(req.topic)],
            "action_plan": fallback_action,
            "impact_score": 70,
            "impact_reasons": [f"Nova failed: {type(e).__name__}", "Used fallback."]
        }

    return AnalyzeResponse(
        topic=req.topic,
        plan=plan,
        papers=papers,
        selected_papers=ensure_selected_list(nova_json.get("selected_papers", [])),
        access_gaps=ensure_grounded_list(nova_json.get("access_gaps", [])),
        research_ideas=ensure_grounded_list(nova_json.get("research_ideas", [])),
        action_plan=ensure_action_plan(nova_json.get("action_plan", {})),
        impact_score=int(nova_json.get("impact_score", 70)),
        impact_reasons=nova_json.get("impact_reasons", ["No reasons provided."])
    )