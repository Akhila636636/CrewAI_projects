import asyncio
import csv
import io
import json
import os
import re
import uuid
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from crewai import Agent, Crew, LLM, Process, Task
from pydantic import BaseModel, Field
from crewai.tools import tool
from crewai.flow.flow import Flow, start, listen, router, and_, or_
from crewai_tools import FileReadTool, FileWriterTool, ScrapeWebsiteTool

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini/gemini-3.1-flash-lite-preview")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
app = FastAPI(title="CREWAI Agent Hub", version="1.0.0")


def unwrap_inputs(payload):
    """Require the common request envelope: {"inputs": {...}}."""
    if not isinstance(payload, dict) or "inputs" not in payload:
        raise HTTPException(422, "Request body must contain an 'inputs' field.")
    value = payload["inputs"]
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        # Permit a single text input where a route expects one main text field.
        return {"text": value}
    raise HTTPException(422, "'inputs' must be an object or text string.")


def parse_form_inputs(value):
    """Parse the multipart 'inputs' field as a JSON object or plain text."""
    if value is None or not str(value).strip():
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {"text": str(value).strip()}
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, str):
        return {"text": parsed}
    raise HTTPException(422, "Multipart 'inputs' must be a JSON object or text string.")


@app.get("/", response_class=HTMLResponse, summary="CREWAI Agent Hub")
def home():
    return """
    <!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>CREWAI Agent Hub</title><style>
    body{font:16px system-ui;max-width:900px;margin:40px auto;padding:0 18px;line-height:1.6}
    .card{padding:16px;border:1px solid #ddd;border-radius:12px;margin:12px 0}
    code{background:#f2f2f2;padding:3px 6px;border-radius:4px}
    </style></head><body><h1>CREWAI Agent Hub</h1>
    <p>Ten CrewAI learning projects powered by Gemini. Use the interactive API docs to send inputs.</p>
    <p><a href="/docs">Open API docs (/docs)</a></p>
    <div class="card"><b>Agent 1 — Blog Post Crew</b><br><code>POST /agent1</code> JSON: <code>{"inputs":{"topic":"..."}}</code></div>
    <div class="card"><b>Agent 2 — Competitor Analysis Crew</b><br><code>POST /agent2</code> JSON: <code>{"inputs":{"competitors":["url1","url2","url3"]}}</code>.</div>
    <div class="card"><b>Agent 3 — Meeting Notes Crew</b><br><code>POST /agent3</code> JSON: <code>{"inputs":{"transcript":"..."}}</code>; multipart uses <code>inputs</code> plus optional file.</div>
    <div class="card"><b>Agent 4 — Data Report Crew</b><br><code>POST /agent4</code> CSV upload or JSON <code>{"inputs":{"csv_content":"..."}}</code>.</div>
    <div class="card"><b>Agent 5 — Customer Support Flow</b><br><code>POST /agent5</code> JSON: <code>{"inputs":{"ticket":"..."}}</code></div>
    <div class="card"><b>Agent 6 — Resume Screening Flow</b><br><code>POST /agent6</code> multipart: <code>inputs</code> JSON with <code>job_description</code>, plus TXT/PDF files under <code>resumes</code>.</div>
    <div class="card"><b>Agent 7 — Bug Triage Flow</b><br><code>POST /agent7</code> JSON: <code>{"inputs":{"title":"...","body":"..."}}</code></div>
    <div class="card"><b>Agent 8 — Content Repurposing Flow</b><br><code>POST /agent8</code> JSON: <code>{"inputs":{"article":"..."}}</code>.</div>
    <div class="card"><b>Agent 9 — Lead Qualification Flow</b><br><code>POST /agent9</code> JSON: <code>{"inputs":{"name":"...","website":"..."}}</code> and optional details.</div>
    <div class="card"><b>Agent 10 — Approval Report Flow</b><br><code>POST /agent10</code> use <code>{"inputs":{"topic":"...","audience":"..."}}</code>; review endpoint also uses <code>inputs</code>.</div>
    </body></html>
    """


@app.get("/health", summary="Health check")
def health():
    return {"status": "running", "model": MODEL_NAME, "agent_endpoints": 10}


@app.post("/agent1", summary="Blog Post Crew")
async def agent1(request: Request):
    data = unwrap_inputs(await request.json())
    topic = str(data.get("topic", "")).strip()
    if not topic:
        raise HTTPException(422, "topic is required and must not be empty.")
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(503, "Set GEMINI_API_KEY before running this agent.")

    llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)

    researcher = Agent(
        role="Researcher",
        goal="Research the topic and gather relevant facts and source leads using DuckDuckGo.",
        backstory="You are a careful researcher. Distinguish supported facts from uncertain claims.",
        llm=llm, verbose=False
    )
    writer = Agent(
        role="Writer",
        goal="Write an approximately 600-word blog post based on the research.",
        backstory="You write clear, useful blog posts and never invent research findings.",
        llm=llm, verbose=False
    )
    editor = Agent(
        role="Editor",
        goal="Improve tone, clarity, structure and factual consistency.",
        backstory="You are a meticulous editor who flags uncertain claims.",
        llm=llm, verbose=False
    )

    try:
        from ddgs import DDGS
        found = list(DDGS().text(topic, max_results=5))
        research_notes = "\n".join(
            f"Title: {x.get('title','')}\nURL: {x.get('href', x.get('url',''))}\nSnippet: {x.get('body','')}"
            for x in found
        )
        if not research_notes.strip():
            research_notes = "No search results were returned. Clearly label claims that require verification."
    except Exception as exc:
        research_notes = f"Web research unavailable ({type(exc).__name__}). Clearly label uncertain claims and do not invent sources."

    research_task = Task(
        description="Research this topic: {topic}\nDuckDuckGo results:\n{research_notes}\n"
                    "Organize key facts, useful source URLs, caveats, and claims needing verification.",
        expected_output="A research brief with source URLs and clearly stated uncertainties.",
        agent=researcher
    )
    writing_task = Task(
        description="Write an approximately 600-word blog post about {topic} using the research brief. "
                    "Do not invent facts or citations; label uncertain claims.",
        expected_output="A complete blog post draft of approximately 600 words in Markdown.",
        agent=writer, context=[research_task]
    )
    editing_task = Task(
        description="Edit the draft about {topic} for clarity, structure, tone and factual consistency. "
                    "Preserve source URLs and label uncertain claims. Return only the final Markdown post.",
        expected_output="A polished approximately 600-word Markdown blog post.",
        agent=editor, context=[writing_task]
    )
    try:
        result = await asyncio.to_thread(
            lambda: Crew(agents=[researcher, writer, editor], tasks=[research_task, writing_task, editing_task],
                         process=Process.sequential, verbose=False).kickoff(
                             inputs={"topic": topic, "research_notes": research_notes})
        )
    except Exception as exc:
        raise HTTPException(502, f"Blog Post Crew failed: {type(exc).__name__}. Check the server log.")
    content = str(result)
    (OUTPUT_DIR / "blog_post.md").write_text(content, encoding="utf-8")
    return {"agent": "Blog Post Crew", "output": content, "file": "outputs/blog_post.md"}


@app.post("/agent2", summary="Competitor Analysis Crew")
async def agent2(request: Request):
    data = unwrap_inputs(await request.json())
    competitors = data.get("competitors")
    if not isinstance(competitors, list) or len(competitors) != 3:
        raise HTTPException(422, "competitors must contain exactly three website URLs.")
    for url in competitors:
        parsed = urlparse(str(url))
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise HTTPException(422, f"Invalid competitor URL: {url}")
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(503, "Set GEMINI_API_KEY before running this agent.")

    llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
    manager = Agent(
        role="Competitor Analysis Manager",
        goal="Delegate competitor research and create an evidence-backed comparison and SWOT analysis.",
        backstory="You coordinate a site scraper, pricing analyst, news analyst and strategist.",
        llm=llm, allow_delegation=True, verbose=False
    )
    scraper = Agent(
        role="Site Scraper",
        goal="Summarize information available on competitor websites.",
        backstory="You use only accessible public page evidence and identify inaccessible data.",
        llm=llm, verbose=False
    )
    pricing = Agent(
        role="Pricing Analyst",
        goal="Compare available plans and prices without inventing missing values.",
        backstory="Unavailable prices must be labelled unavailable, never free or zero.",
        llm=llm, verbose=False
    )
    news = Agent(
        role="News Analyst",
        goal="Find recent relevant public news about the competitors.",
        backstory="You distinguish search snippets from verified facts.",
        llm=llm, verbose=False
    )
    strategist = Agent(
        role="Strategist",
        goal="Produce an evidence-backed competitor comparison and SWOT table.",
        backstory="You do not treat missing evidence as proof of absence.",
        llm=llm, verbose=False
    )
    site_evidence = []
    for url in competitors:
        try:
            response = requests.get(str(url), timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for node in soup(["script", "style", "nav", "footer", "noscript"]):
                node.decompose()
            text = " ".join(soup.get_text(" ").split())[:7000]
            site_evidence.append(f"URL: {url}\nAccessible page text:\n{text or '[No readable text]'}")
        except Exception as exc:
            site_evidence.append(f"URL: {url}\nWebsite unavailable: {type(exc).__name__}. Do not guess its details.")
    try:
        from ddgs import DDGS
        search_evidence = []
        for url in competitors:
            try:
                hits = list(DDGS().text(f'"{urlparse(url).netloc}" news', max_results=3))
                search_evidence.append(f"{url}\n" + "\n".join(
                    f"{h.get('title','')} | {h.get('href',h.get('url',''))} | {h.get('body','')}" for h in hits
                ))
            except Exception:
                search_evidence.append(f"{url}: news search unavailable.")
        news_notes = "\n\n".join(search_evidence)
    except Exception:
        news_notes = "DuckDuckGo search unavailable; recent news could not be verified."

    t_scrape = Task(
        description="Analyze the three competitor pages and extract products, audience, positioning, differentiators and evidence. "
                    "Page evidence:\n{site_evidence}",
        expected_output="Three evidence-based competitor summaries with missing data clearly marked.",
        agent=scraper
    )
    t_price = Task(
        description="Compare pricing and plan details using only the supplied page evidence. If a price is not visible, say unavailable. "
                    "Evidence:\n{site_evidence}",
        expected_output="Pricing comparison table; unavailable prices explicitly marked.",
        agent=pricing, context=[t_scrape]
    )
    t_news = Task(
        description="Summarize recent news evidence for these competitors. If search failed or evidence is absent, say so. "
                    "News search notes:\n{news_notes}",
        expected_output="Recent news summary with URLs or an explicit unavailable note.",
        agent=news
    )
    t_strategy = Task(
        description="Create a Markdown report with competitor summaries, comparison table, pricing availability, recent news and a SWOT table. "
                    "Base every claim on this evidence. Site evidence:\n{site_evidence}\nNews evidence:\n{news_notes}",
        expected_output="Complete evidence-backed competitor report in Markdown.",
        agent=strategist, context=[t_scrape, t_price, t_news]
    )
    try:
        result = await asyncio.to_thread(
            lambda: Crew(agents=[scraper, pricing, news, strategist],
                         tasks=[t_scrape, t_price, t_news, t_strategy],
                         process=Process.hierarchical, manager_agent=manager, verbose=False).kickoff(
                             inputs={"site_evidence": "\n\n".join(site_evidence), "news_notes": news_notes})
        )
    except Exception as exc:
        raise HTTPException(502, f"Competitor Analysis Crew failed: {type(exc).__name__}. Check the server log.")
    content = str(result)
    (OUTPUT_DIR / "report.md").write_text(content, encoding="utf-8")
    return {"agent": "Competitor Analysis Crew", "output": content, "file": "outputs/report.md",
            "competitors": competitors}


@app.post("/agent3", summary="Meeting Notes Crew")
async def agent3(request: Request):
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        form_inputs = parse_form_inputs(form.get("inputs"))
        transcript = str(form_inputs.get("transcript", form_inputs.get("text", ""))).strip()
        uploaded = form.get("file")
        if uploaded and hasattr(uploaded, "read"):
            raw = await uploaded.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Transcript file exceeds 5 MB.")
            transcript = raw.decode("utf-8", errors="replace").strip()
    else:
        data = unwrap_inputs(await request.json())
        transcript = str(data.get("transcript", data.get("text", ""))).strip()
    if not transcript:
        raise HTTPException(422, "Provide non-empty transcript text or upload a transcript.txt file.")
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(503, "Set GEMINI_API_KEY before running this agent.")

    llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
    summarizer = Agent(role="Summarizer", goal="Create a concise five-line meeting summary.",
                       backstory="You summarize only what the transcript supports.", llm=llm, verbose=False)
    action_finder = Agent(role="Action Finder", goal="Extract action items, owners and due dates.",
                          backstory="If owner or date is missing, write 'unspecified'. Never guess.", llm=llm, verbose=False)
    decision_logger = Agent(role="Decision Logger", goal="List decisions made during the meeting.",
                            backstory="If no decision is identified, explicitly say none was identified.", llm=llm, verbose=False)
    formatter = Agent(role="Formatter", goal="Combine the summary, actions and decisions into notes.md.",
                      backstory="You preserve all extracted details and use the required headings.", llm=llm, verbose=False)

    summary_task = Task(description="Summarize this meeting in exactly five concise lines. Transcript:\n{transcript}",
                        expected_output="Five-line summary.", agent=summarizer, async_execution=True)
    actions_task = Task(description="Extract action items as a table with action, owner and due date. Use 'unspecified' when absent. Transcript:\n{transcript}",
                        expected_output="Action-item table; missing owners/dates marked unspecified.", agent=action_finder, async_execution=True)
    decisions_task = Task(description="List explicit decisions in the transcript. If none are identifiable, say none were identified. Transcript:\n{transcript}",
                          expected_output="Decisions list or explicit no-decisions note.", agent=decision_logger, async_execution=True)
    format_task = Task(description="Combine the completed task results into Markdown with sections Summary, Action Items, and Decisions. Do not add facts.",
                       expected_output="Complete meeting notes Markdown.", agent=formatter,
                       context=[summary_task, actions_task, decisions_task])
    try:
        result = await asyncio.to_thread(
            lambda: Crew(agents=[summarizer, action_finder, decision_logger, formatter],
                         tasks=[summary_task, actions_task, decisions_task, format_task],
                         process=Process.sequential, verbose=False).kickoff(inputs={"transcript": transcript})
        )
    except Exception as exc:
        raise HTTPException(502, f"Meeting Notes Crew failed: {type(exc).__name__}. Check the server log.")
    output = str(result)
    (OUTPUT_DIR / "notes.md").write_text(output, encoding="utf-8")
    return {"agent": "Meeting Notes Crew", "output": output, "file": "outputs/notes.md"}



# ---------------------------------------------------------------------------
# Agent 4 — Data Report Crew (Crew + FileReadTool + pandas tool + FileWriterTool)
# ---------------------------------------------------------------------------
class SalesStatistics(BaseModel):
    row_count: int = Field(description="Number of valid sales rows")
    total_units: float
    total_revenue: float
    average_price: float
    revenue_by_product: dict[str, float]
    units_by_product: dict[str, float]


@tool("Pandas sales CSV analyzer")
def analyze_sales_csv(csv_path: str) -> str:
    """Read a local sales CSV with product, units, price columns and return calculated statistics."""
    frame = pd.read_csv(csv_path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    required = {"product", "units", "price"}
    if not required.issubset(frame.columns):
        return "ERROR: CSV must contain product, units, price columns."
    frame["units"] = pd.to_numeric(frame["units"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.dropna(subset=["product", "units", "price"])
    frame["revenue"] = frame["units"] * frame["price"]
    stats = {
        "row_count": int(len(frame)),
        "total_units": float(frame["units"].sum()),
        "total_revenue": round(float(frame["revenue"].sum()), 2),
        "average_price": round(float(frame["price"].mean()), 2) if len(frame) else 0.0,
        "revenue_by_product": {str(k): round(float(v), 2) for k, v in frame.groupby("product")["revenue"].sum().items()},
        "units_by_product": {str(k): float(v) for k, v in frame.groupby("product")["units"].sum().items()},
    }
    return json.dumps(stats, indent=2)


@app.post("/agent4", summary="Data Report Crew")
async def agent4(request: Request):
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        form_inputs = parse_form_inputs(form.get("inputs"))
        uploaded = form.get("file")
        if not uploaded or not hasattr(uploaded, "read"):
            raise HTTPException(422, "Upload sales.csv using the 'file' field.")
        raw = await uploaded.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "CSV file exceeds 5 MB.")
        csv_content = raw.decode("utf-8-sig", errors="replace")
        filename = Path(getattr(uploaded, "filename", "sales.csv") or "sales.csv").name
        if not csv_content:
            csv_content = str(form_inputs.get("csv_content", form_inputs.get("text", ""))).strip()
    else:
        data = unwrap_inputs(await request.json())
        csv_content = str(data.get("csv_content", data.get("text", ""))).strip()
        filename = "sales.csv"
    if not csv_content:
        raise HTTPException(422, "Provide a CSV upload or csv_content.")
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(503, "Set GEMINI_API_KEY before running this agent.")
    if not filename.lower().endswith(".csv"):
        raise HTTPException(415, "Agent 4 accepts CSV files only.")

    input_path = OUTPUT_DIR / f"agent4_{uuid.uuid4().hex}.csv"
    input_path.write_text(csv_content, encoding="utf-8")
    try:
        df = pd.read_csv(io.StringIO(csv_content))
    except Exception as exc:
        input_path.unlink(missing_ok=True)
        raise HTTPException(422, f"Could not parse CSV: {type(exc).__name__}.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"product", "units", "price"}.issubset(df.columns):
        input_path.unlink(missing_ok=True)
        raise HTTPException(422, "CSV must contain product, units, price columns.")
    llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.1)
    file_reader = FileReadTool()
    file_writer = FileWriterTool()
    analyst = Agent(role="Data Analyst", goal="Read sales.csv, use the pandas tool to calculate accurate statistics, and return structured statistics.",
                    backstory="Never estimate numbers. Use the provided FileReadTool and pandas analysis tool.", llm=llm,
                    tools=[file_reader, analyze_sales_csv], verbose=False)
    insight_writer = Agent(role="Insight Writer", goal="Write exactly the top three evidence-based findings from the statistics.",
                           backstory="Every numeric claim must match the supplied calculated values.", llm=llm, verbose=False)
    reviewer = Agent(role="Reviewer", goal="Verify every number and save the corrected report as report.md.",
                     backstory="Correct discrepancies against the calculated statistics and use FileWriterTool to save report.md.",
                     llm=llm, tools=[file_writer], verbose=False)
    t1 = Task(description="Use FileReadTool to read this uploaded CSV path: {csv_path}. Use the Pandas sales CSV analyzer on the same path. Return the statistics as the SalesStatistics Pydantic schema. Required columns are product, units, price.",
              expected_output="Validated structured sales statistics.", agent=analyst, output_pydantic=SalesStatistics)
    t2 = Task(description="Using these computed statistics, provide exactly three findings. Do not invent numbers.\nStatistics: {statistics}",
              expected_output="Exactly three evidence-backed findings.", agent=insight_writer, context=[t1])
    t3 = Task(description="Review the statistics and findings; check every number. Create a Markdown report with Summary, Top 3 Findings, and Review. Save it to report.md using FileWriterTool. Statistics: {statistics}",
              expected_output="Reviewed report.md with every number checked.", agent=reviewer, context=[t1, t2])
    try:
        result = await asyncio.to_thread(lambda: Crew(agents=[analyst, insight_writer, reviewer], tasks=[t1, t2, t3],
            process=Process.sequential, verbose=False).kickoff(inputs={"csv_path": str(input_path), "statistics": "Use the structured statistics returned by the first task."}))
    except Exception as exc:
        raise HTTPException(502, f"Data Report Crew failed: {type(exc).__name__}. Check the server log.")
    # Deterministic fallback ensures the promised artifact exists even if FileWriterTool is unavailable.
    report_path = OUTPUT_DIR / "report.md"
    output = str(result)
    if not report_path.exists() or not report_path.read_text(encoding="utf-8", errors="ignore").strip():
        report_path.write_text(output, encoding="utf-8")
    input_path.unlink(missing_ok=True)
    return {"agent": "Data Report Crew", "output": output, "file": str(report_path),
            "note": "Numeric values should be checked against the provided CSV."}


# ---------------------------------------------------------------------------
# Agent 5 — Customer Support Flow (Flow + router + conditional listeners)
# ---------------------------------------------------------------------------
class SupportState(BaseModel):
    ticket: str = ""
    category: str = ""
    priority: str = "medium"
    reason: str = ""
    resolution: str = ""
    route: str = ""


class CustomerSupportFlow(Flow[SupportState]):
    @start()
    def receive_ticket(self):
        self.state.ticket = self.state.ticket.strip()
        return self.state.ticket

    @listen(receive_ticket)
    def categorize_ticket(self):
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.1)
        agent = Agent(role="Ticket Categorizer", goal="Classify a support ticket as billing, account_access, or other and set priority.",
                      backstory="Return JSON only. Allowed priority: high, medium, low.", llm=llm, verbose=False)
        task = Task(description='Classify the ticket. Return JSON only: {"category":"billing|account_access|other","priority":"high|medium|low","reason":"..."}\nTicket: {ticket}',
                    expected_output="Classification JSON.", agent=agent)
        raw = str(Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(inputs={"ticket": self.state.ticket}))
        match = re.search(r"\{.*\}", raw, re.S)
        try: data = json.loads(match.group(0)) if match else {}
        except Exception: data = {}
        self.state.category = data.get("category", "other")
        if self.state.category not in ("billing", "account_access", "other"): self.state.category = "other"
        self.state.priority = data.get("priority", "medium")
        if self.state.priority not in ("high", "medium", "low"): self.state.priority = "medium"
        self.state.reason = str(data.get("reason", ""))
        return self.state.category

    @router(categorize_ticket)
    def route_by_category(self):
        return self.state.category

    @listen("billing")
    def handle_billing(self):
        self.state.route = "handle_billing"
        return self._resolve("Billing Support Handler", "Give safe billing troubleshooting steps; do not claim refunds or changes were made.")

    @listen("account_access")
    def handle_access(self):
        self.state.route = "handle_access"
        return self._resolve("Account Access Handler", "Never ask for passwords or verification codes. Do not claim account changes.")

    @listen("other")
    def handle_other(self):
        self.state.route = "handle_other"
        return self._resolve("General Support Handler", "Give practical troubleshooting and escalation steps; do not claim actions were performed.")

    def _resolve(self, role, rule):
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
        agent = Agent(role=role, goal="Draft an empathetic, safe customer-support resolution.", backstory=rule, llm=llm, verbose=False)
        task = Task(description="Respond to this ticket. Category={category}; priority={priority}; reason={reason}; rules={rule}\nTicket: {ticket}",
                    expected_output="Customer-facing resolution suggestion.", agent=agent)
        result = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(
            inputs={"ticket": self.state.ticket, "category": self.state.category, "priority": self.state.priority,
                    "reason": self.state.reason, "rule": rule})
        self.state.resolution = str(result)
        return self.state.resolution

    @listen(or_(handle_billing, handle_access, handle_other))
    def resolve_ticket(self):
        return {"ticket": self.state.ticket, "category": self.state.category, "priority": self.state.priority,
                "route": self.state.route, "reason": self.state.reason, "resolution": self.state.resolution}


@app.post("/agent5", summary="Customer Support Flow")
async def agent5(request: Request):
    data = unwrap_inputs(await request.json())
    ticket = str(data.get("ticket", "")).strip()
    if not ticket: raise HTTPException(422, "ticket is required.")
    if not os.getenv("GEMINI_API_KEY"): raise HTTPException(503, "Set GEMINI_API_KEY before running this agent.")
    try:
        result = await asyncio.to_thread(lambda: CustomerSupportFlow().kickoff(inputs={"ticket": ticket}))
    except Exception as exc:
        raise HTTPException(502, f"Customer Support Flow failed: {type(exc).__name__}. Check the server log.")
    return {"agent": "Customer Support Flow", "result": result}


# ---------------------------------------------------------------------------
# Agent 6 — Resume Screening Flow (FileReadTool + pypdf PDF text reader + router)
# ---------------------------------------------------------------------------
class ResumeState(BaseModel):
    job_description: str = ""
    resumes: list[dict] = Field(default_factory=list)
    current_resume: dict = Field(default_factory=dict)
    score: float = 0
    result: dict = Field(default_factory=dict)
    results: list[dict] = Field(default_factory=list)


class ResumeScreeningFlow(Flow[ResumeState]):
    @start()
    def load_resumes(self):
        if not self.state.job_description.strip() or not self.state.resumes:
            raise ValueError("job_description and at least one readable resume are required")
        return self.state.resumes

    @listen(load_resumes)
    def score_resume(self):
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.1)
        reader = FileReadTool()
        scorer = Agent(role="Resume Scorer", goal="Score each resume from 0 to 10 against job-related requirements.",
                       backstory="Use only job-relevant evidence. Never infer protected characteristics. Human makes final decisions.",
                       llm=llm, tools=[reader], verbose=False)
        self.state.results = []
        for item in self.state.resumes:
            task = Task(description="Use FileReadTool to read resume text at {resume_path}, then score it 0-10 against this job description. Return JSON only with score, rationale, strengths, gaps. Ignore protected characteristics.\nJob: {job_description}\nResume path: {resume_path}",
                        expected_output='JSON with score (0-10), rationale, strengths, gaps.', agent=scorer)
            try:
                raw = str(Crew(agents=[scorer], tasks=[task], process=Process.sequential, verbose=False).kickoff(
                    inputs={"job_description": self.state.job_description, "resume_path": item["path"]}))
                match = re.search(r"\{.*\}", raw, re.S)
                data = json.loads(match.group(0)) if match else {}
                score = max(0.0, min(10.0, float(data.get("score", 0))))
            except Exception as exc:
                data, score, raw = {"rationale": f"Scoring failed: {type(exc).__name__}"}, 0.0, ""
            self.state.results.append({"candidate": item["filename"], "score": score, "rationale": data.get("rationale", raw),
                                       "strengths": data.get("strengths", []), "gaps": data.get("gaps", [])})
        return self.state.results

    @router(score_resume)
    def check_score(self):
        # The flow routes each candidate by the score threshold.
        return "shortlist" if any(x["score"] >= 7 for x in self.state.results) else "reject"

    @listen("shortlist")
    def shortlist(self):
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
        agent = Agent(role="Summary Crew", goal="Summarize shortlisted candidates' strengths and gaps and propose interview questions.",
                      backstory="Decision support only; do not make a hiring decision.", llm=llm, verbose=False)
        rejection_agent = Agent(role="Rejection Draft Crew", goal="Draft a polite rejection email for recruiter review.",
                                backstory="Do not imply a final hiring decision or send the email.", llm=llm, verbose=False)
        for item in self.state.results:
            if item["score"] >= 7:
                task = Task(description="Summarize this candidate's job-related strengths and gaps and provide three interview questions.\\nCandidate: {candidate}\\nEvaluation: {evaluation}",
                            expected_output="Shortlist summary with strengths, gaps, and three interview questions.", agent=agent)
                item["follow_up"] = str(Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(
                    inputs={"candidate": item["candidate"], "evaluation": json.dumps(item)}))
                item["decision"] = "shortlist_for_human_review"
            else:
                task = Task(description="Draft a polite rejection email for recruiter review; do not mention an AI score. Candidate: {candidate}",
                            expected_output="Polite rejection email draft.", agent=rejection_agent)
                item["follow_up"] = str(Crew(agents=[rejection_agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(
                    inputs={"candidate": item["candidate"]}))
                item["decision"] = "below_threshold_for_human_review"
        return self.state.results

    @listen("reject")
    def reject(self):
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
        agent = Agent(role="Rejection Draft Crew", goal="Draft a polite rejection email for recruiter review.",
                      backstory="Do not imply a final hiring decision or send the email.", llm=llm, verbose=False)
        for item in self.state.results:
            if item["score"] < 7:
                task = Task(description="Draft a polite rejection email for recruiter review; do not mention an AI score. Candidate: {candidate}",
                            expected_output="Polite rejection email draft.", agent=agent)
                item["follow_up"] = str(Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(
                    inputs={"candidate": item["candidate"]}))
        return self.state.results

    @listen(or_(shortlist, reject))
    def save_results(self):
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=["candidate", "score", "decision", "rationale", "strengths", "gaps", "follow_up"])
        writer.writeheader()
        for item in self.state.results:
            writer.writerow({k: json.dumps(item.get(k, "")) if isinstance(item.get(k, ""), (dict, list)) else item.get(k, "") for k in writer.fieldnames})
        (OUTPUT_DIR / "results.csv").write_text(out.getvalue(), encoding="utf-8")
        return {"results": self.state.results, "csv": out.getvalue(), "file": "outputs/results.csv",
                "note": "Decision support only. Human review is required."}


@app.post("/agent6", summary="Resume Screening Flow")
async def agent6(request: Request):
    if "multipart/form-data" not in request.headers.get("content-type", ""):
        raise HTTPException(415, "Use multipart/form-data with an inputs JSON field containing job_description and one or more TXT/PDF files named resumes.")
    form = await request.form()
    form_inputs = parse_form_inputs(form.get("inputs"))
    job = str(form_inputs.get("job_description", "")).strip()
    uploads = form.getlist("resumes")
    if not job: raise HTTPException(422, "job_description is required.")
    if not uploads: raise HTTPException(422, "Upload at least one TXT or PDF file under resumes.")
    if len(uploads) > 50: raise HTTPException(422, "Upload no more than 50 resumes.")
    if not os.getenv("GEMINI_API_KEY"): raise HTTPException(503, "Set GEMINI_API_KEY before running this agent.")
    from pypdf import PdfReader
    from io import BytesIO
    staged = []
    for upload in uploads:
        filename = Path(getattr(upload, "filename", "resume.txt") or "resume.txt").name
        suffix = Path(filename).suffix.lower()
        raw = await upload.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES: continue
        try:
            if suffix == ".txt":
                text_content = raw.decode("utf-8-sig", errors="replace").strip()
            elif suffix == ".pdf":
                text_content = "\n".join(p.extract_text() or "" for p in PdfReader(BytesIO(raw)).pages).strip()
            else:
                continue
        except Exception:
            continue
        if not text_content: continue
        staged_path = OUTPUT_DIR / f"resume_{uuid.uuid4().hex}.txt"
        staged_path.write_text(text_content[:30000], encoding="utf-8")
        staged.append({"filename": filename, "path": str(staged_path)})
    if not staged: raise HTTPException(422, "No readable TXT/PDF resume text was found.")
    try:
        flow = ResumeScreeningFlow()
        result = await asyncio.to_thread(lambda: flow.kickoff(inputs={"job_description": job, "resumes": staged}))
    except Exception as exc:
        raise HTTPException(502, f"Resume Screening Flow failed: {type(exc).__name__}. Check the server log.")
    for item in staged: Path(item["path"]).unlink(missing_ok=True)
    return {"agent": "Resume Screening Flow", "result": result}


# ---------------------------------------------------------------------------
# Agent 7 — Bug Triage Flow + category-specific crews
# ---------------------------------------------------------------------------
class BugTriageState(BaseModel):
    title: str = ""
    body: str = ""
    category: str = ""
    triage: str = ""
    reply: str = ""


class BugTriageFlow(Flow[BugTriageState]):
    @start()
    def read_issue(self):
        return {"title": self.state.title, "body": self.state.body}

    @listen(read_issue)
    def classify_issue(self):
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.1)
        agent = Agent(role="Issue Classifier", goal="Classify issue as bug, feature, or question.",
                      backstory='Return JSON only with category and reason.', llm=llm, verbose=False)
        task = Task(description='Classify this issue. Return JSON: {"category":"bug|feature|question","reason":"..."}\nTitle: {title}\nBody: {body}',
                    expected_output="Issue category JSON.", agent=agent)
        raw = str(Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(
            inputs={"title": self.state.title, "body": self.state.body}))
        match = re.search(r"\{.*\}", raw, re.S)
        try: data = json.loads(match.group(0)) if match else {}
        except Exception: data = {}
        self.state.category = data.get("category", "question")
        if self.state.category not in ("bug", "feature", "question"): self.state.category = "question"
        return self.state.category

    @router(classify_issue)
    def route_type(self):
        return self.state.category

    @listen("bug")
    def bug_crew(self):
        return self._run_specialist("Bug Crew", "Assess severity and diagnostic steps. Include P0-P3 severity, expected/actual behavior and reproduction steps.")

    @listen("feature")
    def feature_crew(self):
        return self._run_specialist("Feature Crew", "Draft a user story, user value, acceptance criteria, edge cases and open questions.")

    @listen("question")
    def answer_crew(self):
        return self._run_specialist("Answer Crew", "Draft a cautious answer. No docs were supplied, so identify what must be verified.")

    def _run_specialist(self, role, goal):
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
        agent = Agent(role=role, goal=goal, backstory="Do not claim an action occurred or invent unverified facts.", llm=llm, verbose=False)
        task = Task(description="{goal}\nTitle: {title}\nBody: {body}", expected_output="Useful evidence-based issue triage.", agent=agent)
        result = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(
            inputs={"goal": goal, "title": self.state.title, "body": self.state.body})
        self.state.triage = str(result)
        return self.state.triage

    @listen(or_(bug_crew, feature_crew, answer_crew))
    def post_reply(self):
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
        agent = Agent(role="Issue Tracker Reply Writer", goal="Draft a concise reply based on the triage result.",
                      backstory="Never claim an action was performed.", llm=llm, verbose=False)
        task = Task(description="Draft a concise issue-tracker comment based on this triage:\n{triage}",
                    expected_output="Suggested reply comment.", agent=agent)
        self.state.reply = str(Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(
            inputs={"triage": self.state.triage}))
        return {"category": self.state.category, "triage": self.state.triage, "reply": self.state.reply}


@app.post("/agent7", summary="Bug Triage Flow")
async def agent7(request: Request):
    data = unwrap_inputs(await request.json())
    title, body = str(data.get("title", "")).strip(), str(data.get("body", "")).strip()
    if not title or not body: raise HTTPException(422, "Both title and body are required.")
    if not os.getenv("GEMINI_API_KEY"): raise HTTPException(503, "Set GEMINI_API_KEY before running this agent.")
    try:
        result = await asyncio.to_thread(lambda: BugTriageFlow().kickoff(inputs={"title": title, "body": body}))
    except Exception as exc:
        raise HTTPException(502, f"Bug Triage Flow failed: {type(exc).__name__}. Check the server log.")
    return {"agent": "Bug Triage Flow", "result": result}


# ---------------------------------------------------------------------------
# Agent 8 — Content Repurposing Flow (parallel crews joined with and_)
# ---------------------------------------------------------------------------
class ContentState(BaseModel):
    article: str = ""
    article_text: str = ""
    key_points: str = ""
    linkedin: str = ""
    twitter: str = ""
    newsletter: str = ""
    output: dict = Field(default_factory=dict)


class ContentRepurposingFlow(Flow[ContentState]):
    @start()
    def load_article(self):
        value = self.state.article.strip()
        parsed = urlparse(value)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            response = requests.get(value, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for node in soup(["script", "style", "nav", "footer", "noscript"]): node.decompose()
            value = " ".join(soup.get_text(" ").split())[:30000]
        if not value: raise ValueError("Article text could not be read.")
        self.state.article_text = value
        return value

    @listen(load_article)
    def extract_points(self):
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
        agent = Agent(role="Article Analyst", goal="Extract thesis, evidence, caveats and takeaways.",
                      backstory="Preserve the source meaning and never add unsupported claims.", llm=llm, verbose=False)
        task = Task(description="Extract the main thesis, facts, evidence, caveats and takeaways:\n{article_text}",
                    expected_output="Structured article brief.", agent=agent)
        self.state.key_points = str(Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(
            inputs={"article_text": self.state.article_text}))
        return self.state.key_points

    @listen(extract_points)
    def linkedin_crew(self):
        self.state.linkedin = self._write("LinkedIn Crew", "Write a professional LinkedIn post around 150 words with a truthful hook.")
        return self.state.linkedin

    @listen(extract_points)
    def twitter_crew(self):
        self.state.twitter = self._write("Twitter Crew", "Write a coherent six-post numbered thread, 1/6 through 6/6.")
        return self.state.twitter

    @listen(extract_points)
    def newsletter_crew(self):
        self.state.newsletter = self._write("Newsletter Crew", "Write a newsletter introduction, three key bullets and closing takeaway.")
        return self.state.newsletter

    def _write(self, role, instruction):
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.3)
        agent = Agent(role=role, goal=instruction, backstory="Use only the source-supported key points.", llm=llm, verbose=False)
        task = Task(description="{instruction}\nArticle brief:\n{key_points}", expected_output=instruction, agent=agent)
        return str(Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(
            inputs={"instruction": instruction, "key_points": self.state.key_points}))

    @listen(and_(linkedin_crew, twitter_crew, newsletter_crew))
    def publish_pack(self):
        # and_ ensures all three branch crews finish before the three files are written.
        contents = [self.state.linkedin, self.state.twitter, self.state.newsletter]
        names = ["linkedin_post.md", "x_thread.md", "newsletter.md"]
        for name, content in zip(names, contents):
            (OUTPUT_DIR / name).write_text(content, encoding="utf-8")
        self.state.output = {"linkedin_post": self.state.linkedin, "x_thread": self.state.twitter,
                             "newsletter": self.state.newsletter,
                             "files": [f"outputs/{n}" for n in names]}
        return self.state.output


@app.post("/agent8", summary="Content Repurposing Flow")
async def agent8(request: Request):
    data = unwrap_inputs(await request.json())
    article = str(data.get("article", "")).strip()
    if not article: raise HTTPException(422, "article is required: provide article text or a public URL.")
    if not os.getenv("GEMINI_API_KEY"): raise HTTPException(503, "Set GEMINI_API_KEY before running this agent.")
    try:
        result = await asyncio.to_thread(lambda: ContentRepurposingFlow().kickoff(inputs={"article": article}))
    except Exception as exc:
        raise HTTPException(502, f"Content Repurposing Flow failed: {type(exc).__name__}. Check the server log.")
    return {"agent": "Content Repurposing Flow", "result": result}


# ---------------------------------------------------------------------------
# Agent 9 — Lead Qualification Flow (ScrapeWebsiteTool + FileWriterTool)
# ---------------------------------------------------------------------------
class LeadState(BaseModel):
    name: str = ""
    website: str = ""
    business_need: str = ""
    budget: str = ""
    timeline: str = ""
    website_evidence: str = ""
    enrichment: str = ""
    score: int = 0
    category: str = "warm"
    rationale: str = ""
    action: str = ""
    next_step: str = ""
    record: dict = Field(default_factory=dict)


class LeadQualificationFlow(Flow[LeadState]):
    @start()
    def capture_lead(self):
        return {"name": self.state.name, "website": self.state.website}

    @listen(capture_lead)
    def enrich_lead(self):
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
        scraper = ScrapeWebsiteTool(website_url=self.state.website)
        agent = Agent(role="Enrich Crew", goal="Scrape the lead's public website and summarize relevant company context.",
                      backstory="Use website evidence only and explicitly state unknowns.", llm=llm, tools=[scraper], verbose=False)
        task = Task(description="Use ScrapeWebsiteTool to examine this public website and summarize evidence relevant to qualification.\nName: {name}\nWebsite: {website}\nBusiness need: {business_need}\nBudget: {budget}\nTimeline: {timeline}",
                    expected_output="Website-based enrichment with evidence and unknowns.", agent=agent)
        self.state.website_evidence = str(Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(
            inputs={"name": self.state.name, "website": self.state.website, "business_need": self.state.business_need,
                    "budget": self.state.budget, "timeline": self.state.timeline}))
        return self.state.website_evidence

    @listen(enrich_lead)
    def score_lead(self):
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.1)
        agent = Agent(role="Lead Scoring Analyst", goal="Score lead fit from 0 to 100 and classify hot, warm, cold.",
                      backstory="Use transparent criteria; do not infer missing budget or intent.", llm=llm, verbose=False)
        task = Task(description='Return JSON only: {"score":0,"category":"hot|warm|cold","rationale":"..."}.\nName: {name}\nNeed: {business_need}\nBudget: {budget}\nTimeline: {timeline}\nWebsite evidence: {website_evidence}',
                    expected_output="JSON with score, category and rationale.", agent=agent)
        raw = str(Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(
            inputs={"name": self.state.name, "business_need": self.state.business_need, "budget": self.state.budget,
                    "timeline": self.state.timeline, "website_evidence": self.state.website_evidence}))
        match = re.search(r"\{.*\}", raw, re.S)
        try: data = json.loads(match.group(0)) if match else {}
        except Exception: data = {}
        try: self.state.score = max(0, min(100, int(float(data.get("score", 0)))))
        except Exception: self.state.score = 0
        self.state.category = data.get("category", "warm")
        if self.state.category not in ("hot", "warm", "cold"): self.state.category = "warm"
        self.state.rationale = str(data.get("rationale", raw))
        return self.state.category

    @router(score_lead)
    def score_router(self):
        return self.state.category

    @listen("hot")
    def outreach_crew(self):
        self.state.action = "outreach_draft"
        return self._branch("Outreach Crew", "Draft a personalized first-touch email with subject and CTA. Do not send it.")

    @listen("warm")
    def nurture_crew(self):
        self.state.action = "nurture_recommendation"
        return self._branch("Nurture Crew", "Recommend a low-pressure follow-up and optional message draft. Do not send or add anyone to a list.")

    @listen("cold")
    def archive_lead(self):
        self.state.action = "archive_demo_record"
        return self._branch("Cold Lead Recorder", "Write an internal note explaining fit concerns and what evidence could change the classification. Do not contact the lead.")

    def _branch(self, role, goal):
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
        agent = Agent(role=role, goal=goal, backstory="Never send messages or claim external actions were completed.", llm=llm, verbose=False)
        task = Task(description="{goal}\nLead: {name}\nWebsite: {website}\nNeed: {business_need}\nBudget: {budget}\nTimeline: {timeline}\nScore: {score}; category: {category}; rationale: {rationale}",
                    expected_output="Recommended next step or draft only.", agent=agent)
        self.state.next_step = str(Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(
            inputs={"goal": goal, "name": self.state.name, "website": self.state.website, "business_need": self.state.business_need,
                    "budget": self.state.budget, "timeline": self.state.timeline, "score": self.state.score,
                    "category": self.state.category, "rationale": self.state.rationale}))
        return self.state.next_step

    @listen(or_(outreach_crew, nurture_crew, archive_lead))
    def log_lead(self):
        record = {"name": self.state.name, "website": self.state.website, "score": self.state.score,
                  "category": self.state.category, "recommended_action": self.state.action, "next_step": self.state.next_step}
        writer = FileWriterTool()
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.1)
        agent = Agent(role="Lead Log Writer", goal="Save the lead qualification record to leads.csv using FileWriterTool.",
                      backstory="Write exactly the supplied record; do not modify its values.", llm=llm, tools=[writer], verbose=False)
        task = Task(description="Save this lead record as CSV at outputs/leads.csv using FileWriterTool. Record: {record}",
                    expected_output="Lead record saved to outputs/leads.csv.", agent=agent)
        try:
            Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(inputs={"record": json.dumps(record)})
        except Exception:
            pass
        csv_path = OUTPUT_DIR / "leads.csv"
        # Append a deterministic record if the tool did not create the CSV.
        exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=list(record.keys()))
            if not exists: writer_csv.writeheader()
            writer_csv.writerow(record)
        self.state.record = record
        return record


@app.post("/agent9", summary="Lead Qualification Flow")
async def agent9(request: Request):
    data = unwrap_inputs(await request.json())
    name, website = str(data.get("name", "")).strip(), str(data.get("website", "")).strip()
    parsed = urlparse(website)
    if not name: raise HTTPException(422, "name is required.")
    if parsed.scheme not in ("http", "https") or not parsed.netloc: raise HTTPException(422, "website must be a valid HTTP/HTTPS URL.")
    if not os.getenv("GEMINI_API_KEY"): raise HTTPException(503, "Set GEMINI_API_KEY before running this agent.")
    inputs = {"name": name, "website": website, "business_need": str(data.get("business_need", "")),
              "budget": str(data.get("budget", "")), "timeline": str(data.get("timeline", ""))}
    try:
        result = await asyncio.to_thread(lambda: LeadQualificationFlow().kickoff(inputs=inputs))
    except Exception as exc:
        raise HTTPException(502, f"Lead Qualification Flow failed: {type(exc).__name__}. Check the server log.")
    return {"agent": "Lead Qualification Flow", "result": result,
            "note": "Lead score is advisory. No emails are sent."}


# ---------------------------------------------------------------------------
# Agent 10 — Approval Report Flow (DuckDuckGo research + human-review cycle)
# ---------------------------------------------------------------------------
PENDING_REPORTS = {}


class ApprovalState(BaseModel):
    topic: str = ""
    audience: str = ""
    research_notes: str = ""
    draft: str = ""


class ApprovalReportFlow(Flow[ApprovalState]):
    @start()
    def collect_topic(self):
        return {"topic": self.state.topic, "audience": self.state.audience}

    @listen(collect_topic)
    def research(self):
        try:
            from ddgs import DDGS
            hits = list(DDGS().text(f"{self.state.topic} background evidence", max_results=5))
            self.state.research_notes = "\n".join(
                f"{h.get('title','')} | {h.get('href', h.get('url',''))} | {h.get('body','')}" for h in hits)
        except Exception:
            self.state.research_notes = "DuckDuckGo search unavailable; claims need independent verification."
        if not self.state.research_notes.strip():
            self.state.research_notes = "No search results found; claims need independent verification."
        return self.state.research_notes

    @listen(research)
    def writing_crew(self):
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
        researcher = Agent(role="Research Crew", goal="Organize source-backed findings and caveats.",
                           backstory="Treat DuckDuckGo results as leads, not proof.", llm=llm, verbose=False)
        writer = Agent(role="Writing Crew", goal="Draft a structured report for the specified audience.",
                       backstory="Include limitations and never invent citations.", llm=llm, verbose=False)
        t1 = Task(description="Research {topic} for audience {audience}. Search notes:\n{research_notes}",
                  expected_output="Research brief with sources and caveats.", agent=researcher)
        t2 = Task(description="Write a Markdown report about {topic} for {audience}. Include executive summary, findings, recommendations and limitations. End with STATUS: AWAITING HUMAN REVIEW.",
                  expected_output="Markdown report awaiting human review.", agent=writer, context=[t1])
        self.state.draft = str(Crew(agents=[researcher, writer], tasks=[t1, t2], process=Process.sequential, verbose=False).kickoff(
            inputs={"topic": self.state.topic, "audience": self.state.audience, "research_notes": self.state.research_notes}))
        return self.state.draft


@app.post("/agent10", summary="Approval Report Flow — create draft for human review")
async def agent10(request: Request):
    data = unwrap_inputs(await request.json())
    topic, audience = str(data.get("topic", "")).strip(), str(data.get("audience", "")).strip()
    if not topic or not audience: raise HTTPException(422, "topic and audience are required.")
    if not os.getenv("GEMINI_API_KEY"): raise HTTPException(503, "Set GEMINI_API_KEY before running this agent.")
    try:
        result = await asyncio.to_thread(lambda: ApprovalReportFlow().kickoff(inputs={"topic": topic, "audience": audience}))
    except Exception as exc:
        raise HTTPException(502, f"Approval Report Flow failed: {type(exc).__name__}. Check the server log.")
    draft = result.get("draft", "") if isinstance(result, dict) else str(result)
    review_id = str(uuid.uuid4())
    PENDING_REPORTS[review_id] = {"topic": topic, "audience": audience, "draft": draft, "revisions": 0}
    return {"agent": "Approval Report Flow", "status": "awaiting_review", "review_id": review_id, "draft": draft,
            "next_step": "POST /agent10/review with review_id, review_decision='approved' or 'revise', and feedback when revising."}


@app.post("/agent10/review", summary="Approval Report Flow — human feedback")
async def agent10_review(request: Request):
    data = unwrap_inputs(await request.json())
    review_id = str(data.get("review_id", "")).strip()
    decision = str(data.get("review_decision", "")).strip().lower()
    feedback = str(data.get("feedback", "")).strip()
    if decision not in ("approved", "revise"): raise HTTPException(422, "review_decision must be 'approved' or 'revise'.")
    report = PENDING_REPORTS.get(review_id)
    if not report: raise HTTPException(404, "review_id not found. In-memory drafts can be lost after a server restart.")
    if decision == "approved":
        final_text = report["draft"]
        writer = FileWriterTool()
        llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.1)
        agent = Agent(role="Report Publisher", goal="Save the approved report as final.md using FileWriterTool.",
                      backstory="Publish only the draft approved by the human reviewer.", llm=llm, tools=[writer], verbose=False)
        task = Task(description="Save this approved Markdown report to outputs/final.md using FileWriterTool. Report:\n{draft}",
                    expected_output="Approved report saved to outputs/final.md.", agent=agent)
        try:
            await asyncio.to_thread(lambda: Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff(inputs={"draft": final_text}))
        except Exception:
            pass
        final_path = OUTPUT_DIR / "final.md"
        if not final_path.exists() or not final_path.read_text(encoding="utf-8", errors="ignore").strip():
            final_path.write_text(final_text, encoding="utf-8")
        return {"agent": "Approval Report Flow", "status": "approved", "review_id": review_id,
                "output": final_text, "file": "outputs/final.md"}
    if not feedback: raise HTTPException(422, "feedback is required when review_decision='revise'.")
    if report["revisions"] >= 3: raise HTTPException(409, "Maximum of three revisions reached.")
    llm = LLM(model=MODEL_NAME, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)
    reviser = Agent(role="Revision Crew", goal="Revise the report according to human reviewer feedback.",
                    backstory="Address each feedback item, preserve supported facts, and do not invent evidence.", llm=llm, verbose=False)
    task = Task(description="Revise the report according to feedback. Topic: {topic}; audience: {audience}\nFeedback: {feedback}\nDraft:\n{draft}\nEnd with STATUS: AWAITING HUMAN REVIEW.",
                expected_output="Revised Markdown awaiting another human review.", agent=reviser)
    try:
        revised = str(await asyncio.to_thread(lambda: Crew(agents=[reviser], tasks=[task], process=Process.sequential, verbose=False).kickoff(
            inputs={"topic": report["topic"], "audience": report["audience"], "feedback": feedback, "draft": report["draft"]})))
    except Exception as exc:
        raise HTTPException(502, f"Revision failed: {type(exc).__name__}. Check the server log.")
    report["draft"] = revised
    report["revisions"] += 1
    return {"agent": "Approval Report Flow", "status": "revision_requested", "review_id": review_id,
            "revision_number": report["revisions"], "draft": revised,
            "next_step": "Review this revision and submit approved or revise again."}
