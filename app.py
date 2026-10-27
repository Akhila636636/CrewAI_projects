import os
from typing import Annotated
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM

app = FastAPI(
    title="CrewAI Projects API",
    description="Ten independent CrewAI project endpoints powered by Gemini 3.1 Flash Lite Preview.",
    version="1.0.0",
)

class Query(BaseModel):
    input: Annotated[str, Field(min_length=1, description="Text input for the selected project")]

@app.get("/")
def home():
    return {
        "message": "CrewAI Projects API is running",
        "projects": ["/agent1", "/agent2", "/agent3", "/agent4", "/agent5",
                     "/agent6", "/agent7", "/agent8", "/agent9", "/agent10"],
        "method": "POST",
        "body": {"input": "your text"},
    }

@app.post("/agent1")
def agent1(q: Query):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it to your Colab secrets or deployment environment variables.",
        )
    llm = LLM(
        model="gemini/gemini-3.1-flash-lite-preview",
        api_key=api_key,
        temperature=0.3,
    )
    researcher = Agent(
        role="Researcher",
        goal="Identify useful background, key facts, examples, and caveats for the requested blog topic.",
        backstory="You are a careful research assistant. Separate established facts from assumptions and never invent citations.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    writer = Agent(
        role="Blog Writer",
        goal="Write a clear, engaging blog post for the intended audience using the supplied topic and research notes.",
        backstory="You write practical, well-structured articles with useful headings, examples, and a strong conclusion.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    editor = Agent(
        role="Editor",
        goal="Review the draft for clarity, factual caution, structure, grammar, and repetition; return the polished final post.",
        backstory="You are a rigorous editor who preserves useful detail and flags unsupported claims.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    research_task = Task(
        description="Research the following topic from your existing knowledge. Give key points, context, examples, likely reader questions, and claims that need verification. Topic/request: {input}",
        expected_output="Concise research notes with facts, examples, caveats, and a suggested outline.",
        agent=researcher,
    )
    writing_task = Task(
        description="Using the research notes and original request, write a useful blog post. Original request: {input}. Aim for approximately 600 words unless the user requests another length. Do not fabricate sources.",
        expected_output="A complete blog post draft with title, headings, introduction, body, and conclusion.",
        agent=writer,
        context=[research_task],
    )
    editing_task = Task(
        description="Edit the draft into a final publish-ready blog post. Preserve the user's intent, remove repetition, and mark any claim that requires external verification rather than pretending it was verified.",
        expected_output="Final polished blog post in Markdown.",
        agent=editor,
        context=[writing_task],
    )
    result = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, writing_task, editing_task],
        process=Process.sequential,
        verbose=False,
    ).kickoff(inputs={"input": q.input})
    return {"output": str(result)}

@app.post("/agent2")
def agent2(q: Query):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it to your Colab secrets or deployment environment variables.",
        )
    llm = LLM(
        model="gemini/gemini-3.1-flash-lite-preview",
        api_key=api_key,
        temperature=0.3,
    )
    site_analyst = Agent(
        role="Website and Product Analyst",
        goal="Extract competitor positioning, products, target customers, and differentiators from the text supplied by the user.",
        backstory="You analyse business information carefully and do not claim to have browsed URLs unless their content is supplied.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    pricing_analyst = Agent(
        role="Pricing Analyst",
        goal="Compare pricing details, packaging, and commercial models contained in the input.",
        backstory="You distinguish known prices from missing or unverified pricing information.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    strategist = Agent(
        role="Competitive Strategy Consultant",
        goal="Produce a concise competitor comparison and actionable SWOT-style recommendations.",
        backstory="You translate competitor observations into practical, evidence-conscious business recommendations.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    analysis_task = Task(
        description="Analyse the supplied competitor information. Identify each competitor's product, audience, positioning, strengths, and weaknesses. Input may contain URLs, notes, copied website text, or multiple competitors. Do not imply you accessed a URL if only the URL is provided. Input: {input}",
        expected_output="Structured competitor observations, with unknowns clearly marked.",
        agent=site_analyst,
    )
    pricing_task = Task(
        description="Extract and compare all pricing, plan, feature-tier, and monetisation information present in the input and prior analysis. Use 'not provided' for missing prices; do not invent numbers.",
        expected_output="A concise pricing and packaging comparison.",
        agent=pricing_analyst,
        context=[analysis_task],
    )
    strategy_task = Task(
        description="Combine the competitor and pricing analyses into a report. Include an executive summary, comparison table, strengths, weaknesses, opportunities, threats, information gaps, and 3 practical next steps. Input: {input}",
        expected_output="Markdown competitor analysis report with transparent unknowns and actionable recommendations.",
        agent=strategist,
        context=[analysis_task, pricing_task],
    )
    result = Crew(
        agents=[site_analyst, pricing_analyst, strategist],
        tasks=[analysis_task, pricing_task, strategy_task],
        process=Process.sequential,
        verbose=False,
    ).kickoff(inputs={"input": q.input})
    return {"output": str(result)}

@app.post("/agent3")
def agent3(q: Query):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it to your Colab secrets or deployment environment variables.",
        )
    llm = LLM(
        model="gemini/gemini-3.1-flash-lite-preview",
        api_key=api_key,
        temperature=0.3,
    )
    summarizer = Agent(
        role="Meeting Summarizer",
        goal="Summarize the supplied meeting transcript without losing important context.",
        backstory="You produce concise, neutral meeting summaries and never add details absent from the transcript.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    action_finder = Agent(
        role="Action Item Extractor",
        goal="Extract action items, owners, deadlines, and unresolved follow-ups.",
        backstory="You are meticulous about accountability and explicitly mark missing owners or due dates as unspecified.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    decision_logger = Agent(
        role="Decision Logger",
        goal="Identify decisions made, options discussed, and open questions.",
        backstory="You distinguish final decisions from suggestions and unresolved discussion.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    summary_task = Task(
        description="Summarize this meeting transcript in five to eight bullets, preserving key context and important updates. Do not infer missing details. Transcript/notes: {input}",
        expected_output="Concise meeting summary.",
        agent=summarizer,
    )
    actions_task = Task(
        description="Extract every explicit action item from the input. Present a table with task, owner, due date, and status/context. Write 'Unspecified' where the transcript gives no owner or date. Input: {input}",
        expected_output="Action-item table with no invented assignments or deadlines.",
        agent=action_finder,
    )
    decisions_task = Task(
        description="List confirmed decisions separately from proposals and open questions using the supplied transcript. Input: {input}",
        expected_output="Sections for decisions, proposals not yet approved, and open questions.",
        agent=decision_logger,
    )
    formatter = Agent(
        role="Meeting Notes Formatter",
        goal="Combine the summary, action items, and decision log into a clean meeting-notes document.",
        backstory="You create readable, consistent Markdown notes while preserving each specialist's findings.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    format_task = Task(
        description="Create final meeting notes with headings: Summary, Decisions, Action Items, Open Questions. Include only information supported by the input and specialist notes.",
        expected_output="Complete Markdown meeting notes ready to share.",
        agent=formatter,
        context=[summary_task, actions_task, decisions_task],
    )
    result = Crew(
        agents=[summarizer, action_finder, decision_logger, formatter],
        tasks=[summary_task, actions_task, decisions_task, format_task],
        process=Process.sequential,
        verbose=False,
    ).kickoff(inputs={"input": q.input})
    return {"output": str(result)}

@app.post("/agent4")
def agent4(q: Query):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it to your Colab secrets or deployment environment variables.",
        )
    llm = LLM(
        model="gemini/gemini-3.1-flash-lite-preview",
        api_key=api_key,
        temperature=0.3,
    )
    analyst = Agent(
        role="Data Analyst",
        goal="Inspect text containing tabular data or CSV rows and calculate or describe only defensible statistics.",
        backstory="You are methodical about data quality, missing values, units, and limitations. If data is too large or unclear, say so.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    insight_writer = Agent(
        role="Insight Writer",
        goal="Turn the analyst's findings into a clear report with the most important patterns and implications.",
        backstory="You communicate quantitative findings to nontechnical readers without overstating causality.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    reviewer = Agent(
        role="Report Reviewer",
        goal="Check that every numerical claim is supported by the supplied data and clarify limitations.",
        backstory="You are a skeptical reviewer who catches inconsistent calculations and unsupported conclusions.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    analysis_task = Task(
        description="Analyse the supplied data text/CSV. Identify columns, row-level patterns visible in the input, missing or inconsistent values, and useful descriptive statistics. Do not pretend to run code or inspect an attached file; work only from text supplied here. Input: {input}",
        expected_output="Data overview, defensible statistics, data-quality notes, and caveats.",
        agent=analyst,
    )
    insights_task = Task(
        description="Write a readable data report from the analysis. Highlight up to three key findings, explain why they matter, and distinguish observation from possible explanation.",
        expected_output="Report draft with executive summary, findings, and recommendations.",
        agent=insight_writer,
        context=[analysis_task],
    )
    review_task = Task(
        description="Review the report and check all numerical statements against the supplied input and analysis. Remove or qualify unsupported numbers, mention small-sample limitations, and return the final Markdown report.",
        expected_output="Reviewed Markdown data report with transparent caveats.",
        agent=reviewer,
        context=[analysis_task, insights_task],
    )
    result = Crew(
        agents=[analyst, insight_writer, reviewer],
        tasks=[analysis_task, insights_task, review_task],
        process=Process.sequential,
        verbose=False,
    ).kickoff(inputs={"input": q.input})
    return {"output": str(result)}

@app.post("/agent5")
def agent5(q: Query):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it to your Colab secrets or deployment environment variables.",
        )
    llm = LLM(
        model="gemini/gemini-3.1-flash-lite-preview",
        api_key=api_key,
        temperature=0.3,
    )
    classifier = Agent(
        role="Support Ticket Classifier",
        goal="Classify the incoming support message by category, urgency, and sentiment.",
        backstory="You route support issues consistently and never assume facts not present in the ticket.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    resolver = Agent(
        role="Customer Support Specialist",
        goal="Draft a helpful response or a safe escalation recommendation based on the ticket.",
        backstory="You are empathetic, clear, and careful not to promise refunds, account changes, or fixes without authorization.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    classification_task = Task(
        description="Classify this support ticket. Choose a category such as billing, technical issue, account/access, delivery, feature request, or other. Assign priority low/medium/high/urgent and explain briefly. Ticket: {input}",
        expected_output="Category, priority, sentiment, and short rationale.",
        agent=classifier,
    )
    response_task = Task(
        description="Using the ticket and classification, produce a support handling result. Include: category, priority, customer-facing reply, internal next step, and whether escalation is recommended. For security, payment, safety, or account ownership concerns, recommend the appropriate human review. Do not claim an action has already been taken.",
        expected_output="Structured support response with reply and escalation guidance.",
        agent=resolver,
        context=[classification_task],
    )
    result = Crew(
        agents=[classifier, resolver],
        tasks=[classification_task, response_task],
        process=Process.sequential,
        verbose=False,
    ).kickoff(inputs={"input": q.input})
    return {"output": str(result)}

@app.post("/agent6")
def agent6(q: Query):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it to your Colab secrets or deployment environment variables.",
        )
    llm = LLM(
        model="gemini/gemini-3.1-flash-lite-preview",
        api_key=api_key,
        temperature=0.3,
    )
    screener = Agent(
        role="Resume-to-Requirements Screener",
        goal="Compare the candidate information provided against the role criteria provided by the user.",
        backstory="You assess job-related evidence consistently, ignore protected personal characteristics, and never infer ability from names, age, gender, photos, or other protected traits.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    reviewer = Agent(
        role="Screening Quality Reviewer",
        goal="Audit the screening for evidence, relevance, fairness, and missing information.",
        backstory="You ensure recommendations are explainable and do not treat missing evidence as proof that a candidate lacks a skill.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    screening_task = Task(
        description="The input should contain a job description/criteria and resume text. Compare evidence against each job-related requirement. If one is missing, mark it 'Not evidenced'. Do not make a final hiring decision or use protected traits. Input: {input}",
        expected_output="Requirement-by-requirement evidence matrix, relevant strengths, gaps, and questions for a human interviewer.",
        agent=screener,
    )
    review_task = Task(
        description="Review the assessment for bias and unsupported assumptions. Provide a transparent fit summary based only on job-related evidence. If a numeric score is requested, explain its rubric and limitations. The output supports human review and must not be presented as an automatic hiring decision.",
        expected_output="Fair, evidence-based screening summary and human-review recommendation.",
        agent=reviewer,
        context=[screening_task],
    )
    result = Crew(
        agents=[screener, reviewer],
        tasks=[screening_task, review_task],
        process=Process.sequential,
        verbose=False,
    ).kickoff(inputs={"input": q.input})
    return {"output": str(result)}

@app.post("/agent7")
def agent7(q: Query):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it to your Colab secrets or deployment environment variables.",
        )
    llm = LLM(
        model="gemini/gemini-3.1-flash-lite-preview",
        api_key=api_key,
        temperature=0.3,
    )
    triager = Agent(
        role="Issue Triage Specialist",
        goal="Categorize a software issue and assess severity based on the supplied issue report.",
        backstory="You distinguish reproducible bugs from feature requests and ask for missing reproduction details.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    technical_analyst = Agent(
        role="Technical Issue Analyst",
        goal="Suggest diagnostic steps, relevant information to collect, and a concise response for the reporter.",
        backstory="You are a pragmatic engineer. You do not claim to have run code, accessed a repository, or reproduced a bug.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    triage_task = Task(
        description="Classify the issue as bug, feature request, question, or other. Estimate priority P0-P3 based only on stated impact, explain the rationale, and identify missing details. Issue: {input}",
        expected_output="Issue type, priority, impact rationale, and missing reproduction details.",
        agent=triager,
    )
    response_task = Task(
        description="Produce a triage record with title, type, priority, affected area if known, reproduction steps if supplied, diagnostic checklist, and suggested reply. Do not invent logs, fixes, or reproduction results.",
        expected_output="Actionable bug-triage record in Markdown.",
        agent=technical_analyst,
        context=[triage_task],
    )
    result = Crew(
        agents=[triager, technical_analyst],
        tasks=[triage_task, response_task],
        process=Process.sequential,
        verbose=False,
    ).kickoff(inputs={"input": q.input})
    return {"output": str(result)}

@app.post("/agent8")
def agent8(q: Query):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it to your Colab secrets or deployment environment variables.",
        )
    llm = LLM(
        model="gemini/gemini-3.1-flash-lite-preview",
        api_key=api_key,
        temperature=0.3,
    )
    repurposer = Agent(
        role="Content Repurposing Strategist",
        goal="Identify the central message, audience, tone, and reusable ideas in the supplied source content.",
        backstory="You preserve the source's meaning and do not add claims that were not in the original.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    social_writer = Agent(
        role="Multi-Platform Content Writer",
        goal="Adapt the source into distinct platform-ready pieces without repetitive copy.",
        backstory="You understand concise social writing and adapt format while keeping facts and voice consistent.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    quality_editor = Agent(
        role="Content Quality Editor",
        goal="Ensure every adapted item is faithful to the source, readable, and appropriate for its platform.",
        backstory="You remove unsupported claims, duplicated phrasing, and generic filler.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    strategy_task = Task(
        description="Analyse the supplied source content. Identify its main message, target audience, tone, factual claims, and possible content angles. Source: {input}",
        expected_output="Content brief with key message, audience, tone, and reusable angles.",
        agent=repurposer,
    )
    writing_task = Task(
        description="Repurpose the source into: (1) a LinkedIn post, (2) a short X-style post, (3) an Instagram caption, (4) a short email/newsletter intro, and (5) five possible headlines. Adapt each format distinctly and do not invent facts. Source: {input}",
        expected_output="Five clearly labelled content adaptations.",
        agent=social_writer,
        context=[strategy_task],
    )
    editing_task = Task(
        description="Edit the adaptations for clarity, platform fit, and fidelity to the source. Avoid repeated openings and unsupported statistics. Return the final content with clear labels.",
        expected_output="Final set of platform-ready content in Markdown.",
        agent=quality_editor,
        context=[strategy_task, writing_task],
    )
    result = Crew(
        agents=[repurposer, social_writer, quality_editor],
        tasks=[strategy_task, writing_task, editing_task],
        process=Process.sequential,
        verbose=False,
    ).kickoff(inputs={"input": q.input})
    return {"output": str(result)}

@app.post("/agent9")
def agent9(q: Query):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it to your Colab secrets or deployment environment variables.",
        )
    llm = LLM(
        model="gemini/gemini-3.1-flash-lite-preview",
        api_key=api_key,
        temperature=0.3,
    )
    qualifier = Agent(
        role="Lead Qualification Analyst",
        goal="Evaluate a sales lead against explicit qualification criteria and supplied evidence.",
        backstory="You use transparent business criteria, distinguish evidence from assumptions, and flag missing details.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    sales_writer = Agent(
        role="Sales Follow-up Writer",
        goal="Recommend an appropriate next step and draft a respectful follow-up message.",
        backstory="You write concise, personalized outreach without pretending the lead has agreed to anything.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    qualification_task = Task(
        description="Assess the lead details and any qualification criteria in the input. Consider fit, need, authority, budget, timing, and engagement only where evidence is supplied. If a score is useful, give a transparent 0-10 provisional score and explain the rubric; missing data must remain unknown. Input: {input}",
        expected_output="Qualification summary, evidence, unknowns, and provisional score with rationale.",
        agent=qualifier,
    )
    followup_task = Task(
        description="Based on the qualification, recommend a next action and draft a short follow-up message. Include a clear status such as high-priority follow-up, nurture, or needs more information, but do not treat it as a factual CRM status. Input: {input}",
        expected_output="Lead disposition recommendation, next action, and ready-to-edit outreach draft.",
        agent=sales_writer,
        context=[qualification_task],
    )
    result = Crew(
        agents=[qualifier, sales_writer],
        tasks=[qualification_task, followup_task],
        process=Process.sequential,
        verbose=False,
    ).kickoff(inputs={"input": q.input})
    return {"output": str(result)}

@app.post("/agent10")
def agent10(q: Query):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it to your Colab secrets or deployment environment variables.",
        )
    llm = LLM(
        model="gemini/gemini-3.1-flash-lite-preview",
        api_key=api_key,
        temperature=0.3,
    )
    researcher = Agent(
        role="Report Research and Outline Specialist",
        goal="Organize the supplied topic, audience, and source notes into a structured report plan.",
        backstory="You separate provided evidence from background knowledge and flag claims that need checking.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    writer = Agent(
        role="Report Writer",
        goal="Draft a clear report based on the provided brief and research plan.",
        backstory="You create structured reports with clear findings, recommendations, and limitations.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    approver = Agent(
        role="Report Quality Reviewer",
        goal="Review the draft and identify issues that should be resolved before human approval.",
        backstory="You check consistency, unsupported claims, privacy concerns, and missing context; you do not claim human approval.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    research_task = Task(
        description="Create a report outline from the user's topic, audience, purpose, and any supplied source notes. List evidence gaps and verification needs. Input: {input}",
        expected_output="Report outline, key questions, and verification checklist.",
        agent=researcher,
    )
    writing_task = Task(
        description="Draft the report using the brief and outline. Include an executive summary, main sections, recommendations, and limitations as appropriate. Do not invent evidence or citations. Input: {input}",
        expected_output="Complete report draft in Markdown.",
        agent=writer,
        context=[research_task],
    )
    review_task = Task(
        description="Review the report for clarity, consistency, unsupported factual claims, sensitive information, and missing caveats. Return an approval-review package with status 'Ready for human review' or 'Needs revision', a list of issues, and the revised report if feasible. Never claim a human has approved or published it.",
        expected_output="Review status, checklist, revision notes, and reviewed Markdown report.",
        agent=approver,
        context=[research_task, writing_task],
    )
    result = Crew(
        agents=[researcher, writer, approver],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=False,
    ).kickoff(inputs={"input": q.input})
    return {"output": str(result)}

