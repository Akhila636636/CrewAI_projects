import asyncio
import csv
import io
import json
import logging
import os
import re
import uuid

from pathlib import Path
from urllib.parse import urlparse
from io import BytesIO

import pandas as pd
import requests

from bs4 import BeautifulSoup
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    File,
    Form,
)
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from crewai.flow.flow import (
    Flow,
    start,
    listen,
    router,
    and_,
    or_,
)

from crewai_tools import (
    FileReadTool,
    FileWriterTool,
    ScrapeWebsiteTool,
)

from pypdf import PdfReader


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini/gemini-3.1-flash-lite-preview",
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

app = FastAPI(
    title="CREWAI Agent Hub",
    version="1.0.0",
)


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


# ---------------------------------------------------------------------------
# Web UI (added): a self-contained project-library front end served at /ui.
# Pure addition: it does not touch any agent, route or helper defined elsewhere.
# ---------------------------------------------------------------------------
UI_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#FBF4E4">
<title>CrewAI Project Library</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%23FBF4E4'/%3E%3Crect x='5' y='8' width='5' height='18' rx='1.5' fill='%23F7B500'/%3E%3Crect x='12' y='5' width='5' height='21' rx='1.5' fill='%23C9167A'/%3E%3Crect x='19' y='10' width='5' height='16' rx='1.5' fill='%23B88B5E'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Figtree:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--cream:#fbf4e4;--cream-2:#f4e9d0;--paper:#fffcf5;--brown-100:#ecdec6;--brown-300:#cfae86;--brown-500:#9a7650;--ink:#33241a;--ink-2:#6b5644;--yellow:#f7b500;--yellow-soft:#ffe7a0;--magenta:#c9167a;--magenta-deep:#a50f64;--magenta-soft:#f9d6e9;--ok:#3f7d3a;--display:"Bricolage Grotesque",system-ui,sans-serif;--body:"Figtree",system-ui,sans-serif;--mono:"JetBrains Mono",ui-monospace,monospace;--r:14px}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--cream);color:var(--ink);font:16px/1.6 var(--body);-webkit-font-smoothing:antialiased}
a{color:inherit}h1,h2,h3,h4{font-family:var(--display);line-height:1.12;margin:0;letter-spacing:-.01em}
:focus-visible{outline:3px solid var(--magenta);outline-offset:2px;border-radius:6px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:14px 24px;max-width:1120px;margin:0 auto}
.brand{display:flex;align-items:center;gap:10px;text-decoration:none;font:800 20px var(--display)}
.topnav{display:flex;align-items:center;gap:18px;font-size:15px;font-weight:500}
.topnav a{text-decoration:none;color:var(--ink-2)}.topnav a:hover{color:var(--magenta)}
.pill{display:inline-flex;align-items:center;gap:8px;padding:5px 12px;border-radius:999px;background:var(--paper);border:1px solid var(--brown-100);font-size:13px;color:var(--ink-2)}
.pill i{width:8px;height:8px;border-radius:50%;background:var(--brown-300)}.pill.up i{background:var(--ok)}.pill.down i{background:var(--magenta)}
.hero{display:grid;grid-template-columns:1.05fr 1fr;gap:48px;align-items:end;padding:48px 0 40px}
.hero h1{font-size:clamp(38px,5.4vw,62px);font-weight:800;max-width:12ch}
.hero-copy p{font-size:18px;color:var(--ink-2);max-width:46ch;margin:18px 0 26px}
.hero-tools{display:grid;gap:14px}
.search input{width:100%;max-width:440px;padding:13px 16px;border-radius:12px;border:2px solid var(--brown-300);background:var(--paper);font:inherit;color:var(--ink)}
.search input:focus{border-color:var(--magenta);outline:none;box-shadow:0 0 0 4px var(--magenta-soft)}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{padding:7px 16px;border-radius:999px;border:2px solid var(--brown-300);background:transparent;font:600 14px var(--body);color:var(--ink);cursor:pointer}
.chip:hover{background:var(--cream-2)}.chip.on{background:var(--ink);border-color:var(--ink);color:var(--cream)}
.shelf{justify-self:end;width:100%;max-width:480px}
.spines{display:flex;align-items:flex-end;justify-content:center;gap:6px;padding:0 8px}
.spine{flex:1;max-width:42px;min-width:24px;display:flex;flex-direction:column;align-items:center;justify-content:space-between;gap:8px;padding:10px 0;border-radius:6px 6px 2px 2px;text-decoration:none;transform-origin:bottom;transition:transform .18s ease;border:2px solid var(--ink)}
.spine:hover,.spine:focus-visible{transform:translateY(-12px) rotate(-2deg)}
.spine b{font:700 13px var(--mono)}
.spine span{writing-mode:vertical-rl;transform:rotate(180deg);font:700 13px var(--display);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-height:74%}
.spine.crew{background:var(--yellow)}.spine.flow{background:var(--magenta);color:#fff}
.plank{height:14px;background:var(--brown-300);border:2px solid var(--ink);border-radius:3px;margin-top:-2px;box-shadow:0 6px 0 var(--brown-500)}
.legend{margin-top:22px;display:flex;gap:8px;align-items:center;font-size:13px;color:var(--ink-2);justify-content:center}
.legend i{width:12px;height:12px;border-radius:3px;border:2px solid var(--ink);margin-left:10px}.k-crew{background:var(--yellow)}.k-flow{background:var(--magenta)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:20px;padding-bottom:56px}
.card{position:relative;display:flex;flex-direction:column;gap:10px;padding:22px 22px 18px 30px;background:var(--paper);border:2px solid var(--brown-100);border-radius:var(--r);text-decoration:none;transition:border-color .15s,transform .15s;overflow:hidden}
.card::before{content:"";position:absolute;inset:0 auto 0 0;width:10px;background:var(--yellow)}.card.flow::before{background:var(--magenta)}
.card:hover{border-color:var(--ink);transform:translateY(-3px)}
.card-head{display:flex;justify-content:space-between;align-items:center}
.callno{font:500 13px var(--mono);color:var(--brown-500)}
.card h3{font-size:22px;font-weight:700}.card p{margin:0;color:var(--ink-2);font-size:15px;flex:1}
.badge{padding:3px 10px;border-radius:999px;font:700 12px var(--body);border:2px solid var(--ink)}.badge.crew{background:var(--yellow)}.badge.flow{background:var(--magenta);color:#fff}
.tags{list-style:none;display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 0;padding:0}
.tags li{font-size:12.5px;padding:2px 10px;border-radius:6px;background:var(--cream-2);color:var(--ink-2)}
.card-foot{display:flex;justify-content:space-between;gap:10px;margin-top:8px;padding-top:12px;border-top:1px dashed var(--brown-300);font-size:13px;color:var(--ink-2)}
.open{font-weight:700;color:var(--magenta);white-space:nowrap}.card:hover .open{text-decoration:underline}
.empty{text-align:center;padding:40px 0 70px;color:var(--ink-2)}
.page{padding-bottom:60px}
.back{display:inline-block;margin:14px 0 22px;font-weight:600;color:var(--magenta);text-decoration:none}.back::before{content:"\2190  "}.back:hover{text-decoration:underline}
.page-head h1{font-size:clamp(32px,4.6vw,50px);font-weight:800;margin:12px 0 10px}.page-head p{font-size:18px;color:var(--ink-2);max-width:62ch;margin:0}
.split{display:grid;grid-template-columns:300px 1fr;gap:36px;margin-top:34px;align-items:start}
.about{position:sticky;top:16px;display:grid;gap:26px}.about h2,.play h2{font-size:20px;margin-bottom:12px}
.steps{margin:0;padding-left:22px;display:grid;gap:8px;color:var(--ink-2);font-size:15px}.steps li::marker{color:var(--magenta);font-weight:700}
.team{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:6px}
.team li{padding:4px 12px;border-radius:999px;background:var(--yellow-soft);border:1.5px solid var(--brown-300);font-size:13.5px;font-weight:500}
.endpoint{display:block;padding:10px 14px;background:var(--ink);color:var(--cream);border-radius:10px;font:500 14px var(--mono);margin-bottom:6px}.endpoint b{color:var(--yellow);margin-right:6px}
.about small{color:var(--ink-2)}
.play{background:var(--paper);border:2px solid var(--ink);border-radius:18px;padding:26px;box-shadow:6px 6px 0 var(--brown-300);min-width:0}
.play-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px}.play-head h2{margin:0;font-size:24px}
.field{display:grid;gap:6px;margin-bottom:16px}.field label{font-weight:600;font-size:15px}
.field label em{font-style:normal;font-weight:500;font-size:12.5px;color:var(--magenta);margin-left:8px}
.field small,.rblock small{color:var(--ink-2);font-size:13px}
input[type=text],input[type=url],textarea{width:100%;padding:11px 14px;border-radius:10px;border:2px solid var(--brown-300);background:#fff;font:inherit;color:var(--ink)}
textarea{resize:vertical}textarea.mono{font:400 14px/1.5 var(--mono)}
input:focus,textarea:focus{outline:none;border-color:var(--magenta);box-shadow:0 0 0 4px var(--magenta-soft)}
input[type=file]{width:100%;padding:10px;border:2px dashed var(--brown-300);border-radius:10px;background:var(--cream);font:inherit;font-size:14px}
input[type=file]::file-selector-button{margin-right:12px;padding:6px 14px;border:2px solid var(--ink);border-radius:8px;background:var(--yellow);font:600 13px var(--body);color:var(--ink);cursor:pointer}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:6px}
.btn{padding:11px 22px;border-radius:12px;border:2px solid var(--ink);font:700 15px var(--body);cursor:pointer;color:var(--ink);background:var(--yellow)}
.btn:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 3px 0 var(--ink)}.btn:disabled{opacity:.55;cursor:not-allowed}
.btn-primary{background:var(--magenta);color:#fff}.btn-primary:hover:not(:disabled){background:var(--magenta-deep)}
.btn-ghost{background:transparent}.btn-ghost:hover:not(:disabled){background:var(--cream-2);box-shadow:none;transform:none}
.btn.sm{padding:5px 12px;font-size:13px;border-radius:9px}
.working{margin-top:22px;padding:16px;background:var(--yellow-soft);border-radius:12px;border:2px solid var(--brown-300)}
.working p{margin:10px 0 0;font-size:14px}.working b{font:700 15px var(--mono);margin-right:8px}
.bar{height:8px;border-radius:99px;background:#fff;overflow:hidden}.bar i{display:block;height:100%;width:35%;background:var(--magenta);border-radius:99px;animation:slide 1.4s ease-in-out infinite}
@keyframes slide{from{margin-left:-35%}to{margin-left:100%}}
.alert{margin:16px 0 0;padding:12px 16px;border-radius:10px;background:var(--magenta-soft);border:2px solid var(--magenta);color:#5c0a37;font-size:15px}
.result{margin-top:28px;padding-top:24px;border-top:2px dashed var(--brown-300)}
.result-top{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:14px;flex-wrap:wrap}.result-top h2{margin:0}.result-top small{color:var(--ink-2)}
.rmeta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.tag{padding:3px 12px;border-radius:999px;font:600 13px var(--body);background:var(--cream-2);border:1.5px solid var(--brown-300);text-transform:capitalize}
.tag.brand{background:var(--yellow);border-color:var(--ink)}.tag.hot{background:var(--magenta);color:#fff;border-color:var(--ink)}.tag.warn{background:var(--yellow-soft);border-color:var(--brown-500)}.tag.ok{background:#dcefd8;border-color:var(--ok)}
.why{margin:0 0 14px;color:var(--ink-2)}
.rblock{background:#fff;border:2px solid var(--brown-100);border-radius:12px;padding:18px 20px;margin-bottom:16px}
.rhead{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:10px}.rhead h3{font-size:18px}.rbtns{display:flex;gap:6px}
.md{overflow-wrap:anywhere;max-width:78ch}.md h1,.md h2,.md h3,.md h4{margin:22px 0 8px}.md h1{font-size:26px}.md h2{font-size:21px}.md h3,.md h4{font-size:17px}
.md>:first-child{margin-top:0}.md p,.md ul,.md ol{margin:0 0 12px}.md a{color:var(--magenta)}
.md code{font:13.5px var(--mono);background:var(--cream-2);padding:1px 6px;border-radius:5px}
.md pre{background:var(--ink);color:var(--cream);padding:14px;border-radius:10px;overflow-x:auto}.md pre code{background:none}
.md table{display:block;overflow-x:auto;border-collapse:collapse;margin:0 0 14px;font-size:14.5px}
.md th,.md td{border:1.5px solid var(--brown-300);padding:7px 12px;text-align:left;vertical-align:top}.md th{background:var(--yellow-soft)}
.md blockquote{margin:0 0 12px;padding-left:14px;border-left:4px solid var(--magenta);color:var(--ink-2)}
.md hr{border:0;border-top:2px dashed var(--brown-300);margin:18px 0}
.review textarea{margin:6px 0 4px}.review label{font-weight:600}
.tabs{display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap}
.tabs button{padding:7px 16px;border-radius:999px;border:2px solid var(--brown-300);background:transparent;font:600 14px var(--body);color:var(--ink);cursor:pointer}
.tabs button.on{background:var(--magenta);border-color:var(--magenta);color:#fff}
.panel[hidden]{display:none}
.lead{display:grid;grid-template-columns:auto 1fr auto;gap:18px;align-items:center;padding:16px 20px;background:#fff;border:2px solid var(--brown-100);border-radius:12px;margin-bottom:14px}
.lead small{display:block;color:var(--ink-2);font-size:13px}.lead b{font:800 38px var(--display)}.lead b small{display:inline;font-size:15px;font-weight:500}
.meter{display:block;height:10px;background:var(--cream-2);border-radius:99px;overflow:hidden;border:1.5px solid var(--brown-300)}
.meter i{display:block;height:100%;background:linear-gradient(90deg,var(--yellow),var(--magenta))}.meter.big{height:14px}
.cands{display:grid;gap:10px}.cand{border:2px solid var(--brown-100);border-radius:10px;background:var(--paper)}.cand[open]{border-color:var(--ink)}
.cand summary{display:grid;grid-template-columns:minmax(0,1fr) 90px auto;gap:14px;align-items:center;padding:12px 16px;cursor:pointer;list-style:none}.cand summary::-webkit-details-marker{display:none}
.cname{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cand summary b{font:800 18px var(--display)}.cand summary b small{font-size:12px;font-weight:500;color:var(--ink-2)}
.cbody{padding:0 16px 14px}.cbody h4{font-size:15px;margin:12px 0 4px}.cbody ul{margin:0;padding-left:20px}
.raw{margin-top:6px;font-size:14px}.raw summary{cursor:pointer;color:var(--ink-2)}
.raw pre{background:var(--ink);color:var(--cream);padding:14px;border-radius:10px;overflow:auto;max-height:340px;font:12.5px/1.5 var(--mono)}
.foot{max-width:1120px;margin:0 auto;padding:26px 24px 40px;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;font-size:13.5px;color:var(--ink-2);border-top:2px solid var(--brown-100)}
@media(max-width:880px){.hero{grid-template-columns:1fr;gap:36px;padding-top:28px}.shelf{justify-self:stretch;max-width:none}.split{grid-template-columns:1fr}.about{position:static}.topnav a{display:none}}
@media(max-width:520px){.wrap{padding:0 16px}.play{padding:18px;box-shadow:4px 4px 0 var(--brown-300)}.grid{grid-template-columns:1fr}.lead{grid-template-columns:1fr}.cand summary{grid-template-columns:minmax(0,1fr) auto}.cand .meter{display:none}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
</style>
</head>
<body>
<header class="topbar">
  <a class="brand" href="#/" aria-label="CrewAI Project Library home">
    <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden="true"><rect x="5" y="8" width="5" height="18" rx="1.5" fill="#F7B500"/><rect x="12" y="5" width="5" height="21" rx="1.5" fill="#C9167A"/><rect x="19" y="10" width="5" height="16" rx="1.5" fill="#B88B5E"/><rect x="3" y="26" width="26" height="2.5" rx="1" fill="#33241A"/></svg>
    <span>Crew Library</span>
  </a>
  <nav class="topnav">
    <span id="pill" class="pill"><i></i><span>Checking backend</span></span>
    <a href="/docs">API docs</a>
    <a href="https://github.com/Akhila636636/CrewAI_projects" target="_blank" rel="noreferrer">GitHub</a>
  </nav>
</header>
<main id="app"></main>
<footer class="foot"><span>Ten CrewAI projects on a FastAPI backend, powered by Gemini.</span><span>Outputs are AI-generated. Check facts and numbers before using them.</span></footer>

<script>
(function () {
"use strict";
var sampleCsv = "product,units,price\nNotebook,120,4.5\nPen Set,300,2.25\nDesk Lamp,45,19.99\nNotebook,80,4.5\nBackpack,60,34\nPen Set,150,2.25";
var env = function (o) { return { json: { inputs: o } }; };
var oneFile = function (v) { var f = new FormData(); f.append("inputs", "{}"); f.append("file", v.file[0]); return { form: f }; };

var AGENTS = [
 {id:1,type:"Crew",name:"Blog Post Crew",path:"/agent1",process:"Sequential",blurb:"Researches a topic on the web, drafts a 600-word post, then edits it for clarity and honesty.",tags:["Research","Writing","DuckDuckGo"],team:["Researcher","Writer","Editor"],steps:["Search DuckDuckGo for the topic","Researcher builds a brief with sources","Writer drafts about 600 words","Editor polishes and flags uncertain claims"],output:"md",
  fields:[{key:"topic",label:"Topic",type:"text",placeholder:"e.g. How small teams can adopt multi-agent AI",required:true}],example:{topic:"How small teams can adopt multi-agent AI"},build:function(v){return env({topic:v.topic});}},
 {id:2,type:"Crew",name:"Competitor Analysis Crew",path:"/agent2",process:"Hierarchical",blurb:"A manager delegates scraping, pricing and news research, then writes a comparison with a SWOT table.",tags:["Strategy","Scraping","SWOT"],team:["Manager","Site Scraper","Pricing Analyst","News Analyst","Strategist"],steps:["Fetch the three competitor pages","Search recent news for each domain","Specialists summarise pages, prices and news","Strategist writes the report and SWOT"],output:"md",
  fields:[{key:"c1",label:"Competitor 1",type:"url",placeholder:"https://",required:true},{key:"c2",label:"Competitor 2",type:"url",placeholder:"https://",required:true},{key:"c3",label:"Competitor 3",type:"url",placeholder:"https://",required:true}],
  example:{c1:"https://www.notion.so",c2:"https://www.coda.io",c3:"https://clickup.com"},build:function(v){return env({competitors:[v.c1,v.c2,v.c3].map(function(s){return s.trim();})});}},
 {id:3,type:"Crew",name:"Meeting Notes Crew",path:"/agent3",process:"Sequential, parallel tasks",blurb:"Turns a transcript into a five-line summary, an action table with owners, and a decision log.",tags:["Productivity","Async tasks","Transcripts"],team:["Summarizer","Action Finder","Decision Logger","Formatter"],steps:["Three agents read the transcript in parallel","Formatter merges summary, actions and decisions","Missing owners or dates are marked unspecified"],output:"md",
  fields:[{key:"transcript",label:"Transcript",type:"textarea",rows:7,placeholder:"Paste the meeting transcript, or upload a .txt file below."},{key:"file",label:"Or upload a transcript",type:"file",accept:".txt,text/plain",hint:"TXT, up to 5 MB. Overrides pasted text."}],
  example:{transcript:"Priya: We agreed to ship the beta on 14 Nov. Rahul will fix the login bug by Friday. Meera, can you draft the launch email? Meera: Yes, I'll send a draft next Tuesday. We decided to postpone the pricing page."},
  validate:function(v){return !v.transcript.trim()&&!v.file.length?"Paste a transcript or upload a .txt file.":null;},
  build:function(v){return v.file.length?oneFile(v):env({transcript:v.transcript});}},
 {id:4,type:"Crew",name:"Data Report Crew",path:"/agent4",process:"Sequential, custom tools",blurb:"Analyses a sales CSV with a pandas tool and writes a reviewed report with the top three findings.",tags:["Data","pandas","Custom tool"],team:["Data Analyst","Insight Writer","Reviewer"],steps:["Analyst reads the CSV and runs the pandas tool","Insight Writer extracts three findings","Reviewer checks every number and saves report.md"],output:"md",
  fields:[{key:"csv_content",label:"CSV content",type:"textarea",rows:7,mono:true,placeholder:"product,units,price\nNotebook,120,4.5",hint:"Needs product, units and price columns."},{key:"file",label:"Or upload a CSV",type:"file",accept:".csv,text/csv",hint:"CSV, up to 5 MB. Overrides pasted text."}],
  example:{csv_content:sampleCsv},validate:function(v){return !v.csv_content.trim()&&!v.file.length?"Paste CSV text or upload a .csv file.":null;},
  build:function(v){return v.file.length?oneFile(v):env({csv_content:v.csv_content});}},
 {id:5,type:"Flow",name:"Customer Support Flow",path:"/agent5",process:"Router + listeners",blurb:"Classifies a ticket, routes it to a billing, access or general handler, and drafts a safe reply.",tags:["Routing","Support","Router"],team:["Ticket Categorizer","Billing Handler","Access Handler","General Handler"],steps:["Receive and clean the ticket","Categorize and set priority","Router picks billing, account access or other","Specialist drafts the resolution"],output:"support",
  fields:[{key:"ticket",label:"Support ticket",type:"textarea",rows:5,placeholder:"I was charged twice for my subscription this month.",required:true}],example:{ticket:"I was charged twice for my subscription this month and I can't find the invoice."},build:function(v){return env({ticket:v.ticket});}},
 {id:6,type:"Flow",name:"Resume Screening Flow",path:"/agent6",process:"Score, then route",blurb:"Scores TXT or PDF resumes against a job description and drafts follow-ups for human review.",tags:["HR","PDF","Human in the loop"],team:["Screening Analyst","Follow-up Writer"],steps:["Extract text from each resume","Score 0 to 10 against the job","Scores of 7 and above get interview questions","Lower scores get a polite rejection draft for review"],output:"resumes",
  fields:[{key:"job_description",label:"Job description",type:"textarea",rows:6,placeholder:"Paste the role requirements...",required:true,hint:"Up to 15,000 characters."},{key:"resumes",label:"Resumes",type:"file",multiple:true,accept:".txt,.pdf",required:true,hint:"TXT or PDF, up to 50 files, 5 MB each."}],
  example:{job_description:"Junior ML engineer. Python, PyTorch, REST APIs, and one shipped ML project. Bonus: LangGraph or CrewAI."},
  build:function(v){var f=new FormData();f.append("inputs",JSON.stringify({job_description:v.job_description}));v.resumes.forEach(function(x){f.append("resumes",x);});return {form:f};}},
 {id:7,type:"Flow",name:"Bug Triage Flow",path:"/agent7",process:"Router + listeners",blurb:"Reads an issue, decides if it is a bug, feature or question, and drafts the triage and a reply.",tags:["Developer tools","Triage","Router"],team:["Issue Classifier","Bug Crew","Feature Crew","Answer Crew","Reply Writer"],steps:["Read title and body","Classify as bug, feature or question","Matching crew writes the triage","Reply writer drafts the issue comment"],output:"triage",
  fields:[{key:"title",label:"Issue title",type:"text",placeholder:"App crashes when uploading a PNG",required:true},{key:"body",label:"Issue body",type:"textarea",rows:5,placeholder:"Steps, expected and actual behaviour...",required:true}],
  example:{title:"App crashes when uploading a PNG",body:"On v2.3 on Android 14, tapping Upload with a PNG over 4 MB closes the app. Smaller files work."},build:function(v){return env({title:v.title,body:v.body});}},
 {id:8,type:"Flow",name:"Content Repurposing Flow",path:"/agent8",process:"Parallel branches, and_()",blurb:"Turns one article or URL into a LinkedIn post, a six-part X thread and a newsletter intro.",tags:["Marketing","Parallel","Writing"],team:["Article Analyst","LinkedIn Crew","Twitter Crew","Newsletter Crew"],steps:["Load the article text or fetch the URL","Extract thesis, evidence and takeaways","Three crews write in parallel","Publish pack once all three finish"],output:"content",
  fields:[{key:"article",label:"Article text or public URL",type:"textarea",rows:6,placeholder:"https://example.com/post or paste the full text",required:true}],example:{article:"https://en.wikipedia.org/wiki/Multi-agent_system"},build:function(v){return env({article:v.article});}},
 {id:9,type:"Flow",name:"Lead Qualification Flow",path:"/agent9",process:"Router + tools",blurb:"Scrapes a lead's website, scores fit out of 100, and drafts outreach, nurture or an internal note.",tags:["Sales","Scraping","Scoring"],team:["Enrich Crew","Scoring Analyst","Outreach Crew","Nurture Crew","Lead Recorder"],steps:["Scrape the public website","Score 0 to 100 as hot, warm or cold","Router picks outreach, nurture or archive","Record the lead to leads.csv"],output:"lead",
  fields:[{key:"name",label:"Lead name",type:"text",placeholder:"Acme Foods",required:true},{key:"website",label:"Website",type:"url",placeholder:"https://",required:true},{key:"business_need",label:"Business need",type:"text",placeholder:"Automate order follow-ups"},{key:"budget",label:"Budget",type:"text",placeholder:"e.g. Rs 2 lakh"},{key:"timeline",label:"Timeline",type:"text",placeholder:"e.g. This quarter"}],
  example:{name:"Acme Foods",website:"https://www.example.com",business_need:"Automate order follow-ups",budget:"Rs 2 lakh",timeline:"This quarter"},build:function(v){return env({name:v.name,website:v.website,business_need:v.business_need,budget:v.budget,timeline:v.timeline});}},
 {id:10,type:"Flow",name:"Approval Report Flow",path:"/agent10",process:"Human review loop",blurb:"Drafts a researched report, then waits for you to approve it or request up to three revisions.",tags:["Human in the loop","Reports","Review"],team:["Research Crew","Writing Crew","Revision Crew","Report Publisher"],steps:["Search for background evidence","Research and writing crews draft the report","You approve or ask for a revision","Approved drafts are saved as final.md"],output:"approval",
  fields:[{key:"topic",label:"Topic",type:"text",placeholder:"State of AI agents in Indian MSMEs",required:true},{key:"audience",label:"Audience",type:"text",placeholder:"e.g. Non-technical founders",required:true}],
  example:{topic:"State of AI agents in Indian MSMEs",audience:"Non-technical founders"},build:function(v){return env({topic:v.topic,audience:v.audience});}}
];

/* ---------- helpers ---------- */
var $app = document.getElementById("app");
function esc(s){return String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}
function pretty(s){return String(s||"").replace(/_/g," ");}
var STORE = [];
function inline(s){
  s = esc(s);
  s = s.replace(/`([^`]+)`/g,"<code>$1</code>").replace(/\*\*([^*]+)\*\*/g,"<strong>$1</strong>")
       .replace(/(^|[^*])\*([^*\s][^*]*)\*/g,"$1<em>$2</em>")
       .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,'<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s;
}
function md(src){
  var L = String(src==null?"":src).replace(/\r/g,"").split("\n"), out = "", i = 0, m;
  var isBlock = function(l){return /^(#{1,6}\s|```|\s*[-*+]\s+|\s*\d+[.)]\s+|>|\s*\|)/.test(l);};
  while (i < L.length) {
    var l = L[i];
    if (/^```/.test(l)) { var b=[]; i++; while(i<L.length && !/^```/.test(L[i])) b.push(L[i++]); i++; out += "<pre><code>"+esc(b.join("\n"))+"</code></pre>"; continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*)/))) { out += "<h"+m[1].length+">"+inline(m[2])+"</h"+m[1].length+">"; i++; continue; }
    if (/^\s*([-*_])\1{2,}\s*$/.test(l)) { out += "<hr>"; i++; continue; }
    if (/^\s*\|.*\|\s*$/.test(l) && i+1<L.length && /^\s*\|?\s*:?-{2,}/.test(L[i+1])) {
      var cells = function(r){return r.trim().replace(/^\||\|$/g,"").split("|").map(function(c){return inline(c.trim());});};
      out += "<table><thead><tr>"+cells(l).map(function(c){return "<th>"+c+"</th>";}).join("")+"</tr></thead><tbody>"; i += 2;
      while (i<L.length && /^\s*\|.*\|\s*$/.test(L[i])) { out += "<tr>"+cells(L[i]).map(function(c){return "<td>"+c+"</td>";}).join("")+"</tr>"; i++; }
      out += "</tbody></table>"; continue;
    }
    if (/^\s*[-*+]\s+/.test(l)) { out += "<ul>"; while(i<L.length && /^\s*[-*+]\s+/.test(L[i])) out += "<li>"+inline(L[i++].replace(/^\s*[-*+]\s+/,""))+"</li>"; out += "</ul>"; continue; }
    if (/^\s*\d+[.)]\s+/.test(l)) { out += "<ol>"; while(i<L.length && /^\s*\d+[.)]\s+/.test(L[i])) out += "<li>"+inline(L[i++].replace(/^\s*\d+[.)]\s+/,""))+"</li>"; out += "</ol>"; continue; }
    if (/^>/.test(l)) { var q=[]; while(i<L.length && /^>/.test(L[i])) q.push(L[i++].replace(/^>\s?/,"")); out += "<blockquote>"+inline(q.join(" "))+"</blockquote>"; continue; }
    if (!l.trim()) { i++; continue; }
    var p=[]; while(i<L.length && L[i].trim() && !isBlock(L[i])) p.push(L[i++]);
    if (!p.length) p.push(L[i++]);
    out += "<p>"+inline(p.join(" "))+"</p>";
  }
  return '<div class="md">'+out+"</div>";
}
async function api(path, o){
  o = o || {};
  var init = { method: o.method || "POST", signal: o.signal };
  if (o.form) init.body = o.form;
  else if (o.json !== undefined) { init.headers = {"Content-Type":"application/json"}; init.body = JSON.stringify(o.json); }
  var res;
  try { res = await fetch(path, init); } catch (e) { if (e.name==="AbortError") throw e; throw new Error("Could not reach the server. Check your connection."); }
  var text = await res.text(), body = null;
  try { body = JSON.parse(text); } catch (e) {}
  if (!res.ok || body === null) {
    var d = body && body.detail;
    var msg = typeof d==="string" ? d : Array.isArray(d) ? d.map(function(x){return x.msg||JSON.stringify(x);}).join("; ")
      : (res.status===502||res.status===504) ? "The server took too long or is still waking up. Wait a moment and run it again." : "Request failed ("+res.status+").";
    throw new Error(msg);
  }
  return body;
}

/* ---------- result views ---------- */
function acts(text, name, mime){
  STORE.push({text:String(text==null?"":text), name:name, mime:mime||"text/markdown"});
  return '<div class="rbtns"><button type="button" class="btn btn-ghost sm" data-copy="'+(STORE.length-1)+'">Copy</button><button type="button" class="btn btn-ghost sm" data-dl="'+(STORE.length-1)+'">Download</button></div>';
}
function block(title, text, name){return '<section class="rblock"><div class="rhead"><h3>'+esc(title)+"</h3>"+(name?acts(text,name):"")+"</div>"+md(text)+"</section>";}
function chip(t, tone){return '<span class="tag '+(tone||"")+'">'+esc(t)+"</span>";}
function list(x){return "<ul>"+[].concat(x).map(function(s){return "<li>"+esc(typeof s==="string"?s:JSON.stringify(s))+"</li>";}).join("")+"</ul>";}

function approvalView(s){
  var done = s.status==="approved", text = done ? s.output : s.draft;
  return '<div class="rmeta">'+chip(done?"Approved":"Awaiting your review",done?"ok":"warn")+(s.revision_number>0?chip("Revision "+s.revision_number+" of 3"):"")+"</div>"+
    block(done?"Final report":"Draft report", text, done?"final.md":"draft.md")+
    (done?"":'<section class="rblock review"><h3>Your decision</h3><label for="fb">Feedback <small>(required to request a revision)</small></label><textarea id="fb" rows="3" placeholder="e.g. Shorten the summary and add a risks section."></textarea><div id="rv-err"></div><div class="actions"><button type="button" class="btn btn-primary" data-review="approved">Approve report</button><button type="button" class="btn btn-ghost" data-review="revise">Request revision</button></div></section>');
}
function resultBody(a, d){
  var r = d.result, o = r && typeof r==="object" ? r : null;
  switch (a.output) {
    case "md": return block("Report", d.output, a.name.toLowerCase().replace(/\s+/g,"-")+".md");
    case "support": if (o) return '<div class="rmeta">'+chip(pretty(o.category),"brand")+chip(o.priority+" priority",o.priority==="high"?"hot":"")+chip(pretty(o.route))+"</div>"+(o.reason?'<p class="why"><b>Why:</b> '+esc(o.reason)+"</p>":"")+block("Suggested reply",o.resolution,"resolution.md"); break;
    case "resumes":
      var rows = (d.results||[]).slice().sort(function(x,y){return y.score-x.score;});
      return '<div class="rmeta">'+chip((d.resumes_processed!=null?d.resumes_processed:rows.length)+" resumes")+chip("Human review required","warn")+'</div><section class="rblock"><div class="rhead"><h3>Ranked candidates</h3>'+(d.csv?acts(d.csv,"results.csv","text/csv"):"")+'</div><div class="cands">'+
        rows.map(function(c){var sc=Number(c.score)||0;return '<details class="cand"><summary><span class="cname">'+esc(c.candidate)+'</span><span class="meter" aria-hidden="true"><i style="width:'+(sc*10)+'%"></i></span><b>'+sc.toFixed(1)+"<small>/10</small></b></summary><div class=\"cbody\">"+
          (c.decision?chip(pretty(c.decision),sc>=7?"ok":""):"")+md(c.rationale)+(c.strengths&&c.strengths.length?"<h4>Strengths</h4>"+list(c.strengths):"")+(c.gaps&&c.gaps.length?"<h4>Gaps</h4>"+list(c.gaps):"")+(c.follow_up?"<h4>Suggested follow-up</h4>"+md(c.follow_up):"")+"</div></details>";}).join("")+"</div></section>";
    case "triage": if (o) return '<div class="rmeta">'+chip(o.category,"brand")+"</div>"+block("Triage",o.triage,"triage.md")+block("Suggested issue comment",o.reply,"reply.md"); break;
    case "content": if (o) { var t=[["LinkedIn post",o.linkedin_post,"linkedin_post.md"],["X thread",o.x_thread,"x_thread.md"],["Newsletter",o.newsletter,"newsletter.md"]];
      return '<section class="rblock"><div class="tabs" role="tablist">'+t.map(function(x,k){return '<button type="button" role="tab" data-tab="'+k+'" aria-selected="'+(k===0)+'" class="'+(k===0?"on":"")+'">'+x[0]+"</button>";}).join("")+"</div>"+
        t.map(function(x,k){return '<div class="panel" data-panel="'+k+'"'+(k?" hidden":"")+'><div class="rhead"><h3>'+x[0]+"</h3>"+acts(x[1],x[2])+"</div>"+md(x[1])+"</div>";}).join("")+"</section>"; } break;
    case "lead": if (o) { var sc2=Math.max(0,Math.min(100,Number(o.score)||0));
      return '<div class="lead"><div><small>Lead score</small><b>'+sc2+"<small>/100</small></b></div>"+'<span class="meter big" aria-hidden="true"><i style="width:'+sc2+'%"></i></span>'+chip(o.category,o.category==="hot"?"hot":o.category==="warm"?"warn":"")+"</div>"+'<div class="rmeta">'+chip(pretty(o.recommended_action))+"</div>"+block("Recommended next step",o.next_step,"next-step.md"); } break;
    case "approval": return approvalView(d);
  }
  return block("Response", "```\n"+JSON.stringify(d,null,2)+"\n```");
}
function resultHtml(a, d){
  return '<div class="result"><div class="result-top"><h2>Result</h2>'+(d.note?"<small>"+esc(d.note)+"</small>":"")+'</div><div id="rbody">'+resultBody(a,d)+'</div><details class="raw"><summary>Raw response</summary><pre>'+esc(JSON.stringify(d,null,2))+"</pre></details></div>";
}

/* ---------- pages ---------- */
var callNo = function(a){return (a.type==="Crew"?"CR":"FL")+"-"+String(a.id).padStart(2,"0");};
var HEIGHTS = [190,218,176,204,226,182,210,194,222,178];
function cardHtml(a){
  var t = a.type.toLowerCase();
  return '<a class="card '+t+'" href="#/agent/'+a.id+'"><div class="card-head"><span class="callno">'+callNo(a)+'</span><span class="badge '+t+'">'+a.type+"</span></div><h3>"+esc(a.name)+"</h3><p>"+esc(a.blurb)+'</p><ul class="tags">'+a.tags.map(function(x){return "<li>"+esc(x)+"</li>";}).join("")+'</ul><div class="card-foot"><span>'+a.team.length+" agents · "+esc(a.process)+'</span><span class="open">Open project</span></div></a>';
}
function library(){
  var q = "", type = "All";
  $app.innerHTML = '<div class="wrap"><section class="hero"><div class="hero-copy"><h1>Ten CrewAI projects, ready to run.</h1><p>Browse the shelf, open a project, fill in its inputs and watch a crew or flow do the work. Crews hand tasks between agents. Flows route work with code.</p>'+
   '<div class="hero-tools"><label class="search"><span class="sr">Search projects</span><input id="q" type="text" placeholder="Search by name, tag or agent"></label><div class="chips" role="group" aria-label="Filter by type">'+
   ["All","Crew","Flow"].map(function(t){return '<button type="button" class="chip'+(t==="All"?" on":"")+'" data-type="'+t+'" aria-pressed="'+(t==="All")+'">'+(t==="All"?"All projects":t+"s")+"</button>";}).join("")+'</div></div></div>'+
   '<div class="shelf" aria-label="Shelf of projects"><div class="spines">'+AGENTS.map(function(a,i){return '<a class="spine '+a.type.toLowerCase()+'" href="#/agent/'+a.id+'" style="height:'+HEIGHTS[i]+'px" title="'+esc(a.name)+'"><b>'+String(a.id).padStart(2,"0")+"</b><span>"+esc(a.name.replace(/ (Crew|Flow)$/,""))+"</span></a>";}).join("")+
   '</div><div class="plank"></div><div class="legend"><i class="k-crew"></i> Crew <i class="k-flow"></i> Flow</div></div></section><section aria-live="polite"><div class="grid" id="grid"></div><div id="none"></div></section></div>';
  var grid = document.getElementById("grid"), none = document.getElementById("none");
  function draw(){
    var s = q.trim().toLowerCase();
    var shown = AGENTS.filter(function(a){return (type==="All"||a.type===type)&&(!s||[a.name,a.blurb,a.process].concat(a.tags,a.team).join(" ").toLowerCase().indexOf(s)>-1);});
    grid.innerHTML = shown.map(cardHtml).join("");
    none.innerHTML = shown.length ? "" : '<div class="empty"><p>No project matches “'+esc(q)+'”.</p><button type="button" class="btn btn-ghost" id="clr">Clear search</button></div>';
    var c = document.getElementById("clr"); if (c) c.onclick = function(){q="";type="All";document.getElementById("q").value="";syncChips();draw();};
  }
  function syncChips(){document.querySelectorAll(".chip").forEach(function(b){var on=b.dataset.type===type;b.classList.toggle("on",on);b.setAttribute("aria-pressed",on);});}
  document.getElementById("q").oninput = function(e){q=e.target.value;draw();};
  document.querySelectorAll(".chip").forEach(function(b){b.onclick=function(){type=b.dataset.type;syncChips();draw();};});
  draw();
}
function fieldHtml(f){
  var id = "f-"+f.key, lab = '<label for="'+id+'">'+esc(f.label)+(f.required?"<em> required</em>":"")+"</label>", ctl;
  if (f.type==="textarea") ctl = '<textarea id="'+id+'" name="'+f.key+'" rows="'+(f.rows||5)+'" class="'+(f.mono?"mono":"")+'" placeholder="'+esc(f.placeholder||"")+'"></textarea>';
  else if (f.type==="file") ctl = '<input id="'+id+'" name="'+f.key+'" type="file" accept="'+esc(f.accept||"")+'"'+(f.multiple?" multiple":"")+">";
  else ctl = '<input id="'+id+'" name="'+f.key+'" type="'+(f.type==="url"?"url":"text")+'" placeholder="'+esc(f.placeholder||"")+'">';
  return '<div class="field">'+lab+ctl+(f.hint?"<small>"+esc(f.hint)+"</small>":"")+"</div>";
}
function agentPage(a){
  var t = a.type.toLowerCase(), ctrl = null, timer = null, current = null;
  $app.innerHTML = '<div class="wrap page"><a class="back" href="#/">Back to library</a><header class="page-head"><span class="badge '+t+'">'+a.type+"</span><h1>"+esc(a.name)+"</h1><p>"+esc(a.blurb)+"</p></header>"+
   '<div class="split"><aside class="about"><section><h2>How it runs</h2><ol class="steps">'+a.steps.map(function(s){return "<li>"+esc(s)+"</li>";}).join("")+'</ol></section><section><h2>The team</h2><ul class="team">'+a.team.map(function(s){return "<li>"+esc(s)+"</li>";}).join("")+
   '</ul></section><section><h2>Endpoint</h2><code class="endpoint"><b>POST</b> '+a.path+"</code><small>"+esc(a.process)+'</small></section></aside>'+
   '<section class="play" aria-labelledby="play-h"><div class="play-head"><h2 id="play-h">Try it</h2><button type="button" class="btn btn-ghost sm" id="ex">Fill an example</button></div><form id="f" novalidate>'+a.fields.map(fieldHtml).join("")+
   '<div class="actions"><button class="btn btn-primary" id="run">Run project</button><button type="button" class="btn btn-ghost" id="cancel" hidden>Cancel</button></div></form><div id="status" role="status"></div><div id="out"></div></section></div></div>';
  var form = document.getElementById("f"), status = document.getElementById("status"), out = document.getElementById("out"), run = document.getElementById("run"), cancel = document.getElementById("cancel");
  function stop(){clearInterval(timer);run.disabled=false;run.textContent="Run project";cancel.hidden=true;}
  document.getElementById("ex").onclick = function(){ Object.keys(a.example).forEach(function(k){form.elements[k].value=a.example[k];}); status.innerHTML=""; };
  cancel.onclick = function(){ if (ctrl) ctrl.abort(); stop(); status.innerHTML=""; };
  form.onsubmit = async function(e){
    e.preventDefault();
    var v = {};
    a.fields.forEach(function(f){var el=form.elements[f.key]; v[f.key]=f.type==="file"?Array.prototype.slice.call(el.files):el.value;});
    var err = null;
    a.fields.forEach(function(f){ if(!err && f.required && (f.type==="file"?!v[f.key].length:!String(v[f.key]).trim())) err = f.label+" is required."; });
    if (!err && a.validate) err = a.validate(v);
    if (err) { status.innerHTML = '<div class="alert" role="alert"><b>Could not run.</b> '+esc(err)+"</div>"; return; }
    if (ctrl) ctrl.abort(); ctrl = new AbortController();
    out.innerHTML = ""; run.disabled = true; run.textContent = "Running…"; cancel.hidden = false;
    var secs = 0;
    var tick = function(){status.innerHTML='<div class="working"><div class="bar"><i></i></div><p><b>'+secs+"s</b> The crew is working. Agents make several model calls, so this can take a minute or two."+(secs>20?" If the server was idle it may still be waking up.":"")+"</p></div>";};
    tick(); timer = setInterval(function(){secs++;tick();},1000);
    try {
      var req = a.build(v);
      current = await api(a.path, {json:req.json, form:req.form, signal:ctrl.signal});
      stop(); status.innerHTML = ""; out.innerHTML = resultHtml(a, current);
    } catch (ex) {
      if (ex.name==="AbortError") return;
      stop(); status.innerHTML = '<div class="alert" role="alert"><b>Could not run.</b> '+esc(ex.message)+"</div>";
    }
  };
  /* result interactions (copy, download, tabs, approval) */
  out.onclick = async function(e){
    var b = e.target.closest("button"); if (!b) return;
    if (b.dataset.copy != null) { try { await navigator.clipboard.writeText(STORE[b.dataset.copy].text); var o=b.textContent; b.textContent="Copied"; setTimeout(function(){b.textContent=o;},1500); } catch (x) {} }
    else if (b.dataset.dl != null) { var s=STORE[b.dataset.dl], u=URL.createObjectURL(new Blob([s.text],{type:s.mime})), l=document.createElement("a"); l.href=u; l.download=s.name; l.click(); URL.revokeObjectURL(u); }
    else if (b.dataset.tab != null) { out.querySelectorAll("[data-tab]").forEach(function(x){var on=x===b;x.classList.toggle("on",on);x.setAttribute("aria-selected",on);}); out.querySelectorAll("[data-panel]").forEach(function(p){p.hidden=p.dataset.panel!==b.dataset.tab;}); }
    else if (b.dataset.review) {
      var decision = b.dataset.review, fb = document.getElementById("fb").value, re = document.getElementById("rv-err");
      if (decision==="revise" && !fb.trim()) { re.innerHTML='<div class="alert" role="alert">Add feedback so the crew knows what to change.</div>'; return; }
      re.innerHTML = ""; out.querySelectorAll("[data-review]").forEach(function(x){x.disabled=true;}); b.textContent = "Working…";
      try {
        var r = await api("/agent10/review", {json:{inputs:{review_id:current.review_id,review_decision:decision,feedback:fb}}});
        current = Object.assign({}, r, {review_id: current.review_id});
        document.getElementById("rbody").innerHTML = approvalView(current);
      } catch (ex) {
        out.querySelectorAll("[data-review]").forEach(function(x){x.disabled=false;}); b.textContent = decision==="approved"?"Approve report":"Request revision";
        re.innerHTML = '<div class="alert" role="alert">'+esc(ex.message)+"</div>";
      }
    }
  };
}

function route(){
  var m = location.hash.match(/^#\/agent\/(\d+)/), a = m && AGENTS.filter(function(x){return x.id===Number(m[1]);})[0];
  STORE = []; window.scrollTo(0,0);
  if (a) { document.title = a.name+" · CrewAI Project Library"; agentPage(a); }
  else { document.title = "CrewAI Project Library"; library(); }
}
window.addEventListener("hashchange", route);
route();

api("/health",{method:"GET"}).then(function(h){
  var p=document.getElementById("pill"); p.className="pill up"; p.lastChild.textContent="Backend online"; p.title=h.model||"";
}).catch(function(){var p=document.getElementById("pill"); p.className="pill down"; p.lastChild.textContent="Backend unreachable";});
})();
</script>
</body>
</html>
'''


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return UI_HTML


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
    # Validate request format.
    content_type = request.headers.get("content-type", "").lower()

    if "multipart/form-data" not in content_type:
        raise HTTPException(
            status_code=415,
            detail=(
                'Use form-data with an "inputs" text field '
                'and one or more "resumes" file fields.'
            ),
        )

    form = await request.form()

    # Parse the common inputs envelope.
    raw_inputs = form.get("inputs")

    try:
        if isinstance(raw_inputs, str):
            inputs = json.loads(raw_inputs)
        elif isinstance(raw_inputs, dict):
            inputs = raw_inputs
        else:
            raise ValueError("Missing inputs")

        if not isinstance(inputs, dict):
            raise ValueError("inputs must be a JSON object")

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail='The "inputs" field must contain valid JSON.',
        ) from exc

    job_description = str(inputs.get("job_description", "")).strip()
    uploads = form.getlist("resumes")

    if not job_description:
        raise HTTPException(
            status_code=422,
            detail='Add "job_description" inside the "inputs" JSON.',
        )

    if len(job_description) > 15000:
        raise HTTPException(
            status_code=422,
            detail="Job description must not exceed 15,000 characters.",
        )

    if not uploads:
        raise HTTPException(
            status_code=422,
            detail='Upload at least one file using the "resumes" key.',
        )

    if len(uploads) > 50:
        raise HTTPException(
            status_code=422,
            detail="Upload no more than 50 resumes per request.",
        )

    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured on Render.",
        )

    # Extract resume text before invoking CrewAI.
    resumes = []

    for upload in uploads:
        filename = Path(
            getattr(upload, "filename", "") or ""
        ).name

        suffix = Path(filename).suffix.lower()

        if suffix not in {".txt", ".pdf"}:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file: {filename}. Upload TXT or PDF.",
            )

        raw = await upload.read(MAX_UPLOAD_BYTES + 1)

        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} exceeds the 5 MB limit.",
            )

        try:
            if suffix == ".txt":
                resume_text = raw.decode(
                    "utf-8-sig", errors="replace"
                ).strip()
            else:
                pdf = PdfReader(BytesIO(raw))

                if pdf.is_encrypted:
                    raise ValueError("Encrypted PDF")

                resume_text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                ).strip()

        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not extract text from {filename}.",
            ) from exc

        if not resume_text:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"No readable text in {filename}. "
                    "Scanned PDFs require OCR."
                ),
            )

        resumes.append({
            "filename": filename,
            "text": resume_text[:30000],
        })

    # Run CrewAI outside the async request loop.
    def run_screening():
        llm = LLM(
            model=MODEL_NAME,
            api_key=os.environ["GEMINI_API_KEY"],
            temperature=0.1,
        )

        results = []

        scorer = Agent(
            role="Resume Screening Analyst",
            goal=(
                "Evaluate each resume against the job description "
                "using only evidence in the resume."
            ),
            backstory=(
                "You assess job-related skills and qualifications. "
                "Never infer protected characteristics. Scores are "
                "advisory and must be reviewed by a human."
            ),
            llm=llm,
            verbose=False,
        )

        for resume in resumes:
            task = Task(
                description="""
Evaluate the resume against the job description.

JOB DESCRIPTION:
{job_description}

CANDIDATE RESUME:
{resume_text}

Return ONLY valid JSON in this format:
{
  "score": 0,
  "rationale": "Evidence-based explanation",
  "strengths": ["relevant demonstrated skill"],
  "gaps": ["missing or unverified job requirement"]
}

Rules:
- Score from 0 to 10 based on job-related evidence.
- Do not invent qualifications.
- Treat unmentioned skills as unverified, not proof of inability.
- Do not use protected characteristics in scoring.
- The result is decision support, not a final hiring decision.
""",
                expected_output=(
                    "JSON containing score, rationale, strengths, and gaps."
                ),
                agent=scorer,
            )

            raw_result = str(
                Crew(
                    agents=[scorer],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=False,
                ).kickoff(
                    inputs={
                        "job_description": job_description,
                        "resume_text": resume["text"],
                    }
                )
            )

            # Parse JSON, including JSON wrapped in a Markdown code block.
            match = re.search(r"\{[\s\S]*\}", raw_result)

            if not match:
                raise ValueError(
                    f"Scoring response for {resume['filename']} "
                    "did not contain valid JSON."
                )

            data = json.loads(match.group(0))
            score = float(data["score"])

            if not 0 <= score <= 10:
                raise ValueError("The resume score must be between 0 and 10.")

            strengths = data.get("strengths", [])
            gaps = data.get("gaps", [])

            if not isinstance(strengths, list):
                strengths = [str(strengths)]

            if not isinstance(gaps, list):
                gaps = [str(gaps)]

            results.append({
                "candidate": resume["filename"],
                "score": round(score, 2),
                "rationale": str(data.get("rationale", "")),
                "strengths": strengths,
                "gaps": gaps,
                "decision": "pending_human_review",
                "follow_up": "",
            })

        # Draft follow-up for recruiter review.
        follow_up_agent = Agent(
            role="Recruiter Assistant",
            goal=(
                "Prepare a recruiter-review summary for candidates "
                "scoring 7 or above, or a polite rejection email draft "
                "for candidates scoring below 7."
            ),
            backstory=(
                "You assist recruiters. Never send emails or make "
                "final hiring decisions."
            ),
            llm=llm,
            verbose=False,
        )

        for item in results:
            if item["score"] >= 7:
                item["decision"] = "shortlist_for_human_review"

                instruction = (
                    "Prepare a concise recruiter-review summary with "
                    "job-related strengths, gaps, and three interview "
                    "questions. Do not make a final hiring decision."
                )
            else:
                item["decision"] = "below_threshold_for_human_review"

                instruction = (
                    "Draft a polite rejection email for recruiter review. "
                    "Do not mention an automated score, send the email, "
                    "or imply that a final hiring decision was made."
                )

            task = Task(
                description=(
                    "{instruction}\n\n"
                    "Candidate: {candidate}\n"
                    "Evaluation: {evaluation}"
                ),
                expected_output="A concise draft for human recruiter review.",
                agent=follow_up_agent,
            )

            item["follow_up"] = str(
                Crew(
                    agents=[follow_up_agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=False,
                ).kickoff(
                    inputs={
                        "instruction": instruction,
                        "candidate": item["candidate"],
                        "evaluation": json.dumps(item),
                    }
                )
            )

        # Save the report.
        output = io.StringIO()

        fields = [
            "candidate",
            "score",
            "decision",
            "rationale",
            "strengths",
            "gaps",
            "follow_up",
        ]

        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()

        for item in results:
            writer.writerow({
                key: (
                    json.dumps(item[key], ensure_ascii=False)
                    if isinstance(item[key], (list, dict))
                    else item[key]
                )
                for key in fields
            })

        (OUTPUT_DIR / "results.csv").write_text(
            output.getvalue(),
            encoding="utf-8",
        )

        return results, output.getvalue()

    try:
        results, csv_content = await asyncio.to_thread(run_screening)

        return {
            "agent": "Resume Screening Flow",
            "status": "success",
            "resumes_processed": len(results),
            "results": results,
            "csv": csv_content,
            "file": "outputs/results.csv",
            "note": "All screening results require human review.",
        }

    except Exception as exc:
        logger.exception("Agent 6 failed during resume screening")

        raise HTTPException(
            status_code=502,
            detail=(
                f"Resume Screening Flow failed: "
                f"{type(exc).__name__}: {str(exc)}"
            ),
        ) from exc
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
