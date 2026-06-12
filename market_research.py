"""
Autonomous Market Research System
----------------------------------
Requires: pip install anthropic rich
API key:  export ANTHROPIC_API_KEY=sk-ant-...
Usage:    python market_research.py
"""

import os
import json
import re
import textwrap
from dataclasses import dataclass, field
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "gemini-2.5-flash"
MAX_TOKENS = 2000

Depth  = Literal["concise", "standard", "deep"]
Format = Literal["narrative", "bullets", "structured"]

DEPTH_INSTRUCTIONS: dict[str, str] = {
    "concise":  "Keep the report concise — 3–4 short sections, ~300 words total.",
    "standard": "Produce a balanced report — 5–6 sections, ~600 words total.",
    "deep":     "Produce a comprehensive deep-dive — 7–8 detailed sections, ~1000 words total.",
}

FORMAT_INSTRUCTIONS: dict[str, str] = {
    "narrative":  "Write in flowing narrative prose with proper paragraphs.",
    "bullets":    "Use bullet points and short paragraphs for each section.",
    "structured": "Return a JSON object with keys: title, summary, sections (array of {heading, bullets}), metrics, opportunities, risks.",
}

FOCUS_OPTIONS = [
    "market size & growth",
    "competitive landscape",
    "key trends",
    "customer segments",
    "pricing models",
    "risks & barriers",
]

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Metric:
    label: str
    value: str
    note: str = ""

@dataclass
class Insight:
    type: Literal["opportunity", "risk", "trend"]
    text: str

@dataclass
class Source:
    title: str
    url: str

@dataclass
class ResearchResult:
    report_text: str
    metrics: list[Metric] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)

# ── Core API call ─────────────────────────────────────────────────────────────

def build_system_prompt(focus: str, depth: Depth, fmt: Format) -> str:
    return textwrap.dedent(f"""
        You are an expert market research analyst with access to real-time web search.
        Produce a professional market research report on the given topic.

        Focus areas: {focus or "general market overview"}
        Depth: {DEPTH_INSTRUCTIONS[depth]}
        Format: {FORMAT_INSTRUCTIONS[fmt]}

        After the report output a JSON block on a new line, wrapped exactly like:
        |||JSON_START|||{{...}}|||JSON_END|||

        The JSON must contain:
        {{
          "metrics":  [{{"label":"...","value":"...","note":"..."}}],
          "insights": [{{"type":"opportunity|risk|trend","text":"..."}}],
          "sources":  [{{"title":"...","url":"..."}}]
        }}
        Include 2–4 metrics (market size, CAGR, etc.), 4–6 key insights,
        and 3–5 source references from your web search results.
    """).strip()


def run_research(
    topic: str,
    focus: list[str],
    depth: Depth = "standard",
    fmt: Format = "bullets",
) -> ResearchResult:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=build_system_prompt(", ".join(focus), depth, fmt),
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": f"Research topic: {topic}"}],
    )

    full_text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    # Split report from metadata JSON
    json_match = re.search(
        r"\|\|\|JSON_START\|\|\|(.*?)\|\|\|JSON_END\|\|\|", full_text, re.DOTALL
    )
    report_text = re.sub(
        r"\|\|\|JSON_START\|\|\|.*?\|\|\|JSON_END\|\|\|", "", full_text, flags=re.DOTALL
    ).strip()

    metrics, insights, sources = [], [], []
    if json_match:
        try:
            meta = json.loads(json_match.group(1).strip())
            metrics  = [Metric(**m)  for m in meta.get("metrics", [])]
            insights = [Insight(**i) for i in meta.get("insights", [])]
            sources  = [Source(**s)  for s in meta.get("sources", [])]
        except (json.JSONDecodeError, TypeError):
            pass

    return ResearchResult(report_text, metrics, insights, sources)

# ── Rich display helpers ──────────────────────────────────────────────────────

console = Console()

INSIGHT_COLORS = {"opportunity": "green", "risk": "red", "trend": "blue"}


def display_report(result: ResearchResult, topic: str) -> None:
    console.print()

    # Metrics table
    if result.metrics:
        t = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
        t.add_column("Metric",  style="dim", min_width=22)
        t.add_column("Value",   style="bold white", min_width=14)
        t.add_column("Note",    style="dim")
        for m in result.metrics:
            t.add_row(m.label, m.value, m.note)
        console.print(Panel(t, title=f"[bold]📊 {topic}[/bold]", border_style="white"))

    # Report body
    console.print(
        Panel(
            result.report_text,
            title="[bold]📄 Research Report[/bold]",
            border_style="white",
            padding=(1, 2),
        )
    )

    # Insights
    if result.insights:
        console.print("\n[bold]💡 Key Insights[/bold]")
        for ins in result.insights:
            color = INSIGHT_COLORS.get(ins.type, "white")
            label = f"[{color}][{ins.type.upper()}][/{color}]"
            console.print(f"  {label} {ins.text}")

    # Sources
    if result.sources:
        console.print("\n[bold]🔗 Sources[/bold]")
        for i, s in enumerate(result.sources, 1):
            console.print(f"  [dim]{i}.[/dim] {s.title}")
            console.print(f"     [dim]{s.url}[/dim]")


def save_result(result: ResearchResult, topic: str) -> None:
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:40]
    filename = f"research_{slug}.json"
    data = {
        "topic": topic,
        "report": result.report_text,
        "metrics":  [vars(m) for m in result.metrics],
        "insights": [vars(i) for i in result.insights],
        "sources":  [vars(s) for s in result.sources],
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"\n[dim]Saved to {filename}[/dim]")

# ── Interactive CLI ───────────────────────────────────────────────────────────

def pick_focus() -> list[str]:
    console.print("\n[bold]Focus areas[/bold] (comma-separated numbers, or Enter for defaults 1–3):")
    for i, opt in enumerate(FOCUS_OPTIONS, 1):
        console.print(f"  [dim]{i}.[/dim] {opt}")
    raw = Prompt.ask("Selection", default="1,2,3")
    chosen = []
    for part in raw.split(","):
        try:
            idx = int(part.strip()) - 1
            if 0 <= idx < len(FOCUS_OPTIONS):
                chosen.append(FOCUS_OPTIONS[idx])
        except ValueError:
            pass
    return chosen or FOCUS_OPTIONS[:3]


def pick_option(prompt: str, options: list[str], default: str) -> str:
    opts_display = " / ".join(
        f"[bold]{o}[/bold]" if o == default else o for o in options
    )
    console.print(f"\n[bold]{prompt}[/bold] ({opts_display})")
    val = Prompt.ask("", default=default)
    return val if val in options else default


def main() -> None:
    console.rule("[bold]Autonomous Market Research System[/bold]")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]Error:[/red] set ANTHROPIC_API_KEY environment variable")
        return

    topic = Prompt.ask("\n[bold]Research topic[/bold]")
    focus = pick_focus()
    depth = pick_option("Depth", ["concise", "standard", "deep"], "standard")
    fmt   = pick_option("Format", ["narrative", "bullets", "structured"], "bullets")

    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Searching the web and generating report…", total=None)
        result = run_research(topic, focus, depth=depth, fmt=fmt)  # type: ignore
        progress.update(task, completed=True)

    display_report(result, topic)

    if Confirm.ask("\nSave results to JSON?", default=False):
        save_result(result, topic)


if __name__ == "__main__":
    main()
