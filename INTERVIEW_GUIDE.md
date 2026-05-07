# Interview Guide — BlogForge AI (Blog Automation Project)

---

## 1. One-Line Pitch

> "I built an AI-powered blog generation pipeline using LangGraph and Google Gemini that takes a topic, decides whether it needs live web research, plans a structured outline, writes every section in parallel, and assembles a publication-ready Markdown blog — all orchestrated as a stateful multi-agent graph."

---

## 2. Tech Stack (say this upfront)

| Layer | Technology |
|---|---|
| Agent orchestration | **LangGraph** (stateful graph of nodes) |
| LLM | **Google Gemini** via `langchain-google-genai` |
| Structured output | **Pydantic v2** models with `.with_structured_output()` |
| Web research | **Tavily Search API** |
| Image generation | **Google GenAI SDK** (`gemini-2.5-flash-image`) |
| Frontend | **Streamlit** (wide layout, animated pipeline UI) |
| Language | **Python 3.14** |

---

## 3. High-Level Architecture

```
User Input (topic + date)
        │
        ▼
   ┌─────────┐
   │  Router  │  ← classifies topic, decides if research needed
   └────┬────┘
        │ conditional edge
   ┌────▼────┐         ┌──────────────┐
   │Research │ ──────► │ Orchestrator │  ← generates structured plan
   └─────────┘         └──────┬───────┘
   (optional)                 │ fanout → parallel Send()
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
                [Worker]  [Worker]  [Worker]   ← one per section, run in parallel
                    └─────────┼─────────┘
                              │ sections reduce via operator.add
                              ▼
                    ┌──────────────────┐
                    │  Reducer Subgraph │
                    │  ┌─────────────┐ │
                    │  │merge_content│ │  ← sorts & joins sections
                    │  └──────┬──────┘ │
                    │  ┌──────▼──────┐ │
                    │  │decide_images│ │  ← plans image placeholders
                    │  └──────┬──────┘ │
                    │  ┌──────▼──────┐ │
                    │  │generate &   │ │  ← calls image model, writes file
                    │  │place images │ │
                    │  └─────────────┘ │
                    └──────────────────┘
                              │
                              ▼
                    final Markdown blog
```

---

## 4. What Each Node Does

### Node 1 — `router`

**What it does:**
Classifies the topic into one of three modes using a structured LLM call.

**Three modes:**
- `closed_book` — evergreen topic (e.g., "how recursion works"). No research needed. Uses only the LLM's training knowledge.
- `hybrid` — needs both LLM knowledge and some fresh data (e.g., "best Python frameworks in 2025"). Searches for recent examples/benchmarks.
- `open_book` — highly time-sensitive (e.g., "AI news this week", "latest model pricing"). Requires live web search; LLM is not trusted to invent facts.

**Output to state:** `mode`, `needs_research`, `queries` (3–10 search queries if research needed), `recency_days` (7 / 45 / 3650).

**Key design point:** Uses `.with_structured_output(RouterDecision)` — the LLM returns a validated Pydantic object, not free text. This prevents hallucinated output formats.

---

### Conditional Edge — `route_next`

A plain Python function (not an LLM call) that reads `state["needs_research"]` and returns either `"research"` or `"orchestrator"`. This is LangGraph's **conditional routing** — different graph paths based on state.

---

### Node 2 — `research` (optional)

**What it does:**
Runs the router's generated search queries through Tavily Search API in a loop, then asks the LLM to synthesize the raw results into clean `EvidenceItem` objects (title, URL, snippet, publication date).

**Two-step process:**
1. **Retrieval** — calls Tavily for each query, collects raw results
2. **Synthesis** — passes raw JSON to LLM with `.with_structured_output(EvidencePack)` to deduplicate, normalize dates, and filter noise

**Recency filter:** For `open_book` mode, evidence items without a publication date or older than `recency_days` are dropped — the LLM cannot cite stale sources for a "this week" article.

**Graceful skip:** If no Tavily key is configured, returns empty evidence and continues — the graph never crashes.

---

### Node 3 — `orchestrator`

**What it does:**
Acts as a senior technical writer planning the blog's structure. Takes the topic, mode, and evidence, and produces a full `Plan` object — a structured outline with 5–9 `Task` objects.

**Each Task contains:**
- `title` — section heading
- `goal` — one sentence on what the reader learns
- `bullets` — 3–6 content points to cover
- `target_words` — word count target (120–550)
- Flags: `requires_research`, `requires_citations`, `requires_code`

**Design point:** The Plan is a Pydantic model returned via structured output, not a prompt string. This means every downstream worker gets a machine-readable spec, not an ambiguous instruction.

---

### Fanout — `fanout` function + `Send()`

**What it does:**
This is the **parallel execution mechanism**. Instead of writing sections one by one, the fanout function creates one `Send("worker", payload)` message per task. LangGraph executes all of them concurrently.

**Why this matters:** A 7-section blog that would take 7 sequential LLM calls now runs all 7 simultaneously. This is the biggest performance win in the pipeline.

**State reduction:** Sections write to `state["sections"]` which is typed as `Annotated[list[tuple[int, str]], operator.add]`. The `operator.add` reducer means each worker's output is *appended* to the list — LangGraph handles merging the concurrent results automatically.

---

### Node 4 — `worker` (runs N times in parallel)

**What it does:**
Writes a single Markdown section. Each worker gets its own `Task` spec, the full `Plan` for context, and the collected `Evidence`.

**Grounding rules enforced by the system prompt:**
- `open_book` mode: every factual claim must link to a provided Evidence URL. If not in evidence, write "Not found in provided sources." — no hallucination allowed.
- `hybrid` mode: cite evidence URLs for external claims.
- `requires_code=True`: must include at least one code snippet.

**Output:** `(task_id, section_markdown)` tuple, which the reducer sorts by `task_id` to restore order after parallel execution.

---

### Reducer Subgraph — 3 nodes compiled as a nested graph

The reducer is a **LangGraph subgraph** — a self-contained mini-graph compiled separately and then embedded as a single node in the main graph. This keeps the main graph clean while the reducer has its own internal sequence.

#### Sub-node A — `merge_content`

Sorts all `(id, markdown)` pairs by task ID (restoring the intended order after parallel execution), joins them with double newlines, prepends the blog title as `# H1`, and writes to `state["merged_md"]`.

#### Sub-node B — `decide_images`

If `generate_images` is `False` (default for free tier users), immediately passes `merged_md` through unchanged with no image specs — zero extra API calls.

If enabled, asks the LLM to scan the full Markdown and decide where up to 3 technical diagrams would genuinely improve understanding. It inserts `[[IMAGE_1]]` placeholders at those positions and returns a spec for each image (filename, alt text, caption, generation prompt, size, quality).

**Design point:** Only technical/explanatory images — the prompt explicitly bans decorative images.

#### Sub-node C — `generate_and_place_images`

Iterates over image specs, calls the Google GenAI SDK's `generate_content` with `response_modalities=["IMAGE"]` to get raw bytes, writes them to `images/`, then replaces each `[[IMAGE_N]]` placeholder with a proper Markdown image tag. If generation fails (quota, safety filter), the placeholder is silently removed — the blog remains clean.

---

## 5. State Design

The shared `State` TypedDict is the backbone. Every node reads from and writes to it. Key decisions:

```python
sections: Annotated[list[tuple[int, str]], operator.add]
```
The `Annotated` + `operator.add` pattern tells LangGraph to *merge* parallel worker outputs by concatenation rather than overwriting. Without this, parallel workers would race and clobber each other's output.

All other fields are simple overwrites — each node owns its output key and no two nodes write to the same key except `sections`.

---

## 6. Key Design Decisions to Highlight

**"Why LangGraph instead of a simple chain?"**
> A chain runs sequentially and can't branch or fan out. LangGraph gives me conditional routing (skip research for evergreen topics), parallel execution (write all sections simultaneously), and a subgraph for the reducer — none of which are natural in a chain.

**"Why structured output with Pydantic everywhere?"**
> Free-text LLM output is brittle. By defining `RouterDecision`, `Plan`, `Task`, `EvidencePack` as Pydantic models and using `.with_structured_output()`, the graph fails fast on schema violations rather than propagating garbage downstream. It also makes every node's contract explicit and testable.

**"How do you prevent hallucination in open_book mode?"**
> The worker's system prompt forbids introducing any specific claim not backed by a provided Evidence URL. It must either cite the URL inline or write "Not found in provided sources." The evidence list itself was filtered by publication date in the research node — so only genuinely recent sources reach the writer.

**"Why a subgraph for the reducer?"**
> The reducer has its own 3-step sequence (merge → decide → generate). Wrapping it in a subgraph keeps the main graph's topology readable — from the outside it's just one `reducer` node. It also means the reducer could be reused or independently tested.

**"How do you handle parallelism safely?"**
> LangGraph's `Send()` API dispatches each worker with its own isolated payload copy. Workers don't share mutable state — they only write back to the shared `sections` list via the `operator.add` reducer, which LangGraph merges atomically.

---

## 7. Pydantic Schemas — What to Know

| Schema | Purpose |
|---|---|
| `RouterDecision` | Router output — mode, queries, needs_research flag |
| `Plan` | Orchestrator output — blog title, audience, tone, list of Tasks |
| `Task` | One section spec — title, goal, bullets, word target, flags |
| `EvidenceItem` | One source — title, URL, snippet, publication date |
| `EvidencePack` | Wrapper for list of EvidenceItems (structured output target) |
| `ImageSpec` | One image — placeholder tag, filename, alt, caption, generation prompt |
| `GlobalImagePlan` | Markdown with placeholders + list of ImageSpecs |

---

## 8. Frontend — What to Mention

- Built with **Streamlit** in wide layout
- **Animated pipeline stepper** — 7 stages light up in real time as the graph executes (pending → amber pulsing → green done)
- **Live stats** — shows mode, evidence count, sections planned vs written as the graph runs
- **Session-state key management** — API keys saved to `st.session_state` and restored to `os.environ` on every rerun so they survive Streamlit worker restarts
- **Past blogs panel** — lists previously generated `.md` files with one-click reload
- **Download options** — Markdown only, or ZIP bundle with images

---

## 9. Likely Interview Questions

**Q: What is an agent in this context?**
> Each node is an agent — an autonomous unit that takes state as input, makes a decision (often via an LLM call), and writes back to state. The graph is the multi-agent system coordinating them.

**Q: How is this different from just calling the API multiple times in a for loop?**
> The graph provides: conditional branching (skip research if not needed), true parallelism via Send() (not sequential), shared typed state with automatic reduction, and a subgraph abstraction. A for loop can't express any of that cleanly.

**Q: What happens if one worker fails?**
> LangGraph propagates the exception. In production I'd add a retry decorator or a try/except in the worker that returns a placeholder section so the rest of the blog still assembles.

**Q: How do you ensure the sections appear in the right order after parallel execution?**
> Each worker tags its output with `task.id`. The `merge_content` node sorts by that ID before joining. The `operator.add` reducer just collects tuples — ordering is applied later deliberately, not at write time.

**Q: Why Gemini instead of OpenAI?**
> Gemini has a generous free tier via Google AI Studio (no billing required for text), has native multimodal support for image generation through the same API, and the `langchain-google-genai` integration supports `.with_structured_output()` out of the box with Pydantic.

**Q: What would you improve next?**
> A few things: (1) add a streaming markdown preview so you can watch the blog appear section by section, (2) add human-in-the-loop review of the Plan before workers are dispatched, (3) persist state to a database so you can resume interrupted runs, (4) add a section quality evaluator node that scores each section and re-runs the worker if below threshold.

---

## 10. 30-Second Elevator Pitch (memorise this)

> "I built BlogForge AI — a multi-agent blog generation system using LangGraph and Google Gemini. You give it a topic, and a Router agent classifies whether it needs live web research. If yes, a Research agent pulls sources from the web and filters them by date. Then an Orchestrator agent plans the full blog structure as a validated schema. From there, all sections are written in parallel by independent Worker agents — each with its own task spec and research evidence. Finally a Reducer subgraph merges the sections in order, optionally plans and generates images, and writes the final Markdown file. The whole thing runs as a stateful graph with structured output at every step to prevent hallucination."
