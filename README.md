# BotTuyenSinh — Admissions Information Assistant (AI in Action - VinUni)

Admissions chatbot using RAG (pgvector) + a LangGraph agent (native tool-calling via OpenRouter), served through FastAPI with a Streamlit UI.
 
 **Live demo:**
 https://bottuyensinh-xjudttlmmkbykvdcc8uyzg.streamlit.app/

## Architecture

```mermaid
flowchart LR
    User[User]
    UI[Streamlit UI<br/>src/streamlit_app]
    API[FastAPI<br/>src/app.py]
    Service[Service<br/>src/service.py]
    Graph[LangGraph agent<br/>src/chatbot/graph/]
    Retriever[PgVectorRetriever<br/>src/rag/rag_bridge.py]
    DB[(Postgres + pgvector<br/>Neon)]
    LLM[OpenRouter<br/>chat + embedding]

    User --> UI --> API --> Service --> Graph
    Graph <--> Retriever <--> DB
    Graph <--> LLM
```

```mermaid
flowchart TD
    Start([Question]) --> Guardrail{guardrail}
    Guardrail -->|restricted| RespondRestricted[respond_restricted]
    Guardrail -->|ok| Planner[planner]
    Planner --> QuerySplit[query_split]
    QuerySplit --> Hyde[hyde]
    Hyde --> Retrieve[retrieve]
    Retrieve --> Grounding{grounding_decision}
    Grounding -->|not grounded| RespondNoGrounding[respond_no_grounding]
    Grounding -->|grounded| Agent[agent]
    Agent -->|tool_calls| Tools[tools]
    Tools -->|force_finalize| Finalize
    Tools -->|otherwise| Agent
    Agent -->|done| Finalize[finalize]
    RespondRestricted --> End([Answer])
    RespondNoGrounding --> End
    Finalize --> End
```

Node factory theo file (`src/chatbot/graph/nodes/`): mỗi node 1 file (`guardrail.py`, `planner.py`,
`query_split.py`, `hyde.py`, `retrieve.py`, `grounding.py`, `agent.py`, `tools.py`, `finalize.py`),
nối lại trong `src/chatbot/graph/builder.py`. Graph shape cố định — `planner` (model rẻ riêng,
`JUDGE_MODEL`) phân tích câu hỏi và quyết định `query_split`/`hyde` có chạy hay không qua
`state["plan"]`; cả 2 node luôn có trong graph nhưng tự no-op nếu không được chọn. Thêm kỹ thuật
mới sau này: thêm node vào cuối chuỗi + tên vào danh sách kỹ thuật planner biết chọn — xem
`docs/plan/hyde-and-semantic-query-split.md`.

## Tech stack

- **Backend**: FastAPI, Uvicorn
- **Agent**: LangGraph
- **LLM**: OpenRouter
- **Vector DB**: pgvector (Neon)
- **UI**: Streamlit
- **Test**: pytest
- **Package manager**: uv

## Getting started

Install dependencies:
```
uv sync
```

Configure `.env` (copy from `.env.example`): `DATABASE_URL` (Neon), `OPENAI_API`/`OPENAI_BASE_URL` (chat), `EMBEDDING_API_KEY`/`EMBEDDING_MODEL` (embedding), `TOP_K` (số chunk/lần retrieve), `JUDGE_API`/`JUDGE_MODEL` (model rẻ cho planner, trống thì dùng chung OPENAI_*)

Build RAG data (chunk + embed + store in pgvector, one command):
```
uv run python -m src.rag.pipeline --recreate
```

Chunking strategy — pick with `--strategy {structure,sentence,semantic}` (default `structure`):
- **structure** — splits along document structure: heading hierarchy + paragraph/list-item boundaries (`src/rag/chunking.py`). Falls back to line/sentence/word splitting only when a block exceeds `max_chars`.
- **sentence** — splits by sentence instead of paragraph/block, packing consecutive sentences up to `max_chars`.
- **semantic** — embedding-based (no LLM): embeds each sentence via the configured embedding model, cuts where cosine distance between consecutive sentences spikes (95th percentile breakpoint), then packs groups up to `max_chars` (`src/rag/semantic_chunking.py`).

```
uv run python -m src.rag.pipeline --strategy sentence --recreate
```

Run the FastAPI backend:
```
uv run python -m src.app
```

Run the Streamlit UI (backend must already be running):
```
uv run streamlit run src/streamlit_app/app.py
```

Run the chat CLI (no FastAPI/Streamlit needed):
```
uv run python -m src.chatbot
```

Run tests:
```
uv run pytest
```

Test the Postgres/Neon connection:
```
uv run python -m src.db_client
```
