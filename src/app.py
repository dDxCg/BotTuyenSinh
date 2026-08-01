from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.service import Reply, Service


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=2000)


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = Service()
    yield


app = FastAPI(title="BotTuyenSinh API", lifespan=lifespan)


def get_service() -> Service:
    return app.state.service


ServiceDep = Annotated[Service, Depends(get_service)]


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat", response_model=Reply)
def chat(request: ChatRequest, service: ServiceDep) -> Reply:
    try:
        return service.chat(request.session_id, request.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"API error: {type(exc).__name__}: {exc}", flush=True)
        raise HTTPException(
            status_code=500, detail="Chatbot đang gặp lỗi tạm thời. Vui lòng thử lại."
        ) from exc


@app.post("/api/reset")
def reset(request: ResetRequest, service: ServiceDep) -> dict:
    service.reset(request.session_id)
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Chạy web demo chatbot tuyển sinh")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
