"""
FastAPI backend for the Blackjack Reasoning Tutor.

Streams AG-UI protocol events (SSE) from an AG2 multi-agent swarm.
Each POST to /copilotkit runs one swarm turn until REVERT_TO_USER fires.
State is re-hydrated from the request body on every call.
"""

from __future__ import annotations

import asyncio
import logging
import time
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()

# Attempt AGUIStream import — log result, always use manual SSE path for swarm support
try:
    from autogen.ag_ui import AGUIStream  # noqa: F401
    logging.info("AGUIStream is available but not used: swarm requires manual SSE")
except ImportError:
    logging.warning("AGUIStream not available — using manual SSE fallback")

from ag_ui.core import RunFinishedEvent, RunStartedEvent, StateSnapshotEvent
from ag_ui.encoder import EventEncoder

from swarm_builder import run_swarm_turn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Blackjack Tutor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

encoder = EventEncoder()


@app.post("/copilotkit")
async def copilotkit_endpoint(request: Request) -> StreamingResponse:
    body = await request.json()
    return StreamingResponse(
        _generate_sse(body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _generate_sse(body: dict):
    thread_id: str = body.get("threadId", str(uuid4()))
    run_id: str = str(uuid4())
    messages: list = body.get("messages", [])
    state: dict = body.get("state", {})

    def ts() -> int:
        return int(time.time() * 1000)

    queue: asyncio.Queue = asyncio.Queue()

    yield encoder.encode(RunStartedEvent(thread_id=thread_id, run_id=run_id, timestamp=ts()))

    # Echo back incoming state immediately so the frontend stays in sync
    if state:
        yield encoder.encode(StateSnapshotEvent(snapshot=state, timestamp=ts()))

    task = asyncio.create_task(run_swarm_turn(messages, state, queue))

    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=120.0)
        except asyncio.TimeoutError:
            logger.warning("Swarm turn timed out after 120s")
            break
        if event is None:
            break
        yield encoder.encode(event)

    await task
    yield encoder.encode(RunFinishedEvent(thread_id=thread_id, run_id=run_id, timestamp=ts()))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
