"""Minimalne API HTTP modułu RL (stub zdrowia)."""

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    """Endpoint kontrolny dostępności serwisu."""
    return {"message": "Hello from rl_module"}
