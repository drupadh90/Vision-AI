"""Pydantic request/response contracts for the Vision AI API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GoalRequest(BaseModel):
    goal: str = Field(..., min_length=3, description="High-level objective.")
    user_id: str = "default"


class GoalResponse(BaseModel):
    run_id: str = ""
    goal: str
    strategy: str = ""
    final_answer: str = ""
    tasks: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    activated_skills: list[dict[str, Any]] = []
    reflection: dict[str, Any] = {}
    error: str = ""


class LearnRequest(BaseModel):
    url: str = Field(..., description="YouTube URL or 11-character video id.")
    skill_name: str | None = None
    force: bool = False


class TextSkillRequest(BaseModel):
    title: str
    text: str = Field(..., min_length=20)
    source: str = "manual"


class SkillSearchRequest(BaseModel):
    query: str
    k: int = 5


class MemoryWriteRequest(BaseModel):
    content: str = Field(..., min_length=2)
    kind: str = "preference"
    confidence: float = 0.6
    user_id: str = "default"


class MemoryQueryRequest(BaseModel):
    query: str
    k: int = 6
    user_id: str = "default"


class ReflectRequest(BaseModel):
    task: str
    outcome: str
    success: bool = True
    user_id: str = "default"


class IntuitionRequest(BaseModel):
    context: str
    user_id: str = "default"


class ApprovalDecisionRequest(BaseModel):
    approved: bool
    decided_by: str = "user"
    note: str = ""


class GuardianCheckRequest(BaseModel):
    action_type: str = "shell"
    payload: str
    description: str = ""
    target_path: str | None = None
