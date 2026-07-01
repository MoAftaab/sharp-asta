from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)

    @model_validator(mode="after")
    def at_least_one_user(self) -> "ChatRequest":
        roles = [m.role for m in self.messages]
        if self.messages and "user" not in roles:
            raise ValueError("Must have at least one user message")
        return self


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # single letter: A/B/C/D/E/K/P/S

    @model_validator(mode="after")
    def url_must_be_shl(self) -> "Recommendation":
        if not self.url.startswith("https://www.shl.com/"):
            raise ValueError(f"Recommendation URL must be from shl.com, got: {self.url}")
        return self


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

    @model_validator(mode="after")
    def max_ten_recommendations(self) -> "ChatResponse":
        if len(self.recommendations) > 10:
            raise ValueError("recommendations must have at most 10 items")
        return self


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
