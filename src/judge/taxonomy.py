"""Failure taxonomy and result models for the SQL judge."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FailureType(str, Enum):
    """Classification of how a generated SQL diverges from the ground truth."""

    CORRECT = "CORRECT"
    CORRECT_DIFFERENT = "CORRECT_DIFFERENT"  # same results, different SQL
    PARTIAL_MATCH = "PARTIAL_MATCH"          # results overlap but aren't identical
    WRONG_TABLE = "WRONG_TABLE"
    WRONG_JOIN = "WRONG_JOIN"
    WRONG_COLUMN = "WRONG_COLUMN"
    WRONG_AGGREGATION = "WRONG_AGGREGATION"
    WRONG_FILTER = "WRONG_FILTER"
    WRONG_ORDER = "WRONG_ORDER"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"


class EvalResult(BaseModel):
    """Result of judging a single generated SQL against its ground truth."""

    query_id: int
    nl: str
    difficulty: str
    ground_truth_sql: str
    generated_sql: str
    backend: str

    # Judge outputs
    is_correct: bool
    failure_type: FailureType
    failure_details: str = ""
    ast_diff: dict = Field(default_factory=dict)

    # Execution metadata
    ground_truth_rows: int = 0
    generated_rows: int = 0
    execution_error: str = ""


class ScoreCard(BaseModel):
    """Aggregated evaluation scorecard."""

    total_queries: int
    total_correct: int
    overall_accuracy: float

    accuracy_by_backend: dict[str, float]
    accuracy_by_difficulty: dict[str, float]
    failure_distribution: dict[str, dict[str, int]]  # backend -> {failure_type: count}

    results: list[EvalResult]
