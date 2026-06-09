"""nlqueries.feedback — query feedback capture and local JSONL storage."""

from nlqueries.feedback.models import QueryFeedback
from nlqueries.feedback.store import load_feedback, record_feedback

__all__ = ["QueryFeedback", "load_feedback", "record_feedback"]
