"""Flag system for reporting data quality issues and diagnostics."""

from dataclasses import dataclass


@dataclass
class FlagRecord:
    """Structured flag for diagnostics and quality reporting.

    Attributes:
        scope: 'global' | 'window' | 'piece'
        scope_id: identifier for the scope (e.g., window index, piece label)
        severity: 'info' | 'warn' | 'reject'
        code: flag code (e.g., 'TAIL_TRIM_APPLIED', 'INSUFFICIENT_SAMPLES')
        message: human-readable description
        details: optional additional information
    """

    scope: str
    scope_id: str
    severity: str
    code: str
    message: str
    details: str = ""

    def __str__(self) -> str:
        """Format flag as readable string."""
        parts = [f"[{self.severity.upper()}]", self.code]
        if self.scope != "global":
            parts.append(f"({self.scope}={self.scope_id})")
        parts.append(f"- {self.message}")
        if self.details:
            parts.append(f" [{self.details}]")
        return " ".join(parts)
