"""Phase 3: long-context pricing columns on `models`; per-attempt ledger on `usage_events`.

Revision ID: 0002_phase3
Revises: 0001_initial
Create Date: 2026-10-14

* `models.long_context_threshold / long_input_usd_per_mtok / long_output_usd_per_mtok`:
  prompt-length price tiers (Claude Haiku 5.5 bills prompts > 100K tokens higher).
* `usage_events.attempts`: one JSON object per upstream call (model, status, tokens,
  cost, latency, error code). Fallbacks and escalations are therefore visible in the
  ledger, and every attempt is billed (spec §8.4). It never holds text.
"""

from alembic import op

revision = "0002_phase3"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE models ADD COLUMN long_context_threshold INT")
    op.execute("ALTER TABLE models ADD COLUMN long_input_usd_per_mtok NUMERIC(10,4)")
    op.execute("ALTER TABLE models ADD COLUMN long_output_usd_per_mtok NUMERIC(10,4)")
    op.execute("ALTER TABLE usage_events ADD COLUMN attempts JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE usage_events DROP COLUMN attempts")
    op.execute("ALTER TABLE models DROP COLUMN long_output_usd_per_mtok")
    op.execute("ALTER TABLE models DROP COLUMN long_input_usd_per_mtok")
    op.execute("ALTER TABLE models DROP COLUMN long_context_threshold")
