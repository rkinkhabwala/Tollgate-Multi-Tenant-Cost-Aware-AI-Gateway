"""Model catalog: `config/models.yaml` → `models` table → in-memory lookup.

At startup the YAML is upserted into `models` (platform-managed, no tenant data),
and the rows are kept in memory. Cost calculation reads prices from here on every
request, so it must not hit the database.

Cost formula (spec §11): `input_tokens × input_price + output_tokens × output_price`,
with prices in USD per million tokens. Provider cache discounts arrive in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import yaml
from sqlalchemy.dialects.postgresql import insert

from gateway.db.models import Model
from gateway.db.session import Database

MTOK = Decimal(1_000_000)
USD_QUANT = Decimal("0.000001")  # NUMERIC(12,6)


@dataclass(frozen=True)
class CatalogModel:
    id: str
    provider: str
    provider_model: str
    tier: str
    input_usd_per_mtok: Decimal
    output_usd_per_mtok: Decimal
    context_window: int
    supports_tools: bool = False
    supports_json: bool = False
    enabled: bool = True

    def cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        raw = (input_tokens * self.input_usd_per_mtok + output_tokens * self.output_usd_per_mtok) / MTOK
        return raw.quantize(USD_QUANT)


class ModelCatalog:
    def __init__(self, models: list[CatalogModel], tiers: dict[str, list[str]] | None = None) -> None:
        self._by_id = {m.id: m for m in models}
        self.tiers = tiers or {}

    @classmethod
    def from_yaml(cls, path: Path) -> ModelCatalog:
        raw = yaml.safe_load(path.read_text())
        models = [
            CatalogModel(
                **{
                    **m,
                    "input_usd_per_mtok": Decimal(str(m["input_usd_per_mtok"])),
                    "output_usd_per_mtok": Decimal(str(m["output_usd_per_mtok"])),
                }
            )
            for m in raw["models"]
        ]
        return cls(models, raw.get("tiers", {}))

    def get(self, model_id: str) -> CatalogModel | None:
        m = self._by_id.get(model_id)
        return m if m and m.enabled else None

    def all(self) -> list[CatalogModel]:
        return [m for m in self._by_id.values() if m.enabled]

    def cost(self, model_id: str, input_tokens: int, output_tokens: int) -> Decimal:
        """Price tokens at `model_id`. Unknown models cost 0 (logged by the caller)."""
        m = self.get(model_id)
        return m.cost(input_tokens, output_tokens) if m else Decimal(0).quantize(USD_QUANT)

    async def sync_to_db(self, db: Database) -> None:
        """Upsert every catalog row into `models` (system role; the table is global)."""
        async with db.system_session() as s:
            for m in self._by_id.values():
                values = {k: getattr(m, k) for k in CatalogModel.__dataclass_fields__}
                stmt = insert(Model).values(**values)
                stmt = stmt.on_conflict_do_update(index_elements=[Model.id], set_={k: stmt.excluded[k] for k in values if k != "id"})
                await s.execute(stmt)
