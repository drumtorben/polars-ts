"""Supply-chain demand-sensing agents.

Fuses multi-source signals (POS, social, weather, events) with promotion-effect
estimation, inventory-aware base-stock policies, and multi-echelon order
coordination. Accepts an optional foundation-model baseline for context-aware
forecasting. Closes #163.
"""

from polars_ts._lazy import make_getattr

_IMPORTS: dict[str, tuple[str, str]] = {
    "DemandSensingAgent": ("polars_ts.supply_chain_agents.agents", "DemandSensingAgent"),
    "PromotionEffectAgent": ("polars_ts.supply_chain_agents.agents", "PromotionEffectAgent"),
    "InventoryAgent": ("polars_ts.supply_chain_agents.agents", "InventoryAgent"),
    "EchelonCoordinatorAgent": ("polars_ts.supply_chain_agents.agents", "EchelonCoordinatorAgent"),
    "SupplyChainOrchestrator": ("polars_ts.supply_chain_agents.orchestrator", "SupplyChainOrchestrator"),
    "SupplyChainResult": ("polars_ts.supply_chain_agents.orchestrator", "SupplyChainResult"),
}

__getattr__, __all__ = make_getattr(_IMPORTS, __name__)
