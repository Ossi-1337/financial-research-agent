from .contracts import ScenarioCatalog, ScenarioDefinition


def create_default_scenario_catalog() -> ScenarioCatalog:
    return ScenarioCatalog(
        (
            ScenarioDefinition(
                id="novo-nordisk",
                version="1.0.0",
                query="Novo Nordisk",
                expected_cik="0000353278",
                preferred_ticker="NVO",
                preferred_exchange="NYSE",
                fiscal_years=3,
                filing_form_limits={"20-F": 1, "6-K": 1},
                market_outputsize="full",
                benchmark_symbol="SPY",
                context_resource="novo_nordisk_context.v1.json",
            ),
        )
    )
