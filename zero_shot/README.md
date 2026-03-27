# Cost per area updates

This document summarizes the methodology used by `scripts/cost_per_area` to improve `COST_PER_AREA_USD` in exposure models.

The workflow combines four information sources:

1. **GRM-derived baseline costs** from exposure models.
2. **LLM-derived construction cost ranges** (consolidated across models).
3. **Country CPI adjustment factors**.
4. **Expert judgment values** for selected cases.

Updated values are applied to country exposure files and then summarized for dashboard and analysis outputs.

## 1. Inputs and standardization

### 1.1 Base exposure information

For each region and country, building classes are grouped by:

- `ID_0`, `NAME_0`, `OCCUPANCY`, `SETTLEMENT`, `TAXONOMY`

To support cost estimation, each taxonomy is mapped to a `COST_MATERIAL` class, including hybrid taxonomies (using the predominant material as reference).

| Category | Material Codes | Description |
|----------|----------------|-------------|
| **Adobe** | `MUR+ADO` | Adobe/mud brick |
| **Bamboo** | `W+WBB` | Bamboo materials |
| **Concrete** | `CR`, `C` | Concrete materials |
| **Earth** | `EU`, `ER`, `E+`, contains `+WWD` | Earth-based materials |
| **Informal** | `MATO`, `ME`, `INF` | Informal construction materials |
| **Reinforced/Confined Masonry** | `MR`, `MCF` | Masonry with reinforcement or confinement |
| **Steel** | `S` | Steel materials |
| **Stone** | `MUR+ST` | Stone masonry |
| **Unreinforced Masonry** | `MUR`, `M+`, `M` | Masonry without reinforcement |
| **Wood** | `W` | Wood materials (excluding bamboo) |

For integration and expert review, `COST_MATERIAL` is further simplified:

- `Adobe`, `Earth` → `Adobe|Earth`
- `Bamboo` → `Wood`
- `Informal` → `Others`
- `Concrete`, `Steel`, `Reinforced/Confined Masonry` → `Formal (S|CR|MR|MCF)`

### 1.2 Settlement harmonization

- `SETTLEMENT` is normalized to uppercase.
- For non-residential occupancies, settlement is standardized to `TOTAL` (no urban/rural split).

## 2. Construction cost estimation with LLMs

Five LLMs provide independent cost estimates:

- ChatGPT
- Claude
- DeepSeek
- Gemini
- Grok

The estimation is zero-shot and country-specific, with inputs defined by:

- country
- occupancy (`RES`, `COM`, `IND`)
- settlement
- material / simplified material class

`get_input_llms.py` prepares regional input classes used for LLM estimation.

### 2.1 Prompt design (summary)

The prompt framework includes:

1. **System instructions** (role, data-use rules, output format).
2. **Reference definitions** (quality levels, reliability scale, source format).
3. **Task payload** (country/class inputs + JSON output schema).

### 2.2 Consolidation and selection logic

Across all LLM outputs:

- median `LOW_COST_PER_AREA`
- median `AVG_COST_PER_AREA`

Both are rounded to the nearest multiple of 10.

`LLM_COST_PER_AREA` is selected as follows:

1. `Concrete`, `Steel`, `Reinforced/Confined Masonry` → use `AVG_COST_PER_AREA`.
2. Other materials → use `LOW_COST_PER_AREA`.
3. `Wood` in high-cost contexts (`USA`, `CAN`, `AUS`, `NZL`, `JPN`, `TWN`, and Europe) → use `AVG_COST_PER_AREA`.
4. Informal construction is constrained to the minimum group value when needed.

For more information about how to use and how works this service (see [LLM Cost Service](llms/README.md))

### 2.3 Limitation

LLM estimates are intended for consistent large-scale modeling, not for project-level budgeting or detailed quantity surveying.

## 3. CPI-based updates

CPI is used to scale existing GRM values where configured:

- updated value = existing value × CPI factor
- final value rounded to nearest 10

CPI is globally available and regularly updated, making it practical for broad coverage.

## 4. Expert judgment input

Expert-reviewed values are used for selected country/occupancy/material combinations, supported by construction cost references (including 2024–2025 sources).

For interpretation and benchmarking, outputs are also compared against macroeconomic context (including PLI and GDP PPP per capita).

## 5. Final update criteria for `COST_PER_AREA_USD`

Each country/occupancy combination is assigned one criterion:

- `GRM`: keep GRM reference values (typically recently updated models).
- `CPI`: scale existing GRM values by CPI factor.
- `LLMs`: apply consolidated LLM-based values.
- `EXP`: apply expert-reviewed values.
