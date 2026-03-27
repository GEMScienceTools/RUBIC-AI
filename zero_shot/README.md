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

For more information about how to use and how works this service (see [LLM Cost Service](llms/readme.md))

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

## 6. Using the `cost_per_area` service

Primary scripts:

- `main.py`: applies selected updates to exposure files.
- `utils.py`: computes `GRM`, `CPI`, `LLM`, and `NEW` cost-per-area tables.
- `get_input_llms.py`: prepares regional LLM input classes.
- `get_output_summary.py`: creates global summary outputs for dashboard/analysis.

### 6.1 Overview

This service is a Python-based pipeline that leverages Large Language Models (LLMs) to produce **zero-shot construction cost benchmarks per square meter (USD/m²)** for buildings worldwide.

It supports five LLMs out of the box — **ChatGPT, Claude, DeepSeek, Gemini, and Grok** — each implemented as a self-contained script sharing the same input/output schema, prompts, and normalization logic. This allows direct cross-model comparison under identical conditions.

---

### 6.2 How It Works

```
Input CSV  ──►  Task Prompt (filled template)  ──►  LLM API call  ──►  JSON response  ──►  Normalized output CSV
```

For each row in the input file, the pipeline:

1. **Reads** country, occupancy type, settlement context, and structural material.
2. **Builds a structured user prompt** using a template (`task_prompt_template.txt`), injecting the row values.
3. **Sends** the system prompt, reference definitions, and task prompt to the chosen LLM API.
4. **Parses** the returned JSON response strictly, with fallback cleanup for common LLM formatting slips (e.g., trailing commas, markdown code fences).
5. **Validates and normalizes** all fields (cost ordering, year bounds, reliability clamping).
6. **Retries** up to `MAX_RETRIES` times if the response is deemed weak (zero costs, missing source, zero reliability).
7. **Writes** a clean CSV with all output columns.

---

### 6.3 Repository Structure

```
cost_areas/
│
├── cost_values_ChatGPT.py         # OpenAI GPT script
├── cost_values_Claude.py          # Anthropic Claude script
├── cost_values_DeepSeek.py        # DeepSeek script (OpenAI-compatible)
├── cost_values_Gemini.py          # Google Gemini script
├── cost_values_Grok.py            # xAI Grok script (OpenAI-compatible)
│
├── system_prompt.txt              # LLM system prompt (role + rules)
├── task_prompt_template.txt       # Per-row user prompt template
├── reference_definitions.txt      # Reliability scale + source format guide
│
├── input_{Region}.csv             # Input files per region (e.g., input_North_America.csv)
│
├── out_llms/
│   └── {Region}/
│       ├── values_chatgpt.csv
│       ├── values_claude.csv
│       ├── values_deepseek.csv
│       ├── values_gemini.csv
│       └── values_grok.csv
│
└── api_key/
    ├── openai_api_key.txt         # OpenAI API key
    ├── claude_api_key.txt         # Anthropic API key
    ├── deepseek_api_key.txt       # DeepSeek API key
    ├── gemini_api_key.txt         # Google Gemini API key
    └── grok_api_key.txt           # xAI Grok API key
```

---

### 6.4 Input Format

Each input CSV must contain the following **five columns** (column order does not matter):

| Column | Type | Description |
|---|---|---|
| `ID_0` | string | ISO 3-letter country code (e.g., `CAN`, `MEX`, `USA`) |
| `NAME_0` | string | Full country name (e.g., `Canada`) |
| `OCCUPANCY` | string | Building occupancy class — any descriptive value is accepted |
| `SETTLEMENT` | string | Settlement context — any descriptive value is accepted |
| `COST_MATERIAL` | string | Dominant structural/construction material — any descriptive value is accepted |

#### 6.4.1 Open-Ended Fields

The `OCCUPANCY`, `SETTLEMENT`, and `COST_MATERIAL` fields are **fully open-ended**. The LLM interprets whatever value you provide — you are not limited to a predefined list. Use whatever classification makes sense for your use case.

Some common examples:

```
OCCUPANCY     →  RES, COM, IND, Mix (Commercial and Residential), Healthcare, Educational, ...
SETTLEMENT    →  TOTAL, URBAN, RURAL, Village, Suburban, Big City, Informal Settlement, ...
COST_MATERIAL →  Concrete, Steel, Adobe, Wood, Bahareque, Bamboo, Rammed Earth, Stone, ...
```

**Example rows:**

```csv
ID_0,NAME_0,OCCUPANCY,SETTLEMENT,COST_MATERIAL
CAN,Canada,RES,URBAN,Concrete
MEX,Mexico,RES,RURAL,Adobe
COL,Colombia,Mix (Commercial and Residential),Village,Bahareque
BGD,Bangladesh,RES,Informal Settlement,Bamboo
USA,United States,COM,TOTAL,Steel
```

The model is instructed to interpret all fields contextually and produce realistic cost estimates regardless of the specific terminology used.

---

### 6.5 Output Format

The output CSV contains all input columns plus the following LLM-estimated fields:

| Column | Type | Description |
|---|---|---|
| `MODEL` | string | LLM used (e.g., `Claude`, `ChatGPT`) |
| `USD_Low-End_cost_m2` | float | Lower-bound construction cost in USD/m² (rounded to nearest 10) |
| `USD_Average-End_cost_m2` | float | Central estimate construction cost in USD/m² (rounded to nearest 10) |
| `year` | int | Reference year of the estimate (typically 2024 or 2025) |
| `reliability` | int | Confidence score from 0 to 5 (see [Reliability Scale](#69-reliability-scale)) |
| `source` | string | Provenance of the estimate in standardized format |

> **Cost values are always** `Low-End ≤ Average-End`. The script enforces this automatically.  
> **Failed rows** are recorded with costs of `0.0`, year `1900`, reliability `0`, and an error message in the `source` field — making them easy to filter downstream.

---

### 6.6 Supported LLMs

All five scripts share an identical input/output schema and the same prompt files, enabling direct comparison.

| Script | Model | Provider | API Client | Response Format |
|---|---|---|---|---|
| `cost_values_ChatGPT.py` | `gpt-5.2` | OpenAI | `openai` | Strict JSON schema |
| `cost_values_Claude.py` | `claude-sonnet-4-5-20250929` | Anthropic | `anthropic` | Prompted JSON + parsing |
| `cost_values_DeepSeek.py` | `deepseek-chat` | DeepSeek | `openai` (compatible) | JSON object mode |
| `cost_values_Gemini.py` | `gemini-3-flash-preview` | Google | `google-genai` | Prompted JSON + parsing |
| `cost_values_Grok.py` | `grok-4-1-fast-non-reasoning` | xAI | `openai` (compatible) | JSON object mode |

> Model names are defined in the `MODEL` constant at the top of each script and can be updated independently as new versions are released.

---

### 6.7 Configuration & Key Files

#### 6.7.1 `system_prompt.txt`

Defines the **LLM persona and hard rules**: JSON-only output, evidence hierarchy, differentiation requirements across occupancy/material, and USD 2025 basis preference. Edit this to change the model's estimation philosophy globally across all scripts.

#### 6.7.2 `task_prompt_template.txt`

A **Python `.format()`-compatible template** injected per row. Contains:
- Interpretation rules for occupancy, settlement, and material fields.
- Modeling guidance (e.g., material cost differentiation, settlement gradients).
- The exact JSON schema the model must return.

#### 6.7.3 `reference_definitions.txt`

Sent as a reference message before the task prompt. Defines:
- Construction quality tiers (Low-End vs. Average-End).
- The 0–5 reliability scale.
- Required `source` field format strings.

Since all three files are shared across all five scripts, modifying any of them affects every model uniformly.

---

### 6.8 Prompt Architecture

Each API call uses a **3-layer prompt structure** (adapted per model's API conventions):

```
[System prompt]          ← role, rules, output philosophy
[User message 1]         ← reference_definitions.txt (reliability + source format)
[User message 2]         ← filled task_prompt_template.txt (specific row data)
```

> **Gemini exception**: the Gemini API does not support separate system/user roles in the same way, so all three layers are concatenated into a single prompt string before the API call.

This separation ensures the model always has the scoring rubric in context before receiving the estimation task, and that retries include additional guidance when the first response is weak.

---

### 6.9 Reliability Scale

The `reliability` field (0–5) reflects the quality of evidence behind each estimate:

| Score | Meaning |
|---|---|
| `5` | Official national or local data |
| `4` | Reputable regional or international organization (e.g., World Bank, Turner & Townsend) |
| `3` | Academic or technical publication |
| `2` | Industry or web sources with contextual support |
| `1` | Expert heuristic or analogy (used when no data is available) |
| `0` | No credible basis |

---

### 6.10 Dependencies

The required packages vary slightly per script depending on the API client used:

| Package | Used by | Purpose |
|---|---|---|
| `anthropic` | Claude | Anthropic API client |
| `openai` | ChatGPT, DeepSeek, Grok | OpenAI and OpenAI-compatible API client |
| `google-genai` | Gemini | Google Generative AI client |
| `pandas` | All | CSV I/O and DataFrame processing |
| `json`, `re`, `math`, `pathlib`, `time` | All | stdlib utilities (no install needed) |

Install all dependencies at once:

```bash
pip install anthropic openai google-genai pandas
```

Or install only what you need for a specific model:

```bash
pip install anthropic pandas        # Claude only
pip install openai pandas           # ChatGPT, DeepSeek, or Grok
pip install google-genai pandas     # Gemini only
```

---

### 6.11 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/cost_areas.git
cd cost_areas

# 2. Create a virtual environment named cost_model
python -m venv cost_model
source cost_model/bin/activate        # Linux / macOS
cost_model\Scripts\activate           # Windows

# 3. Install dependencies
pip install anthropic openai google-genai pandas
```

---

### 6.12 API Key Setup

Each script reads its API key from a plain text file inside the `api_key/` folder. Create one file per provider you intend to use:

| File | Provider | Where to get the key |
|---|---|---|
| `api_key/openai_api_key.txt` | OpenAI (ChatGPT) | https://platform.openai.com/api-keys |
| `api_key/claude_api_key.txt` | Anthropic (Claude) | https://console.anthropic.com |
| `api_key/deepseek_api_key.txt` | DeepSeek | https://platform.deepseek.com |
| `api_key/gemini_api_key.txt` | Google (Gemini) | https://aistudio.google.com/app/apikey |
| `api_key/grok_api_key.txt` | xAI (Grok) | https://console.x.ai |

Each file should contain only the key string, with no extra whitespace or newlines:

```
sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

> ⚠️ **Never commit API keys to version control.** Add `api_key/` to your `.gitignore`:

```gitignore
# .gitignore
api_key/
```

---

### 6.13 Usage

#### 6.13.1 Running a single model

```bash
# Activate the virtual environment first
source cost_model/bin/activate      # Linux / macOS
cost_model\Scripts\activate         # Windows

# Then run any of the five model scripts
python cost_values_Claude.py
python cost_values_ChatGPT.py
python cost_values_DeepSeek.py
python cost_values_Gemini.py
python cost_values_Grok.py
```

#### 6.13.2 Multi-region loop

Each script's `__main__` block iterates over a list of regions via a companion `grm.REGIONS` module. You can also override this manually by editing the loop:

```python
regions = ["North_America", "South_America", "Europe", "Africa"]
for region in regions:
    INPUT_FILE  = f"input_{region}.csv"
    OUTPUT_FILE = f"out_llms/{region}/values_claude.csv"
    main()
```

#### 6.13.3 Running all models on the same input

To generate outputs from all five LLMs for the same region, run each script in sequence:

```bash
python cost_values_ChatGPT.py
python cost_values_Claude.py
python cost_values_DeepSeek.py
python cost_values_Gemini.py
python cost_values_Grok.py
```

All outputs will land in `out_llms/{Region}/` under their respective filenames, ready for cross-model comparison.

#### 6.13.4 Key runtime parameters

These constants appear at the top of each script and can be tuned independently per model:

| Constant | Description | Default |
|---|---|---|
| `MODEL` | Model version string sent to the API | varies per script |
| `SLEEP_BETWEEN_CALLS` | Seconds to wait between API calls (rate limiting) | `2`–`3` |
| `MAX_RETRIES` | Max retry attempts for weak/invalid responses | `2` |

---

### 6.14 Notes & Limitations

- **Cost estimates are LLM-generated** and should be treated as benchmarks, not certified appraisals. Always cross-check against local official sources when available.
- **Input fields are open-ended by design.** The model interprets `OCCUPANCY`, `SETTLEMENT`, and `COST_MATERIAL` contextually. Unusual or highly specific values may result in lower reliability scores but will still produce an estimate.
- **Reliability scores are self-reported** by the model based on the evidence it cites. Higher scores do not guarantee accuracy.
- **Costs are rounded to the nearest 10 USD/m²** to avoid false precision in zero-shot estimation.
- **Error rows** (API failures, unparseable responses) are written with `year = 1900` and `reliability = 0` as sentinel values, making them easy to filter out in downstream analysis.
- Scripts use `temperature=0.2` on first attempts and `temperature=0.0` on retries for increased determinism.
- **API costs**: each row consumes approximately 1,000–2,000 tokens. For large datasets, estimate token usage before running. Pricing pages: [OpenAI](https://openai.com/pricing) · [Anthropic](https://www.anthropic.com/pricing) · [DeepSeek](https://platform.deepseek.com/docs) · [Google](https://ai.google.dev/pricing) · [xAI](https://x.ai/api).
