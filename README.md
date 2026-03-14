<div align=center>
   
<img src="image/neuryx.png">

</div>

>## **General-Purpose Neural Sequence Engine**

- A decoder-only transformer built entirely from scratch in pure Python — no PyTorch, no TensorFlow, no NumPy. Train it on *any* sequential data, then generate new sequences from it.

>## What is Neuryx?

- Neuryx is a **general-purpose sequence learner**. You give it data; it learns patterns; it generates new data that follows those patterns. It does not care what the data *means* — names, weather events, server logs, DNA sequences, user actions, or any other text-based sequence will work.

Under the hood it is a **causal (decoder-only) transformer** with:
- Scalar-level automatic differentiation (no ML library required)
- Multi-head self-attention with KV caching
- RMSNorm normalisation
- Adam-style adaptive optimizer
- Character, word, or token-level encoding

---

>## Quick Start

```bash
git clone https://github.com/Chintanpatel24/Neuryx.git
cd Neuryx
python3 neuryx.py

# Optional but recommended — for charts
pip install matplotlib openpyxl

# Interactive mode — it will ask you for everything
python neuryx.py

# Or pass files directly
python neuryx.py --train data/sample_names.txt --predict data/sample_names.txt
```

---

>## Input Format Menu

When you run Neuryx, it presents a numbered menu for every file you load:

```
  [1]  Plain Text  (.txt)   — one document per line
  [2]  CSV         (.csv)   — choose a column
  [3]  Excel       (.xlsx)  — choose a column
  [4]  JSON        (.json)  — array of strings or records
  [5]  TSV         (.tsv)   — tab-separated, choose a column
  [6]  Auto-detect          — guess from file extension
```

Pick the number that matches your file. For CSV / Excel / TSV you will also be asked which column contains the text you want to learn from.

---

>## Included Sample Data

| File | Format | Rows | Use it for |
|------|--------|------|------------|
| `data/sample_names.txt` | TXT | 172 | Character-level name generation |
| `data/sample_weather.csv` | CSV | 800 | Weather-pattern sequence learning |
| `data/sample_logs.json` | JSON | 600 | Server-event sequence modelling |
| `data/sample_events.tsv` | TSV | 700 | UI-action sequence generation |
| `data/sample_text.txt` | TXT | 1 200 | Phrase / sentence generation |
| `data/sample_sequences.xlsx` | Excel | 500 | DNA / music / Morse code generation |

---

>## CLI Flags

```
python neuryx.py [OPTIONS]

  --train       FILE    Training dataset path
  --predict     FILE    Prediction seed dataset path
  --steps       INT     Training steps          (default: 500)
  --samples     INT     Outputs to generate     (default: 20)
  --temperature FLOAT   Sampling temperature    (default: 0.5)
  --mode        STR     char | word | token     (default: char)
  --depth       INT     Embedding dimension     (default: 32)
  --rifts       INT     Transformer blocks      (default: 2)
  --horizon     INT     Context window length   (default: 64)
  --no-chart            Skip matplotlib dashboard
```

>### Temperature

| Value | Effect |
|-------|--------|
| `0.1` | Very conservative — near-deterministic outputs |
| `0.5` | Balanced (default) |
| `1.0` | Maximum creativity / randomness |

### Tokenisation Modes

| Mode | Best for |
|------|----------|
| `char` | Names, DNA, short text (default) |
| `word` | Sentences, paragraphs, phrases |
| `token` | Pre-labelled categorical sequences |

---

## Example Sessions

### 1. Learn names, generate new names
```bash
python neuryx.py \
  --train   data/sample_names.txt \
  --predict data/sample_names.txt \
  --mode    char --steps 600 --temperature 0.4
```

### 2. Learn weather patterns
```bash
python neuryx.py \
  --train   data/sample_weather.csv \
  --predict data/sample_weather.csv \
  --mode    word --steps 400
# → When prompted for column, type: label
```

### 3. Learn server-event sequences
```bash
python neuryx.py \
  --train   data/sample_logs.json \
  --predict data/sample_logs.json \
  --mode    token
# → JSON auto-selects the first string field
```

### 4. Bigger model for richer output
```bash
python neuryx.py \
  --train data/sample_text.txt \
  --depth 64 --rifts 3 --horizon 128 --steps 1000
```

---

>## Repository Structure

```
neuryx/
│
├── neuryx.py              ← Entry point  (python neuryx.py)
│
├── core/                  ← Neural engine (zero external deps)
│   ├── flux.py            — Scalar autodiff (the "tensor" layer)
│   ├── lattice.py         — Decoder-only transformer model
│   ├── apex.py            — Adam-style optimizer
│   └── forge.py           — Training & inference loop
│
├── intake/                ← Data ingestion
│   ├── portal.py          — Multi-format loader (txt/csv/xlsx/json/tsv)
│   └── cipher.py          — Tokeniser / vocabulary builder
│
├── shell/                 ← Terminal UI
│   └── canvas.py          — ANSI colours, progress bars, menus
│
├── render/                ← Visualisation (optional matplotlib)
│   └── prism.py           — 8-panel dashboard
│
└── data/                  ← Sample datasets
    ├── sample_names.txt
    ├── sample_weather.csv
    ├── sample_logs.json
    ├── sample_events.tsv
    ├── sample_text.txt
    └── sample_sequences.xlsx
```

---

>## How It Works

### 1. Data loading (`intake/portal.py`)
Your file is loaded and converted to a flat `list[str]` regardless of format.

### 2. Tokenisation (`intake/cipher.py`)
Each string is split into symbols (chars, words, or tokens). A vocabulary is built from all unique symbols. A special **seal** (BOS) token marks sequence boundaries.

### 3. Model (`core/lattice.py`)
A causal transformer processes one token at a time:
- Token + position embeddings are added
- Each transformer block applies: **RMSNorm → Multi-Head Attention → residual** then **RMSNorm → FFN → residual**
- The output is projected to a probability distribution over the vocabulary

### 4. Training (`core/forge.py + core/apex.py`)
Sequences are fed through the model; cross-entropy loss is computed; gradients flow back through the scalar computation graph (`core/flux.py`); Adam updates the weights.

### 5. Inference
Seed documents are encoded and fed to the model. It predicts the next token, samples from the distribution (controlled by temperature), appends the token, and repeats until the seal token appears or the horizon is reached.

---

>## Requirements

| Package | Required? | Purpose |
|---------|-----------|---------|
| Python ≥ 3.10 | ✅ Yes | Core runtime |
| `matplotlib` | ⬜ Optional | Dashboard charts |
| `openpyxl` | ⬜ Optional | Excel `.xlsx` support |

Install optionals:
```bash
pip install matplotlib openpyxl
```
