# DataXFormer: A Robust Data Transformation Discovery App

**DataXFormer** is an interactive tool for discovering transformations between different data representations. It enables users to input example pairs (x → y) and queries (xq) to retrieve predicted outputs (yq) using a backend powered by Vertica and advanced web table analysis.

This app is based on the system presented in the ICDE 2016 paper:  
[DataXFormer: A Robust Transformation Discovery System](https://ieeexplore.ieee.org/document/7498313)  
by Ziawasch Abedjan, John Morcos, Ihab F. Ilyas, Mourad Ouzzani, Paolo Papotti, and Michael Stonebraker.

---

## 🚀 Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/zenzeii/dataxformer.git
cd dataxformer
```

### 2. Create and Activate a Virtual Environment (Windows)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare to Run App
- Add Database Credentials
- Install and run ollama (optional for LLM validation)


### 5. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 🚀 Components Overview
The architecture of DataXFormer is modular and robust, designed to support example-driven transformation discovery at scale. Below are the core components:
![alt text](https://raw.githubusercontent.com/zenzeii/lsdipro-dataxformer/refs/heads/main/assets/architecture.png "Architecture")
![alt text](https://raw.githubusercontent.com/zenzeii/lsdipro-dataxformer/refs/heads/main/assets/technical_aspect.png "Technical Aspects")

### 🔤 Text Normalization
Standardizes all text inputs before matching:
- Lowercasing 
- Removing punctuation/special characters 
- Stop word removal (e.g., "and", "is")
- Tokenization (splitting into tokens)
- Diacritic removal (e.g., “é” → “e”)
- Stemming (e.g., “running” → “run”)
- Whitespace normalization

Powered by Whoosh’s StandardAnalyzer, this ensures consistent preprocessing for accurate candidate lookup.

### 🔍 Candidate Lookup
Searches for relevant tables/columns in a large preprocessed dataset:
- Utilizes Vertica to store and query the indexed version of the Dresden Web Table Corpus (~120M tables). 
- Finds matches for example pairs using normalized tokens. 
- Computes:
  - Prediction Scores (based on how well the candidate maps examples)
  - Table Scores (based on overall relevance of table)

### 🔎 Fuzzy Matching
Identifies close matches when exact matches fail:
- Uses RapidFuzz and Python’s difflib for approximate string similarity. 
- Enables robustness against typos and inconsistencies (e.g., “Berlinn” → “Berlin”).

### 📊 Scoring and Ranking
Applies an Iterative EM (Expectation-Maximization) Algorithm:
- Alternates between updating table scores and prediction scores until convergence. 
- Ranks final candidate transformations to provide the best suggestion for user queries.

### 🤖 LLM Validation (Optional)
A post-processing step that can validate and refine transformation outputs using a local LLM (e.g., via ollama):
- Filters noisy predictions 
- Improves semantic accuracy of final result

Can be enabled/disabled in the Streamlit interface.
