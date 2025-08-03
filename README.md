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


