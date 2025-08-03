from whoosh.analysis import StandardAnalyzer
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
import streamlit as st
import vertica_python
import pandas as pd
import numpy as np
import requests
import time
import os

# ------------------ App Configuration ------------------
st.set_page_config(page_title="DataXFormer", page_icon="⚡", layout="wide")
st.title("⚡ DataXFormer")
st.text("A robust, fast, and reliable discovery system for data transformations.")

# ------------------ Database Connection ------------------
# TODO: Please enter credentials to connect with the database
conn_info = {}

@st.cache_resource(show_spinner=False)
def get_vertica_connection():
    try:
        return vertica_python.connect(**conn_info)
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

    # ------------------ Utilities ------------------


analyzer = StandardAnalyzer()


def normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    tokens = [token.text for token in analyzer(s)]
    return " ".join(tokens)


def enhanced_normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # Use Whoosh analyzer for better text processing
    tokens = [token.text for token in analyzer(s)]
    normalized = " ".join(tokens)
    # Additional cleaning for special characters
    normalized = normalized.replace("'", "'").replace("\"", '"').replace("\"", '"')
    normalized = normalized.replace("–", "-").replace("—", "-")
    return normalized


def fuzzy_match(a: str, b: str, threshold: int = 85) -> bool:
    return fuzz.ratio(normalize(a), normalize(b)) >= threshold


def sql_escape(s: str) -> str:
    return s.replace("'", "''").replace("\n", " ").replace("\r", " ").strip()


def get_ollama_model_list(url="https://ollama.com/library") -> list:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise error if status code != 200
        soup = BeautifulSoup(response.text, "html.parser")

        model_elements = soup.find_all("a", href=True)
        models = set()

        for a in model_elements:
            href = a['href']
            if href.startswith("/library/") and len(href.split("/")) == 3:
                model_name = href.split("/")[-1].strip()
                if model_name:
                    models.add(model_name)

        return sorted(models)

    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def ollama_semantic_check(predicted: str, expected: str, model: str = "mistral") -> bool:
    """
    Use Ollama to check if predicted and expected answers are semantically equivalent.

    Args:
        predicted: The predicted answer
        expected: The expected/ground truth answer
        model: The Ollama model to use (default: mistral)

    Returns:
        True if semantically equivalent, False otherwise
    """
    try:
        prompt = f"""Are "{predicted}" and "{expected}" semantically equivalent? 
        Consider synonyms, different phrasings, and alternative expressions that mean the same thing.
        Answer only 'Yes' or 'No'."""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )

        if response.status_code == 200:
            content = response.json().get("response", "").strip().lower()
            return "yes" in content
        else:
            st.warning(f"Ollama API error: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to Ollama. Make sure Ollama is running on localhost:11434")
        return False
    except requests.exceptions.Timeout:
        st.warning("⏰ Ollama request timed out")
        return False
    except Exception as e:
        st.error(f"❌ Ollama error: {e}")
        return False


def adaptive_data_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Data splitting following the DataXFormer paper's approach:
    - Use exactly 5 examples for training (as per paper)
    - Use 2-3 examples for validation
    - Remaining data for testing
    - For tiny datasets (< 10 rows): use all data for training
    """
    total_rows = len(df)

    if total_rows < 10:
        # For tiny datasets, use all data for training
        return df, pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    # Paper's approach: exactly 5 training examples
    n_train = 5
    n_val = min(3, total_rows - n_train)  # Use 2-3 for validation

    # Sample exactly 5 examples for training
    train_df = df.sample(n=n_train, random_state=42)
    remaining_df = df.drop(labels=train_df.index.tolist())

    # Use remaining data for validation and test
    if len(remaining_df) > 0:
        val_df = remaining_df.sample(n=min(n_val, len(remaining_df)), random_state=42)
        test_df = remaining_df.drop(labels=val_df.index.tolist())
    else:
        val_df = pd.DataFrame(columns=df.columns)
        test_df = pd.DataFrame(columns=df.columns)

    return train_df, val_df, test_df


# ------------------ Enhanced Data Preprocessing ------------------
def preprocess_csv_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).apply(enhanced_normalize)
    df = df.dropna()
    if len(df.columns) >= 2:
        input_col_name = str(df.columns[0])
        output_col_name = str(df.columns[1])
        expanded_rows = []
        for _, row in df.iterrows():
            input_val = str(row[input_col_name])
            output_vals = str(row[output_col_name]).split(',')
            for output_val in output_vals:
                output_val = output_val.strip()
                if output_val and output_val != 'nan':
                    expanded_rows.append({
                        input_col_name: input_val,
                        output_col_name: output_val
                    })
        if expanded_rows:
            df = pd.DataFrame(expanded_rows)
    return df


# ------------------ Core Transformation Logic ------------------
def original_transformation_discovery(cur, examples: List[Tuple[str, str]], queries: List[str], alpha: float,
                                      prior: float, max_iter: int, threshold: float) -> Dict[str, Dict[str, float]]:
    # Limit examples to prevent SQL complexity error
    max_examples = 50  # Vertica has limits on UNION ALL complexity
    if len(examples) > max_examples:
        examples = examples[:max_examples]

    try:
        values_clause = " UNION ALL ".join(
            f"SELECT '{sql_escape(x)}', '{sql_escape(y)}'" for x, y in examples
        )
        cur.execute(f"""
            WITH examples(x, y) AS ({values_clause})
            SELECT t1.tableid, t1.colid AS colX, t2.colid AS colY, COUNT(*) as cnt
            FROM main_tokenized_super t1
            JOIN main_tokenized_super t2 ON t1.tableid = t2.tableid AND t1.rowid = t2.rowid
            JOIN examples ON t1.tokenized = examples.x AND t2.tokenized = examples.y
            GROUP BY t1.tableid, t1.colid, t2.colid
            HAVING COUNT(*) >= 2;
        """)
    except Exception as e:
        if "too complex" in str(e).lower() or "54001" in str(e):
            # Fallback: use fewer examples
            if len(examples) > 20:
                examples = examples[:20]
                values_clause = " UNION ALL ".join(
                    f"SELECT '{sql_escape(x)}', '{sql_escape(y)}'" for x, y in examples
                )
                cur.execute(f"""
                    WITH examples(x, y) AS ({values_clause})
                    SELECT t1.tableid, t1.colid AS colX, t2.colid AS colY, COUNT(*) as cnt
                    FROM main_tokenized_super t1
                    JOIN main_tokenized_super t2 ON t1.tableid = t2.tableid AND t1.rowid = t2.rowid
                    JOIN examples ON t1.tokenized = examples.x AND t2.tokenized = examples.y
                    GROUP BY t1.tableid, t1.colid, t2.colid
                    HAVING COUNT(*) >= 2;
                """)
            else:
                # If still too complex, return empty results
                return {xq: {} for xq in queries}
        else:
            raise e

    candidates = {}
    for tid, cx, cy, cnt in cur.fetchall():
        candidates[(tid, cx, cy)] = cnt

    if not candidates:
        return {xq: {} for xq in queries}

    mapping_support = {t_key: defaultdict(set) for t_key in candidates}

    if queries:
        # Process queries in batches to avoid complexity
        batch_size = 100
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            placeholders = ','.join(['%s'] * len(batch_queries))
            for (tid, cx, cy) in candidates:
                try:
                    cur.execute(f"""
                        SELECT t1.tokenized, t2.tokenized
                        FROM main_tokenized_super t1
                        JOIN main_tokenized_super t2 ON t1.tableid = t2.tableid AND t1.rowid = t2.rowid
                        WHERE t1.tableid = %s AND t1.colid = %s AND t2.colid = %s
                          AND t1.tokenized IN ({placeholders})
                    """, (tid, cx, cy, *batch_queries))

                    for xq, y in cur.fetchall():
                        mapping_support[(tid, cx, cy)][normalize(xq)].add(normalize(y))
                except Exception as e:
                    # Skip this batch if it causes complexity issues
                    continue

    table_score = {}
    total_examples = len(examples)
    for t_key, cnt in candidates.items():
        table_score[t_key] = cnt / total_examples

    answers = {xq: {} for xq in queries}

    for xq in queries:
        freq = defaultdict(float)
        for t_key in mapping_support:
            for y in mapping_support[t_key].get(xq, []):
                freq[y] += table_score[t_key]
        total = sum(freq.values())
        if total > 0:
            for y, f in freq.items():
                answers[xq][y] = f / total

    prev_scores = table_score.copy()

    for i in range(max_iter):
        for t_key in table_score:
            good, bad = 0.0, 0.0
            total_weight = 0.0

            for xq, ys in mapping_support[t_key].items():
                if not answers.get(xq) or not answers[xq]:
                    continue

                best_y = max(answers[xq].items(), key=lambda kv: kv[1])[0] if answers[xq] else None
                if best_y is None:
                    continue

                for y in ys:
                    weight = answers[xq].get(y, 0)
                    total_weight += weight
                    if y == best_y:
                        good += weight
                    else:
                        bad += weight

            if total_weight > 0:
                num = prior * good
                den = num + (1 - prior) * bad
                table_score[t_key] = alpha * (num / den if den > 0 else 0.0)

        for xq in answers:
            relevant_transforms = [t for t in table_score if xq in mapping_support[t]]
            all_ys_for_xq = set.union(
                *(mapping_support[t][xq] for t in relevant_transforms)) if relevant_transforms else set()

            for y in all_ys_for_xq:
                p_y = 1.0
                for t in relevant_transforms:
                    if y in mapping_support[t][xq]:
                        p_y *= table_score[t]
                    else:
                        p_y *= (1 - table_score[t])
                answers[xq][y] = p_y

            total_prob = sum(answers[xq].values())
            if total_prob > 0:
                for y in answers[xq]:
                    answers[xq][y] /= total_prob

        delta = max((abs(table_score[t] - prev_scores[t]) for t in table_score), default=0)
        if delta < threshold:
            break

        prev_scores = table_score.copy()

    return answers


# ------------------ Advanced Evaluation Metrics ------------------
def calculate_advanced_metrics(predictions: Dict[str, Dict[str, float]], ground_truth: Dict[str, str],
                               fuzzy_threshold: int = 85) -> Dict[str, float]:
    metrics = {}
    total_queries = len(ground_truth)
    correct_predictions = 0
    high_confidence_correct = 0
    high_confidence_total = 0
    precision_scores = []
    recall_scores = []
    for xq, expected in ground_truth.items():
        if xq not in predictions or not predictions[xq]:
            continue
        best_y, confidence = max(predictions[xq].items(), key=lambda kv: kv[1])
        is_correct = fuzzy_match(best_y, expected, fuzzy_threshold)
        if is_correct:
            correct_predictions += 1
        if confidence > 0.7:
            high_confidence_total += 1
            if is_correct:
                high_confidence_correct += 1
        if is_correct:
            precision_scores.append(confidence)
            recall_scores.append(1.0)
        else:
            precision_scores.append(0.0)
            recall_scores.append(0.0)
    metrics['accuracy'] = correct_predictions / total_queries if total_queries > 0 else 0
    metrics[
        'high_confidence_accuracy'] = high_confidence_correct / high_confidence_total if high_confidence_total > 0 else 0
    metrics['avg_precision'] = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    metrics['avg_recall'] = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    if metrics['avg_precision'] + metrics['avg_recall'] > 0:
        metrics['f1_score'] = 2 * (metrics['avg_precision'] * metrics['avg_recall']) / (
                    metrics['avg_precision'] + metrics['avg_recall'])
    else:
        metrics['f1_score'] = 0
    return metrics


def confidence_analysis(predictions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    confidences = []
    high_conf_predictions = 0
    total_predictions = 0
    for xq, y_scores in predictions.items():
        if not y_scores:
            continue
        best_y, confidence = max(y_scores.items(), key=lambda kv: kv[1])
        confidences.append(confidence)
        total_predictions += 1
        if confidence > 0.7:
            high_conf_predictions += 1
    return {
        'avg_confidence': float(sum(confidences) / len(confidences) if confidences else 0),
        'high_confidence_rate': float(high_conf_predictions / total_predictions if total_predictions > 0 else 0),
        'confidence_std': float(np.std(confidences) if len(confidences) > 1 else 0)
    }


# ------------------ Streamlit UI ------------------
tabs = st.tabs(["Manual Mode", "CSV Mode", "Batch CSV Test"])

# ------------------ Manual Mode Tab ------------------
with tabs[0]:
    st.subheader("Manual Transformation Mode")
    st.markdown("Define example input-output pairs and apply the transformation to your own queries.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 1. Enter 5 Examples")
        examples_df = st.data_editor(pd.DataFrame({
            "Input (x)": ["", "", "", "", ""],
            "Output (y)": ["", "", "", "", ""]
        }), num_rows="fixed", hide_index=True, key="examples_manual")
    with col2:
        st.markdown("##### 2. List Your Queries")
        queries_df = st.data_editor(pd.DataFrame({"Query (xq)": [""] * 5}), num_rows="dynamic", hide_index=True,
                                    key="queries_manual")
    st.markdown("##### 3. Configure Parameters")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        alpha = st.slider("Alpha (confidence weight)", 0.0, 1.0, 0.99, key="alpha_manual")
    with p_col2:
        prior = st.slider("Prior (prior weight)", 0.0, 1.0, 0.5, key="prior_manual")
    with p_col3:
        max_iter = st.slider("Max Iterations", 1, 20, 5, key="iter_manual")
    with p_col4:
        threshold = st.slider("Convergence Threshold", 0.0, 0.1, 0.01, 0.005, key="threshold_manual")
    if st.button("Run Transformation", type="primary"):
        examples = [(normalize(x), normalize(y)) for x, y in zip(examples_df['Input (x)'], examples_df['Output (y)']) if
                    x and y]
        queries = [normalize(x) for x in queries_df['Query (xq)'] if x]
        if len(examples) != 5 or not queries:
            st.error("Please enter exactly 5 example pairs and at least 1 query.")
        else:
            try:
                conn = get_vertica_connection()
                if conn:
                    with conn.cursor() as cur:
                        with st.spinner("Finding transformations and making predictions..."):
                            answers = original_transformation_discovery(cur, examples, queries, alpha, prior, max_iter,
                                                                        threshold)
                        rows = []
                        for xq, y_scores in answers.items():
                            if not y_scores:
                                rows.append((xq.upper(), "No prediction", 0.0))
                                continue
                            topk = sorted(y_scores.items(), key=lambda kv: -kv[1])[:3]
                            for y, sc in topk:
                                rows.append((xq.upper(), y.title(), round(sc, 3)))
                        if not rows:
                            st.warning("No results found.")
                        else:
                            df_results = pd.DataFrame(rows, columns=pd.Index(["Input (X)", "Output (Y)", "Score"]))
                            st.session_state["manual_df_result"] = df_results
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    if "manual_df_result" in st.session_state and not st.session_state["manual_df_result"].empty:
        st.success("Top predictions:")
        st.dataframe(st.session_state["manual_df_result"])
        st.markdown("---")
        st.subheader("🤖 Ollama Semantic Analysis (Manual Mode)")
        st.markdown(
            "Check if predicted results are semantically equivalent to your intended output. Make sure Ollama is running locally at `localhost:11434`.")

        manual_df = st.session_state.get("manual_df_result")

        if manual_df is not None and not manual_df.empty:
            # Only allow analysis for rows with a prediction
            predictions = manual_df[manual_df["Output (Y)"] != "No prediction"]
            unique_predictions = predictions.drop_duplicates(subset=["Input (X)"])

            if not predictions.empty:
                st.info(f"Found {len(unique_predictions)} predictions for semantic analysis.")

                # Model selection
                # Ollama model selection
                ollama_models = get_ollama_model_list()
                if not ollama_models:
                    # fallback to static list
                    ollama_models = [
                        "deepseek-r1", "gemma3n", "gemma3", "qwen3", "qwen2.5vl", "llama3.1",
                        "nomic-embed-text", "llama3.2", "mistral", "qwen2.5", "llama3", "llava",
                        "phi3", "gemma2", "qwen2.5-coder", "gemma", "qwen", "mxbai-embed-large",
                        "qwen2", "llama2", "phi4", "minicpm-v", "codellama", "tinyllama",
                        "llama3.3", "command-r", "command-r-plus", "command-r-v2",
                        "command-r-plus-q", "command-r-plus-vision", "command-r7b-arabic"
                    ]

                ollama_model = st.selectbox(
                    "Select Ollama Model:",
                    ollama_models,
                    index=0,
                    help="Choose the Ollama model for semantic analysis"
                )

                if st.button("🔍 Run Semantic Analysis", type="primary", key="semantic_manual_btn"):
                    with st.spinner("Analyzing predictions with Ollama..."):
                        semantic_results = []
                        unique_predictions = predictions.drop_duplicates(subset=["Input (X)"])

                        for idx, row in unique_predictions.iterrows():
                            input_val = row["Input (X)"]
                            predicted = row["Output (Y)"]
                            # Ask user for expected value (manual mode doesn't know ground truth)
                            expected = st.text_input(f"Expected output for input '{input_val}':",
                                                     key=f"expected_{idx}")

                            if expected:
                                is_match = ollama_semantic_check(predicted, expected, ollama_model)
                                semantic_results.append({
                                    "Input": input_val,
                                    "Predicted": predicted,
                                    "Expected": expected,
                                    "Semantic Match": "✅ Yes" if is_match else "❌ No"
                                })

                        if semantic_results:
                            semantic_df = pd.DataFrame(semantic_results)
                            st.success("Semantic analysis complete!")
                            st.dataframe(semantic_df, use_container_width=True)

                            st.download_button(
                                "📥 Download Semantic Analysis as CSV",
                                semantic_df.to_csv(index=False).encode("utf-8"),
                                "manual_semantic_analysis.csv",
                                use_container_width=True
                            )
            else:
                st.success("No predictions available for semantic checking.")

# ------------------ CSV Mode (Adaptive) Tab ------------------
with tabs[1]:
    st.subheader("CSV Mode")
    st.markdown("Upload a CSV file to automatically test predictions.")
    csv_file = st.file_uploader("Upload a CSV file (at least 2 columns required):", type="csv")
    if csv_file:
        try:
            df = pd.read_csv(csv_file, encoding="utf-8")
        except Exception:
            try:
                csv_file.seek(0)
                df = pd.read_csv(csv_file, encoding="ISO-8859-1")
            except Exception as e:
                st.error(
                    f"CSV file could not be read. Please ensure it is a valid UTF-8 or ISO-8859-1 encoded file. Error: {e}")
                st.stop()
        if len(df.columns) < 2:
            st.warning("CSV must have at least two columns.")
            st.stop()
        st.info(f"CSV loaded: {len(df)} rows.")
        df = preprocess_csv_data(df)
        st.info(f"After preprocessing: {len(df)} rows.")
        st.markdown("##### 1. Select Columns and Parameters")
        col1, col2 = st.columns(2)
        with col1:
            input_col = st.selectbox("Select Input Column (x):", df.columns, index=0)
            st.caption("Column containing your input data")
        with col2:
            output_col = st.selectbox("Select Output Column (y):", df.columns, index=1)
            st.caption("Column containing your expected output data")

        st.markdown("##### 2. Algorithm Parameters")
        param_col1, param_col2, param_col3, param_col4 = st.columns(4)

        with param_col1:
            alpha_csv = st.slider("Alpha (confidence weight)", 0.0, 1.0, 0.99, 0.01, key="alpha_csv")
            st.caption("Higher = more confident predictions")

        with param_col2:
            prior_csv = st.slider("Prior (prior weight)", 0.0, 1.0, 0.5, 0.01, key="prior_csv")
            st.caption("Higher = assume transformations likely")

        with param_col3:
            max_iter_csv = st.slider("Max Iterations", 1, 20, 10, key="iter_csv")
            st.caption("More iterations = slower but more refined")

        with param_col4:
            convergence_threshold = st.slider("Convergence Threshold", 0.001, 0.1, 0.01, 0.001, key="conv_csv")
            st.caption("Lower = more thorough optimization")

        st.markdown("##### 3. Evaluation Parameters")
        eval_col1, eval_col2 = st.columns(2)

        with eval_col1:
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, key="confidence_csv")
            st.caption("Minimum confidence for 'Correct' status")

        with eval_col2:
            fuzzy_threshold = st.slider("Fuzzy Match Threshold", 50, 100, 85, 5, key="fuzzy_csv")
            st.caption("String similarity threshold (higher = stricter)")
        if input_col == output_col:
            st.error("Input and Output columns cannot be the same.")
            st.stop()
        df = df[[input_col, output_col]].dropna().astype(str)
        df.columns = ["Input (x)", "Output (y)"]
        # Fix: Ensure df is a DataFrame
        if isinstance(df, pd.Series):
            df = df.to_frame().T
        train_df, val_df, test_df = adaptive_data_split(df)
        if len(train_df) < 5:
            st.warning(f"⚠️ Small dataset: Using {len(train_df)} training examples (less than 5 available)")
        if st.button("Start Prediction", type="primary"):
            examples = [(normalize(x), normalize(y)) for x, y in train_df.values]
            queries = [normalize(x) for x in test_df["Input (x)"]]
            ground_truth = {normalize(x): normalize(y) for x, y in test_df.values}
            try:
                conn = get_vertica_connection()
                if conn:
                    with conn.cursor() as cur:
                        with st.spinner("Finding transformations and making predictions..."):
                            start_time = time.time()
                            answers = original_transformation_discovery(cur, examples, queries, alpha_csv, prior_csv,
                                                                        max_iter_csv, convergence_threshold)
                            query_time = time.time() - start_time
                        results = []
                        for xq in queries:
                            best_y, score = "No prediction", 0.0
                            if answers.get(xq):
                                best_y, score = max(answers[xq].items(), key=lambda kv: kv[1],
                                                    default=("No prediction", 0.0))
                            expected = ground_truth.get(xq, "")
                            is_correct = fuzzy_match(best_y, expected, fuzzy_threshold)
                            status = "No prediction"
                            if best_y != "No prediction":
                                if score < confidence_threshold:
                                    status = "Uncertain"
                                else:
                                    status = "Correct" if is_correct else "Incorrect"
                            results.append({
                                "Input": xq,
                                "Predicted": best_y,
                                "Expected": expected,
                                "Score": round(score, 3),
                                "Status": status
                            })
                        st.session_state["df_result"] = pd.DataFrame(results)
                        st.success(f"Prediction complete! {len(results)} queries processed.")
                        if len(queries) > 0:
                            st.subheader("📊 Advanced Evaluation Metrics")
                            advanced_metrics = calculate_advanced_metrics(answers, ground_truth, fuzzy_threshold)
                            confidence_metrics = confidence_analysis(answers)

                            # Store metrics in session state to persist them
                            st.session_state["advanced_metrics"] = advanced_metrics
                            st.session_state["confidence_metrics"] = confidence_metrics
                            st.session_state["query_time"] = query_time

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Accuracy", f"{advanced_metrics['accuracy']:.3f}")
                                st.metric("F1-Score", f"{advanced_metrics['f1_score']:.3f}")
                                st.metric("Precision", f"{advanced_metrics['avg_precision']:.3f}")
                            with col2:
                                st.metric("Recall", f"{advanced_metrics['avg_recall']:.3f}")
                                st.metric("Query Run Time", f"{query_time:.2f}s")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.session_state["df_result"] = None
        if st.session_state.get("df_result") is not None:
            st.markdown("---")
            st.markdown("##### 2. Review Predictions")
            st.dataframe(st.session_state["df_result"])

            # Display stored metrics if they exist
            if st.session_state.get("advanced_metrics") is not None:
                st.markdown("---")
                st.markdown("##### 📊 Evaluation Metrics")
                advanced_metrics = st.session_state["advanced_metrics"]
                confidence_metrics = st.session_state.get("confidence_metrics", {})
                query_time = st.session_state.get("query_time", 0)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{advanced_metrics['accuracy']:.3f}")
                    st.metric("F1-Score", f"{advanced_metrics['f1_score']:.3f}")
                    st.metric("Precision", f"{advanced_metrics['avg_precision']:.3f}")
                with col2:
                    st.metric("Recall", f"{advanced_metrics['avg_recall']:.3f}")
                    st.metric("Query Run Time", f"{query_time:.2f}s")

            # Ollama Semantic Checking Section
            st.markdown("---")
            st.markdown("##### 4. 🤖 Ollama Semantic Analysis")
            st.markdown("Check if incorrect predictions are semantically equivalent using Ollama. You need to have Ollama installed and run to run semantic analysis.")

            # Get incorrect predictions
            df_result = st.session_state["df_result"]
            incorrect_predictions = df_result[df_result["Status"] == "Incorrect"]

            if len(incorrect_predictions) > 0:
                st.info(f"Found {len(incorrect_predictions)} incorrect predictions to analyze")

                # Ollama model selection
                ollama_models = get_ollama_model_list()
                if not ollama_models:
                    # fallback to static list
                    ollama_models = [
                    "deepseek-r1", "gemma3n", "gemma3", "qwen3", "qwen2.5vl", "llama3.1",
                    "nomic-embed-text", "llama3.2", "mistral", "qwen2.5", "llama3", "llava",
                    "phi3", "gemma2", "qwen2.5-coder", "gemma", "qwen", "mxbai-embed-large",
                    "qwen2", "llama2", "phi4", "minicpm-v", "codellama", "tinyllama",
                    "llama3.3", "command-r", "command-r-plus", "command-r-v2",
                    "command-r-plus-q", "command-r-plus-vision", "command-r7b-arabic"
                ]

                ollama_model = st.selectbox(
                    "Select Ollama Model:",
                    ollama_models,
                    index=0,
                    help="Choose the Ollama model for semantic analysis"
                )

                if st.button("🔍 Run Semantic Analysis", type="primary"):
                    with st.spinner("Analyzing incorrect predictions with Ollama..."):
                        semantic_results = []

                        for idx, row in incorrect_predictions.iterrows():
                            predicted = row["Predicted"]
                            expected = row["Expected"]

                            # Check if semantically equivalent
                            is_semantic_match = ollama_semantic_check(predicted, expected, ollama_model)

                            semantic_results.append({
                                "Input": row["Input"],
                                "Predicted": predicted,
                                "Expected": expected,
                                "Semantic Match": "✅ Yes" if is_semantic_match else "❌ No",
                                "Original Status": row["Status"]
                            })

                            # Add a small delay to avoid overwhelming Ollama
                            time.sleep(0.1)

                        # Display semantic analysis results
                        if semantic_results:
                            semantic_df = pd.DataFrame(semantic_results)
                            st.success("Semantic analysis completed!")
                            st.dataframe(semantic_df, use_container_width=True)

                            # Calculate semantic accuracy
                            semantic_matches = sum(1 for r in semantic_results if "✅ Yes" in r["Semantic Match"])
                            semantic_accuracy = semantic_matches / len(semantic_results) if semantic_results else 0

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Semantic Matches", f"{semantic_matches}/{len(semantic_results)}")
                            with col2:
                                st.metric("Semantic Accuracy", f"{semantic_accuracy:.1%}")

                            # Download semantic analysis results
                            st.download_button(
                                "📥 Download Semantic Analysis as CSV",
                                semantic_df.to_csv(index=False).encode("utf-8"),
                                "semantic_analysis.csv",
                                use_container_width=True
                            )

                            # Update results with semantic analysis
                            st.markdown("---")
                            st.markdown("##### 5. Update Results with Semantic Analysis")
                            st.markdown(
                                "Update the original results to reflect semantic matches as correct predictions.")

                            if st.button("🔄 Update Results", type="primary"):
                                # Create a copy of the original results
                                updated_df = df_result.copy()

                                # Update status for semantic matches
                                semantic_correct_count = 0
                                for idx, row in semantic_df.iterrows():
                                    if "✅ Yes" in row["Semantic Match"]:
                                        # Find the corresponding row in the original results
                                        original_idx = updated_df[updated_df["Input"] == row["Input"]].index
                                        if len(original_idx) > 0:
                                            updated_df.loc[original_idx[0], "Status"] = "Correct (Semantic)"
                                            semantic_correct_count += 1

                                # Update session state with updated results
                                st.session_state["df_result"] = updated_df

                                # Recalculate metrics
                                total_predictions = len(updated_df)
                                correct_predictions = len(
                                    updated_df[updated_df["Status"].isin(["Correct", "Correct (Semantic)"])])
                                updated_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

                                st.success(f"✅ Updated {semantic_correct_count} predictions as semantically correct!")

                                # Display updated metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Updated Accuracy", f"{updated_accuracy:.1%}")
                                with col2:
                                    st.metric("Correct Predictions", f"{correct_predictions}/{total_predictions}")
                                with col3:
                                    st.metric("Semantic Corrections", semantic_correct_count)

                                # Show updated results
                                st.markdown("**Updated Results:**")
                                st.dataframe(updated_df, use_container_width=True)

                                # Updated download button
                                st.download_button(
                                    "📥 Download Updated Results as CSV",
                                    updated_df.to_csv(index=False).encode("utf-8"),
                                    "updated_results.csv",
                                    use_container_width=True
                                )
            else:
                st.success("🎉 No incorrect predictions found! All predictions are correct.")

            st.markdown("---")
            st.markdown("##### 6. Download Results")
            st.download_button(
                "📥 Download Results as CSV",
                st.session_state["df_result"].to_csv(index=False).encode("utf-8"),
                "results.csv",
                use_container_width=True
            )

# ------------------ Batch CSV Test (Enhanced) Tab ------------------
with tabs[2]:
    st.subheader("Batch Test Selected CSV Files")
    st.markdown("This will test selected CSV files using the DataXFormer approach with comprehensive metrics.")

    # =============================================================================
    # PARAMETER CONFIGURATION
    # =============================================================================

    st.markdown("##### Algorithm Parameters")
    param_col1, param_col2, param_col3, param_col4 = st.columns(4)

    with param_col1:
        alpha = st.slider("Alpha (confidence weight)", 0.0, 1.0, 0.99, 0.01, key="alpha_batch")
        st.caption("Higher = more confident predictions")

    with param_col2:
        prior = st.slider("Prior (prior weight)", 0.0, 1.0, 0.5, 0.01, key="prior_batch")
        st.caption("Higher = assume transformations likely")

    with param_col3:
        max_iter = st.slider("Max Iterations", 1, 20, 10, key="iter_batch")
        st.caption("More iterations = slower but more refined")

    with param_col4:
        convergence_threshold = st.slider("Convergence Threshold", 0.001, 0.1, 0.01, 0.001, key="conv_batch")
        st.caption("Lower = more thorough optimization")

    st.markdown("##### Evaluation Parameters")
    eval_col1, eval_col2 = st.columns(2)

    with eval_col1:
        fuzzy_threshold = st.slider("Fuzzy Match Threshold", 50, 100, 85, 5, key="fuzzy_batch")
        st.caption("String similarity threshold (higher = stricter)")

    with eval_col2:
        st.markdown("")
        st.markdown("")
        st.info("📁 Select files below")

    # =============================================================================
    # CSV FILE SELECTION
    # =============================================================================

    st.markdown("##### CSV File Selection")
    st.markdown("Upload multiple CSV files to test using the DataXFormer approach.")

    # File uploader for multiple CSV files
    uploaded_files = st.file_uploader(
        "📁 Upload CSV files for testing:",
        type=['csv'],
        accept_multiple_files=True,
        help="Select multiple CSV files to test. Each file should have at least 2 columns (input and output)."
    )

    # =============================================================================
    # FILE VALIDATION
    # =============================================================================

    # Validation
    if not uploaded_files:
        st.warning("⚠️ No CSV files uploaded for testing.")
        st.info("Please upload CSV files to begin testing.")
        st.stop()

    st.success(f"✅ Selected {len(uploaded_files)} CSV files for testing")

    # Display file information
    st.info(f"📊 Total files selected: {len(uploaded_files)}")

    # Show selected files
    if uploaded_files:
        st.markdown("**Selected files:**")
        for file in uploaded_files:
            st.write(f"📄 {file.name}")


    # =============================================================================
    # BATCH TESTING FUNCTION
    # =============================================================================

    def test_single_csv(csv_path: str, alpha: float, prior: float, max_iter: int, convergence_threshold: float,
                        fuzzy_threshold: int) -> Dict[str, Any]:
        """
        Tests a single CSV file using the DataXFormer approach.

        This function:
        1. Reads and preprocesses the CSV file
        2. Splits data into training/validation/test sets
        3. Runs transformation discovery
        4. Evaluates predictions against ground truth
        5. Returns comprehensive metrics

        Args:
            csv_path: Path to the CSV file to test
            alpha: Confidence weight parameter
            prior: Prior probability weight
            max_iter: Maximum iterations for convergence
            convergence_threshold: Convergence threshold
            fuzzy_threshold: String similarity threshold

        Returns:
            Dictionary containing test results and metrics
        """
        try:
            # Step 1: Read CSV file with error handling for different encodings
            try:
                df = pd.read_csv(csv_path, encoding="utf-8")
            except Exception:
                try:
                    df = pd.read_csv(csv_path, encoding="ISO-8859-1")
                except Exception as e:
                    return {
                        "file": os.path.basename(csv_path),
                        "status": f"Read Error: {e}",
                        "accuracy": 0,
                        "f1_score": 0,
                        "precision": 0,
                        "recall": 0,
                        "query_time": 0,
                        "train_size": 0,
                        "test_size": 0,
                        "total_size": 0
                    }

            # Step 2: Validate data structure
            if len(df.columns) < 2:
                return {
                    "file": os.path.basename(csv_path),
                    "status": "Insufficient columns",
                    "accuracy": 0,
                    "f1_score": 0,
                    "precision": 0,
                    "recall": 0,
                    "query_time": 0,
                    "train_size": 0,
                    "test_size": 0,
                    "total_size": len(df)
                }

            # Step 3: Preprocess data
            df = preprocess_csv_data(df)
            df = df.iloc[:, :2].dropna().astype(str)  # Take first two columns only
            df.columns = ["Input (x)", "Output (y)"]

            # Step 4: Split data into training/validation/test sets
            train_df, val_df, test_df = adaptive_data_split(df)

            if len(test_df) == 0:
                return {
                    "file": os.path.basename(csv_path),
                    "status": "No test data after splitting",
                    "accuracy": 0,
                    "f1_score": 0,
                    "precision": 0,
                    "recall": 0,
                    "query_time": 0,
                    "train_size": len(train_df),
                    "test_size": 0,
                    "total_size": len(df)
                }

            # Step 5: Prepare data for transformation discovery
            examples = [(normalize(x), normalize(y)) for x, y in train_df.values]
            queries = [normalize(x) for x in test_df["Input (x)"]]
            ground_truth = {normalize(x): normalize(y) for x, y in test_df.values}

            # Step 6: Run transformation discovery
            conn = get_vertica_connection()
            if not conn:
                return {
                    "file": os.path.basename(csv_path),
                    "status": "Database connection failed",
                    "accuracy": 0,
                    "f1_score": 0,
                    "precision": 0,
                    "recall": 0,
                    "query_time": 0,
                    "train_size": len(train_df),
                    "test_size": len(test_df),
                    "total_size": len(df)
                }

            # Execute transformation discovery and measure time
            with conn.cursor() as cur:
                start_time = time.time()
                answers = original_transformation_discovery(cur, examples, queries, alpha, prior, max_iter,
                                                            convergence_threshold)
                query_time = time.time() - start_time

            # Step 7: Calculate evaluation metrics
            metrics = calculate_advanced_metrics(answers, ground_truth, fuzzy_threshold)

            # Step 8: Return comprehensive results
            return {
                "file": os.path.basename(csv_path),
                "status": "OK",
                "accuracy": metrics['accuracy'],
                "f1_score": metrics['f1_score'],
                "precision": metrics['avg_precision'],
                "recall": metrics['avg_recall'],
                "query_time": query_time,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "total_size": len(df)
            }

        except Exception as e:
            return {
                "file": os.path.basename(csv_path),
                "status": f"Error: {e}",
                "accuracy": 0,
                "f1_score": 0,
                "precision": 0,
                "recall": 0,
                "query_time": 0,
                "train_size": 0,
                "test_size": 0,
                "total_size": 0
            }


    # =============================================================================
    # BATCH TESTING EXECUTION
    # =============================================================================

    # Run batch test when button is clicked
    if st.button("🚀 Run Batch Test", type="primary"):
        st.markdown("---")
        st.subheader("📊 Batch Test Results")

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        total_files = len(uploaded_files)

        # Process each uploaded CSV file
        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            status_text.text(f"Testing {filename} ({i + 1}/{total_files})")

            # Create a temporary file path for uploaded files
            import tempfile

            with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Test the uploaded file
            result = test_single_csv(tmp_file_path, alpha, prior, max_iter, convergence_threshold, fuzzy_threshold)

            # Clean up temporary file
            os.unlink(tmp_file_path)

            results.append(result)

            # Update progress bar
            progress_bar.progress((i + 1) / total_files)

        status_text.text("✅ Batch test completed!")

        # Create results DataFrame for display
        df_results = pd.DataFrame(results)

        # Display comprehensive results table
        st.dataframe(df_results, use_container_width=True)

        # =============================================================================
        # SUMMARY STATISTICS
        # =============================================================================

        st.markdown("---")
        st.subheader("📈 Summary Statistics")

        # Filter successful results for statistics
        successful_results = df_results[df_results['status'] == 'OK']

        if len(successful_results) > 0:
            # Display key metrics in columns
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                avg_accuracy = successful_results['accuracy'].mean()
                st.metric("Average Accuracy", f"{avg_accuracy:.3f}")

            with col2:
                avg_f1 = successful_results['f1_score'].mean()
                st.metric("Average F1-Score", f"{avg_f1:.3f}")

            with col3:
                avg_precision = successful_results['precision'].mean()
                st.metric("Average Precision", f"{avg_precision:.3f}")

            with col4:
                avg_recall = successful_results['recall'].mean()
                st.metric("Average Recall", f"{avg_recall:.3f}")

            # Additional performance metrics
            col5, col6, col7 = st.columns(3)

            with col5:
                total_time = successful_results['query_time'].sum()
                st.metric("Total Query Time", f"{total_time:.2f}s")

            with col6:
                avg_time = successful_results['query_time'].mean()
                st.metric("Average Query Time", f"{avg_time:.2f}s")

            with col7:
                total_files_tested = len(successful_results)
                st.metric("Files Successfully Tested", f"{total_files_tested}/{total_files}")

            # =============================================================================
            # TOP AND BOTTOM PERFORMERS
            # =============================================================================

            st.markdown("---")
            st.subheader("🏆 Top Performers")

            # Show top 3 performing files
            top_3 = successful_results.nlargest(3, 'accuracy')[['file', 'accuracy', 'f1_score', 'precision', 'recall']]
            st.dataframe(top_3, use_container_width=True)

            st.markdown("---")
            st.subheader("📉 Bottom Performers")

            # Show bottom 3 performing files
            bottom_3 = successful_results.nsmallest(3, 'accuracy')[
                ['file', 'accuracy', 'f1_score', 'precision', 'recall']]
            st.dataframe(bottom_3, use_container_width=True)

        else:
            st.warning("⚠️ No files were successfully processed.")

        # =============================================================================
        # RESULTS DOWNLOAD
        # =============================================================================

        st.markdown("---")
        st.subheader("📥 Download Results")

        # Create downloadable CSV with all results
        csv_data = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Full Results as CSV",
            csv_data,
            "batch_test_results.csv",
            use_container_width=True
        )
