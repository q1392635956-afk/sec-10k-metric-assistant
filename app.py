"""
app.py
------
Streamlit UI for the SEC 10-K Metric Assistant.

The app lets a financial analyst ask questions about Apple's FY2025 10-K in
two modes:
  1. Metric-aware system — retrieval + LLM extraction + Python computation
  2. Baseline — direct LLM answer, no retrieval or formula

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

# Load .env for local development (no-op if the var is already in the environment)
load_dotenv()

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SEC 10-K Metric Assistant",
    page_icon="📊",
    layout="centered",
)

st.title("📊 SEC 10-K Metric Assistant")
st.caption(
    "Compute derived financial metrics from Apple's FY2025 Form 10-K "
    "with cited, auditable evidence."
)

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if not os.environ.get("GEMINI_API_KEY"):
    st.error(
        "**GEMINI_API_KEY is not set.**\n\n"
        "Copy `.env.example` to `.env`, add your Google AI Studio key, then restart:\n"
        "```\nstreamlit run app.py\n```\n\n"
        "Get a free key at https://aistudio.google.com/app/apikey"
    )
    st.stop()

if not os.path.exists("data/apple_2025_10k.txt"):
    st.error(
        "**10-K text file not found** at `data/apple_2025_10k.txt`.\n\n"
        "See `data/README.md` for instructions on how to obtain and place the file."
    )
    st.stop()

# Imports are after the pre-flight checks so missing keys / files surface cleanly.
from retriever import build_or_load_index, retrieve          # noqa: E402
from llm_utils import classify_metric, extract_values, format_answer  # noqa: E402
from metric_engine import compute_metric, get_metric_info    # noqa: E402
from baseline import baseline_answer                          # noqa: E402

# ---------------------------------------------------------------------------
# Sidebar — context and supported metrics
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "This tool helps financial analysts compute and verify derived metrics "
        "from Apple's FY2025 annual 10-K filing.\n\n"
        "**Supported metrics:**\n"
        "- Gross Margin\n"
        "- Operating Margin\n"
        "- Net Profit Margin\n"
        "- Current Ratio\n"
        "- R&D Expense YoY Growth\n\n"
        "**How it works (metric-aware mode):**\n"
        "1. Classify your question → metric key\n"
        "2. Retrieve relevant 10-K passages\n"
        "3. Extract numeric values with LLM\n"
        "4. Compute result with Python\n"
        "5. Generate cited explanation"
    )
    st.divider()
    st.markdown(
        "**Human review note:** Always verify extracted values against "
        "the original 10-K before using in a report or decision."
    )

# ---------------------------------------------------------------------------
# Main input area
# ---------------------------------------------------------------------------
mode = st.radio(
    "Select mode:",
    ["Metric-aware system", "Baseline (direct LLM, no retrieval)"],
    horizontal=True,
)

st.markdown("**Example questions:**")
examples = [
    "What is Apple's gross margin for FY2025?",
    "What was Apple's operating margin in fiscal year 2025?",
    "What is Apple's current ratio?",
    "How much did Apple's R&D expenses grow year over year?",
    "What is Apple's net profit margin for FY2025?",
]
for ex in examples:
    if st.button(ex, key=ex, use_container_width=True):
        st.session_state["question_input"] = ex

question = st.text_input(
    "Or type your own question:",
    key="question_input",
    placeholder="e.g. What is Apple's gross margin for FY2025?",
)

run_button = st.button("Compute", type="primary", disabled=not bool(question))

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
if run_button and question:
    with st.spinner("Processing your question..."):

        # ---- Baseline mode ----
        if "Baseline" in mode:
            st.subheader("Baseline Answer")
            st.info(
                "Baseline mode: the LLM answers from its training knowledge only — "
                "no retrieval, no formula, no Python computation."
            )
            try:
                answer = baseline_answer(question)
                st.write(answer)
            except Exception as e:
                st.error(f"Baseline error: {e}")

        # ---- Metric-aware mode ----
        else:
            # Step 1: Classify
            with st.status("Step 1 — Classifying metric...", expanded=True) as status:
                try:
                    metric_key = classify_metric(question)
                except Exception as e:
                    st.error(f"Classification error: {e}")
                    st.stop()

                if metric_key:
                    st.write(f"Detected metric: **`{metric_key}`**")
                    status.update(label=f"Step 1 — Metric: `{metric_key}`", state="complete")
                else:
                    st.warning(
                        "Could not map this question to a supported metric. "
                        "Try rephrasing, or check the supported metric list in the sidebar."
                    )
                    status.update(label="Step 1 — Metric not recognized", state="error")
                    st.stop()

            # Step 2: Load index and retrieve
            with st.status("Step 2 — Retrieving evidence from 10-K...", expanded=True) as status:
                try:
                    chunks, vectorizer, matrix = build_or_load_index()
                    metric_info = get_metric_info(metric_key)
                    # Augment query with document-native search terms for better TF-IDF recall
                    retrieval_query = " ".join([
                        question,
                        metric_info["name"],
                        " ".join(metric_info.get("search_terms", [])),
                    ])
                    evidence = retrieve(retrieval_query, chunks, vectorizer, matrix, top_k=6)
                    st.write(f"Retrieved **{len(evidence)}** evidence chunks.")
                    status.update(label="Step 2 — Evidence retrieved", state="complete")
                except FileNotFoundError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.error(f"Retrieval error: {e}")
                    st.stop()

            # Step 3: Extract values
            with st.status("Step 3 — Extracting values from evidence...", expanded=True) as status:
                try:
                    extracted = extract_values(
                        metric_key, metric_info["required_fields"], evidence
                    )
                except Exception as e:
                    st.error(f"Extraction error: {e}")
                    st.stop()

                st.write("Extracted values:")
                for k, v in extracted.items():
                    label = f"`{k}`"
                    val = f"{v:,.2f}" if v is not None else "*not found*"
                    st.markdown(f"  - {label}: **{val}**")
                status.update(label="Step 3 — Values extracted", state="complete")

            # Step 4: Compute
            with st.status("Step 4 — Computing metric with Python...", expanded=True) as status:
                try:
                    result = compute_metric(metric_key, extracted)
                except ZeroDivisionError as e:
                    st.error(f"Computation error: {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"Computation error: {e}")
                    st.stop()

                if result is None:
                    missing = [
                        f for f in metric_info["required_fields"]
                        if extracted.get(f) is None
                    ]
                    st.error(
                        f"Cannot compute metric — missing values: {missing}.\n\n"
                        "The relevant numbers may not appear in the retrieved passages. "
                        "Try adding more context to your question."
                    )
                    status.update(label="Step 4 — Computation failed", state="error")
                    st.stop()

                st.write(f"Result: **{result:.4f} {metric_info['unit']}**")
                status.update(
                    label=f"Step 4 — {result:.4f} {metric_info['unit']}",
                    state="complete",
                )

            # Step 5: Format answer
            with st.status("Step 5 — Generating cited explanation...", expanded=True) as status:
                try:
                    explanation = format_answer(
                        question=question,
                        metric_name=metric_info["name"],
                        computed_value=result,
                        unit=metric_info["unit"],
                        extracted_values=extracted,
                        evidence_chunks=evidence,
                        formula=metric_info["formula"],
                    )
                except Exception as e:
                    st.error(f"Explanation error: {e}")
                    st.stop()
                status.update(label="Step 5 — Explanation ready", state="complete")

            # ---- Display results ----
            st.divider()
            st.subheader("Result")

            col1, col2 = st.columns(2)
            col1.metric(metric_info["name"], f"{result:.4f} {metric_info['unit']}")
            col2.metric("Metric key", metric_key)

            st.subheader("Explanation")
            st.write(explanation)

            st.subheader("Computation Details")
            with st.expander("Formula and extracted values"):
                st.markdown(f"**Formula:** `{metric_info['formula']}`")
                st.markdown(f"**Description:** {metric_info['description']}")
                st.markdown("**Extracted values (millions USD where applicable):**")
                for k, v in extracted.items():
                    if v is not None:
                        st.markdown(f"- `{k}` = {v:,.2f}")
                    else:
                        st.markdown(f"- `{k}` = *not found in evidence*")

            st.subheader("Source Evidence")
            with st.expander(f"View {len(evidence)} retrieved 10-K passages"):
                for i, chunk in enumerate(evidence, 1):
                    st.markdown(f"**Passage {i}**")
                    # Truncate very long chunks for readability
                    display = chunk[:900] + ("…" if len(chunk) > 900 else "")
                    st.text(display)
                    if i < len(evidence):
                        st.divider()

            st.caption(
                "Human review note: verify the extracted values above against the "
                "original 10-K filing before using this result in a report or decision."
            )
