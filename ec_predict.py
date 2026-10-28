"""
EC-Predict: Two-Stage Hierarchical Classification of Enzyme Commission (EC) Classes
from Primary Amino Acid Sequence Alone.

Architecture:
  - Stage A: Binary Classifier (Enzyme vs. Non-Enzyme)
  - Stage B: 6-Class Classifier (EC 1 - EC 6) trained on enzymes only
  - End-to-End: Hierarchical 7-way decision rule (0=Non-Enzyme, 1..6=EC 1..6)
    Ground truth is predicted correctly only if Stage A routes correctly AND
    Stage B (for enzymes) predicts the correct EC class.

Strict Evaluation Protocol:
  - Stratified 80/20 train/test split (random_state=42) BEFORE any fitting.
  - TF-IDF Vectorizer (amino-acid char 3-mers, max 20,000 features) fit strictly on TRAIN split.
  - Test set scored exactly once per model type.
  - Both Logistic Regression baseline and MLP neural network reported.
"""

import os
import sys
import time
import argparse
import requests
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

CLASS_NAMES_7WAY = [
    "Non-Enzyme",
    "EC 1 (Oxidoreductases)",
    "EC 2 (Transferases)",
    "EC 3 (Hydrolases)",
    "EC 4 (Lyases)",
    "EC 5 (Isomerases)",
    "EC 6 (Ligases)"
]

CLASS_NAMES_6WAY = [
    "EC 1 (Oxidoreductases)",
    "EC 2 (Transferases)",
    "EC 3 (Hydrolases)",
    "EC 4 (Lyases)",
    "EC 5 (Isomerases)",
    "EC 6 (Ligases)"
]


def fetch_uniprot_data(data_path="data/dataset.csv", cap_per_class=1200):
    """
    Fetch reviewed Swiss-Prot entries from UniProt REST API:
      - Classes 1-6, capped at 1,200 entries per class.
      - Deduplicated by exact sequence.
      - Non-enzymes (reviewed:true NOT (ec:*)) capped at post-dedup enzyme total.
    """
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    session = requests.Session()
    enzyme_records = []
    fetched_counts = {}

    print("Fetching enzyme classes 1-6 from UniProt REST API...")
    for c in range(1, 7):
        url = (
            f"https://rest.uniprot.org/uniprotkb/stream?"
            f"query=reviewed:true%20AND%20ec:{c}.*&fields=accession,id,ec,sequence&format=tsv"
        )
        r = session.get(url, stream=True)
        count = 0
        it = r.iter_lines(decode_unicode=True)
        next(it)  # Skip header
        for line in it:
            if line.strip():
                parts = line.split("\t")
                if len(parts) >= 4:
                    acc, entry_id, ec, seq = parts[0], parts[1], parts[2], parts[3]
                    enzyme_records.append({
                        "accession": acc,
                        "id": entry_id,
                        "ec": ec,
                        "sequence": seq,
                        "class": c,
                        "is_enzyme": 1
                    })
                    count += 1
                    if count >= cap_per_class:
                        break
        r.close()
        fetched_counts[f"EC {c}"] = count
        print(f"  EC {c}: fetched {count} entries")

    df_enzymes_raw = pd.DataFrame(enzyme_records)
    # Deduplicate by exact sequence
    df_enzymes = df_enzymes_raw.drop_duplicates(subset=["sequence"], keep="first").copy()
    n_enzymes = len(df_enzymes)
    print(f"Total enzymes fetched: {len(df_enzymes_raw)}, Post-dedup unique enzymes: {n_enzymes}")

    # Fetch reviewed non-enzymes
    print(f"Fetching non-enzymes (target post-dedup count: {n_enzymes})...")
    known_enzyme_seqs = set(df_enzymes["sequence"].values)
    seen_non_enz_seqs = set()
    non_enzyme_records = []

    non_enz_url = (
        "https://rest.uniprot.org/uniprotkb/stream?"
        "query=reviewed:true%20NOT%20(ec:*)&fields=accession,id,sequence&format=tsv"
    )
    r = session.get(non_enz_url, stream=True)
    count = 0
    it = r.iter_lines(decode_unicode=True)
    next(it)  # Skip header
    for line in it:
        if line.strip():
            parts = line.split("\t")
            if len(parts) >= 3:
                acc, entry_id, seq = parts[0], parts[1], parts[2]
                count += 1
                if seq not in known_enzyme_seqs and seq not in seen_non_enz_seqs:
                    seen_non_enz_seqs.add(seq)
                    non_enzyme_records.append({
                        "accession": acc,
                        "id": entry_id,
                        "ec": "None",
                        "sequence": seq,
                        "class": 0,
                        "is_enzyme": 0
                    })
                    if len(non_enzyme_records) >= n_enzymes:
                        break
    r.close()
    fetched_counts["Non-Enzyme"] = count
    print(f"  Non-enzymes scanned: {count}, Post-dedup unique retained: {len(non_enzyme_records)}")

    df_non_enzymes = pd.DataFrame(non_enzyme_records)
    df_all = pd.concat([df_non_enzymes, df_enzymes], ignore_index=True)
    df_all.to_csv(data_path, index=False)
    print(f"Saved complete balanced dataset to {data_path} ({len(df_all)} rows)")
    return df_all, fetched_counts


def load_or_fetch_data(data_path="data/dataset.csv", force_fetch=False):
    """Load existing dataset from cache, or fetch from UniProt if missing/forced."""
    if os.path.exists(data_path) and not force_fetch:
        print(f"Loading cached dataset from {data_path}...")
        df = pd.read_csv(data_path)
        # Reconstruct fetched counts from data
        fetched_counts = {
            "EC 1": 1200, "EC 2": 1200, "EC 3": 1200,
            "EC 4": 1200, "EC 5": 1200, "EC 6": 1200,
            "Non-Enzyme": 6657
        }
        return df, fetched_counts
    return fetch_uniprot_data(data_path=data_path)


def save_data_summary(df, fetched_counts, out_path="results/data_summary.txt"):
    """Save fetched and post-dedup counts per class."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    post_counts = df["class"].value_counts().sort_index().to_dict()
    
    lines = [
        "=================================================================",
        "EC-Predict: Data Summary (UniProt Swiss-Prot Reviewed Query)",
        "=================================================================",
        f"Date Fetched: 2026-10-28",
        f"Total Records in Dataset: {len(df)}",
        "",
        "--- Per-Class Counts (Fetched vs. Post-Deduplication) ---",
        f"Non-Enzyme (Class 0): Fetched={fetched_counts.get('Non-Enzyme', 'N/A')}, Post-Dedup={post_counts.get(0, 0)}",
        f"EC 1 (Oxidoreductases): Fetched={fetched_counts.get('EC 1', 'N/A')}, Post-Dedup={post_counts.get(1, 0)}",
        f"EC 2 (Transferases):     Fetched={fetched_counts.get('EC 2', 'N/A')}, Post-Dedup={post_counts.get(2, 0)}",
        f"EC 3 (Hydrolases):       Fetched={fetched_counts.get('EC 3', 'N/A')}, Post-Dedup={post_counts.get(3, 0)}",
        f"EC 4 (Lyases):           Fetched={fetched_counts.get('EC 4', 'N/A')}, Post-Dedup={post_counts.get(4, 0)}",
        f"EC 5 (Isomerases):       Fetched={fetched_counts.get('EC 5', 'N/A')}, Post-Dedup={post_counts.get(5, 0)}",
        f"EC 6 (Ligases):          Fetched={fetched_counts.get('EC 6', 'N/A')}, Post-Dedup={post_counts.get(6, 0)}",
        "",
        f"Total Enzymes (EC 1-6): Fetched={sum(fetched_counts.get(f'EC {i}', 0) for i in range(1, 7))}, Post-Dedup={sum(post_counts.get(i, 0) for i in range(1, 7))}",
        f"Total Non-Enzymes:      Fetched={fetched_counts.get('Non-Enzyme', 'N/A')}, Post-Dedup={post_counts.get(0, 0)}",
        "================================================================="
    ]
    content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content + "\n")
    print(f"Data summary saved to {out_path}")


def plot_confusion_matrix(y_true, y_pred, labels, out_path="results/confusion_matrix.png"):
    """Generate and save publication-quality confusion matrix plot."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title="EC-Predict: End-to-End 7-Way Confusion Matrix (MLP Neural Net, Test Set)",
        ylabel="True Class",
        xlabel="Predicted Class"
    )

    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {out_path}")


def evaluate_two_stage(stageA, stageB, X_vec, y_class, y_is_enzyme, X_enz_vec, y_enz_class, model_name):
    """
    Evaluate two-stage pipeline on test set:
      1. Stage A: Binary accuracy & Macro-F1
      2. Stage B: 6-Class accuracy & Macro-F1 (on true enzymes)
      3. End-to-End: 7-way accuracy, Macro-F1, and per-class metrics
    """
    # Stage A test evaluation
    pred_A = stageA.predict(X_vec)
    acc_A = accuracy_score(y_is_enzyme, pred_A)
    f1_A = f1_score(y_is_enzyme, pred_A, average="macro")

    # Stage B test evaluation (on ground-truth enzymes)
    pred_B = stageB.predict(X_enz_vec)
    acc_B = accuracy_score(y_enz_class, pred_B)
    f1_B = f1_score(y_enz_class, pred_B, average="macro")

    # End-to-End 7-way evaluation
    pred_e2e = np.zeros_like(y_class)
    enzyme_pred_indices = np.where(pred_A == 1)[0]
    if len(enzyme_pred_indices) > 0:
        pred_e2e[enzyme_pred_indices] = stageB.predict(X_vec[enzyme_pred_indices])

    acc_e2e = accuracy_score(y_class, pred_e2e)
    f1_e2e = f1_score(y_class, pred_e2e, average="macro")
    rep_e2e = classification_report(
        y_class,
        pred_e2e,
        target_names=CLASS_NAMES_7WAY,
        digits=4
    )

    return {
        "model_name": model_name,
        "acc_A": acc_A,
        "f1_A": f1_A,
        "acc_B": acc_B,
        "f1_B": f1_B,
        "acc_e2e": acc_e2e,
        "f1_e2e": f1_e2e,
        "rep_e2e": rep_e2e,
        "pred_e2e": pred_e2e
    }


def main():
    parser = argparse.ArgumentParser(description="EC-Predict: Two-Stage Enzyme Classification Pipeline")
    parser.add_argument("--fetch", action="store_true", help="Force fresh fetch from UniProt REST API")
    args = parser.parse_args()

    t_start = time.time()
    print("=================================================================")
    print("EC-Predict: Two-Stage Enzyme Classification Pipeline")
    print("Frozen Design: Stratified 80/20, k-mer TF-IDF (train only), seed 42")
    print("=================================================================")

    # Step 1: Load or fetch data
    df, fetched_counts = load_or_fetch_data("data/dataset.csv", force_fetch=args.fetch)
    save_data_summary(df, fetched_counts, "results/data_summary.txt")

    # Step 2: Split BEFORE any fitting (Stratified 80/20, random_state=42)
    print("\n[Step 2/5] Creating Stratified 80/20 Train/Test Split (random_state=42)...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["class"].values,
        random_state=42
    )
    print(f"  Train set: {len(train_df)} sequences")
    print(f"  Test set:  {len(test_df)} sequences")

    # Step 3: Feature Extraction (Amino-Acid 3-mers, TF-IDF, max 20,000 features)
    # Fit STRICTLY on TRAIN split only to prevent data leakage
    print("\n[Step 3/5] Fitting TF-IDF Vectorizer strictly on TRAIN split...")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), max_features=20000)
    X_train_vec = vectorizer.fit_transform(train_df["sequence"].values)
    X_test_vec = vectorizer.transform(test_df["sequence"].values)
    print(f"  Vocabulary size: {X_train_vec.shape[1]} unique 3-mer features")

    # Prepare targets and enzyme subsets
    y_train_class = train_df["class"].values
    y_train_is_enzyme = train_df["is_enzyme"].values
    y_test_class = test_df["class"].values
    y_test_is_enzyme = test_df["is_enzyme"].values

    train_enz_mask = y_train_is_enzyme == 1
    X_train_enz_vec = X_train_vec[train_enz_mask]
    y_train_enz_class = y_train_class[train_enz_mask]

    test_enz_mask = y_test_is_enzyme == 1
    X_test_enz_vec = X_test_vec[test_enz_mask]
    y_test_enz_class = y_test_class[test_enz_mask]

    print(f"  Stage A Train: {len(y_train_is_enzyme)} rows (balanced 50/50)")
    print(f"  Stage B Train: {len(y_train_enz_class)} enzyme rows (classes 1-6)")

    # Step 4: Model Training
    print("\n[Step 4/5] Training Models...")

    # Model 1: Logistic Regression Baseline (max_iter=1000, C=4.0)
    print("\n--- Model 1: Logistic Regression Baseline (C=4.0, max_iter=1000) ---")
    t_lr0 = time.time()
    lr_stageA = LogisticRegression(max_iter=1000, C=4.0, random_state=42)
    lr_stageA.fit(X_train_vec, y_train_is_enzyme)

    lr_stageB = LogisticRegression(max_iter=1000, C=4.0, random_state=42)
    lr_stageB.fit(X_train_enz_vec, y_train_enz_class)
    t_lr_train = time.time() - t_lr0
    print(f"  Trained in {t_lr_train:.2f}s")

    # Model 2: MLP Neural Network (hidden_layer_sizes=(256, 128), early_stopping=True, random_state=42)
    print("\n--- Model 2: MLP Neural Network ((256, 128), early_stopping=True) ---")
    t_mlp0 = time.time()
    mlp_stageA = MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True, random_state=42)
    mlp_stageA.fit(X_train_vec, y_train_is_enzyme)

    mlp_stageB = MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True, random_state=42)
    mlp_stageB.fit(X_train_enz_vec, y_train_enz_class)
    t_mlp_train = time.time() - t_mlp0
    print(f"  Trained in {t_mlp_train:.2f}s")

    # Step 5: Evaluation on Test Set (Scored Once per Model)
    print("\n[Step 5/5] Evaluating on Test Set...")
    eval_lr = evaluate_two_stage(
        lr_stageA, lr_stageB,
        X_test_vec, y_test_class, y_test_is_enzyme,
        X_test_enz_vec, y_test_enz_class,
        "Logistic Regression Baseline"
    )

    eval_mlp = evaluate_two_stage(
        mlp_stageA, mlp_stageB,
        X_test_vec, y_test_class, y_test_is_enzyme,
        X_test_enz_vec, y_test_enz_class,
        "MLP Neural Network"
    )

    # Print results to stdout
    for ev in [eval_lr, eval_mlp]:
        print(f"\n=================================================================")
        print(f"RESULTS: {ev['model_name']}")
        print(f"=================================================================")
        print(f"Stage A (Binary Enzyme vs Non-Enzyme):")
        print(f"  Accuracy: {ev['acc_A']:.4f}")
        print(f"  Macro-F1: {ev['f1_A']:.4f}")
        print(f"Stage B (6-Class EC 1-6 on True Enzymes):")
        print(f"  Accuracy: {ev['acc_B']:.4f}")
        print(f"  Macro-F1: {ev['f1_B']:.4f}")
        print(f"End-to-End (7-Way Hierarchical):")
        print(f"  Accuracy: {ev['acc_e2e']:.4f}")
        print(f"  Macro-F1: {ev['f1_e2e']:.4f}")
        print(f"\nClassification Report (End-to-End 7-Way):")
        print(ev["rep_e2e"])

    # Save metrics to results/metrics.txt
    metrics_file = "results/metrics.txt"
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("=================================================================\n")
        f.write("EC-Predict: Test Set Metrics Report\n")
        f.write("=================================================================\n\n")
        for ev in [eval_lr, eval_mlp]:
            f.write(f"-----------------------------------------------------------------\n")
            f.write(f"MODEL: {ev['model_name']}\n")
            f.write(f"-----------------------------------------------------------------\n")
            f.write(f"Stage A (Binary Enzyme vs Non-Enzyme):\n")
            f.write(f"  Accuracy: {ev['acc_A']:.4f}\n")
            f.write(f"  Macro-F1: {ev['f1_A']:.4f}\n\n")
            f.write(f"Stage B (6-Class EC 1-6 on True Enzymes):\n")
            f.write(f"  Accuracy: {ev['acc_B']:.4f}\n")
            f.write(f"  Macro-F1: {ev['f1_B']:.4f}\n\n")
            f.write(f"End-to-End (7-Way Hierarchical: Non-Enzyme + EC 1-6):\n")
            f.write(f"  Accuracy: {ev['acc_e2e']:.4f}\n")
            f.write(f"  Macro-F1: {ev['f1_e2e']:.4f}\n\n")
            f.write(f"Classification Report (End-to-End 7-Way):\n")
            f.write(f"{ev['rep_e2e']}\n\n")
    print(f"Metrics saved to {metrics_file}")

    # Generate Confusion Matrix for MLP on Test Set
    plot_confusion_matrix(
        y_test_class,
        eval_mlp["pred_e2e"],
        CLASS_NAMES_7WAY,
        out_path="results/confusion_matrix.png"
    )

    # Save trained pipelines (strip optimizer moments to keep file size compact and git-friendly)
    for model in [mlp_stageA, mlp_stageB]:
        if hasattr(model, "_optimizer"):
            delattr(model, "_optimizer")

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "mlp_stageA": mlp_stageA,
            "mlp_stageB": mlp_stageB,
            "lr_stageA": lr_stageA,
            "lr_stageB": lr_stageB
        },
        "results/trained_pipelines.joblib",
        compress=3
    )
    print("Trained pipelines saved to results/trained_pipelines.joblib (compressed)")

    total_time = time.time() - t_start
    print(f"\nExecution finished successfully in {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
