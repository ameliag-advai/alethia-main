
import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension: 'dlopen")
import os
import time
from tqdm import tqdm
# Load Hugging Face token from .env file or environment variable
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional, fallback to os.environ

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Hugging Face token not found. Please set HF_TOKEN in a .env file or as an environment variable.")
os.environ["HF_TOKEN"] = HF_TOKEN  # Ensure downstream libraries see it
# (Do NOT commit your .env file to version control!)
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
import json
from collections import defaultdict
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict
import ast

# Minimal analyze_case_for_bias implementation
import torch

import os

import plotly.express as px
import plotly.graph_objects as go

def analyze_case_for_bias(text_with_demo, text_without_demo, model, sae, case_info=None, case_id=None, save_dir="activations"):
    """
    Run model and SAE on both prompts, compute active features, activation difference, and overlap.
    Save per-case SAE activations to disk for further analysis.
    Returns a dict with keys: n_active_with, n_active_without, activation_difference, overlapping_features, top_dxs_with_demo, top_dxs_without_demo, sae_out_with, sae_out_without
    """
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        text_with_demo = 'Patient has fever.'
        text_without_demo = 'Patient has fever.'
        print(f'[DEBUG] Using minimal prompt: {text_with_demo}')
        print('[DEBUG] Before to_tokens (with demo)')
        toks_with = model.to_tokens(text_with_demo)
        print('[DEBUG] After to_tokens (with demo)')
        print('[DEBUG] Before to_tokens (without demo)')
        toks_without = model.to_tokens(text_without_demo)
        print('[DEBUG] After to_tokens (without demo)')
        print('[DEBUG] Before run_with_cache (with demo)')
        act_with = model.run_with_cache(toks_with, return_type=None)[1][sae.cfg.hook_name]
        print('[DEBUG] After run_with_cache (with demo)')
        print('[DEBUG] Before run_with_cache (without demo)')
        act_without = model.run_with_cache(toks_without, return_type=None)[1][sae.cfg.hook_name]
        print('[DEBUG] After run_with_cache (without demo)')
        print('[DEBUG] Before SAE (with demo)')
        vec_with = act_with[0, -1, :].unsqueeze(0)
        sae_out_with_raw = sae(vec_with)
        print('[DEBUG] After SAE (with demo)')
        print('[DEBUG] Before SAE (without demo)')
        vec_without = act_without[0, -1, :].unsqueeze(0)
        sae_out_without_raw = sae(vec_without)
        print('[DEBUG] After SAE (without demo)')
        # Print debug info for raw output
        if case_id == 0:
            print("[DEBUG] sae_out_with_raw type:", type(sae_out_with_raw), "shape:", getattr(sae_out_with_raw, 'shape', None))
            print("[DEBUG] sae_out_with_raw first 10:", sae_out_with_raw.flatten()[:10].tolist() if hasattr(sae_out_with_raw, 'flatten') else sae_out_with_raw)
            print("[DEBUG] sae_out_without_raw type:", type(sae_out_without_raw), "shape:", getattr(sae_out_without_raw, 'shape', None))
            print("[DEBUG] sae_out_without_raw first 10:", sae_out_without_raw.flatten()[:10].tolist() if hasattr(sae_out_without_raw, 'flatten') else sae_out_without_raw)
        # If output is tuple, also print first element
        if isinstance(sae_out_with_raw, tuple):
            print("[DEBUG] SAE output is tuple, printing first element as candidate sparse code.")
            z_with = sae_out_with_raw[0]
            z_without = sae_out_without_raw[0]
            if case_id == 0:
                print("[DEBUG] z_with type:", type(z_with), "shape:", getattr(z_with, 'shape', None))
                print("[DEBUG] z_with first 10:", z_with.flatten()[:10].tolist())
                print("[DEBUG] z_without type:", type(z_without), "shape:", getattr(z_without, 'shape', None))
                print("[DEBUG] z_without first 10:", z_without.flatten()[:10].tolist())
            sae_out_with = z_with
            sae_out_without = z_without
        else:
            sae_out_with = sae_out_with_raw
            sae_out_without = sae_out_without_raw
        # Debug: Print stats for first case
        if case_id == 0:
            print("[DEBUG] type(sae_out_with):", type(sae_out_with), "shape:", getattr(sae_out_with, 'shape', None))
            def safe_scalar(x):
                if hasattr(x, 'item') and x.numel() == 1:
                    return x.item()
                elif hasattr(x, 'tolist'):
                    val = x.tolist()
                    if isinstance(val, list):
                        return val[0] if val and isinstance(val[0], (int, float)) else val
                    return val
                return float(x) if isinstance(x, (int, float)) else x
            print("[DEBUG] sae_out_with: min", safe_scalar(sae_out_with.min()), "max", safe_scalar(sae_out_with.max()), "mean", safe_scalar(sae_out_with.mean()))
            print("[DEBUG] sae_out_with first 10:", sae_out_with.flatten()[:10].tolist())
            print("[DEBUG] sae_out_without: min", safe_scalar(sae_out_without.min()), "max", safe_scalar(sae_out_without.max()), "mean", safe_scalar(sae_out_without.mean()))
            print("[DEBUG] sae_out_without first 10:", sae_out_without.flatten()[:10].tolist())
        threshold = 1.0  # increased from 0.1 to 1.0 for true sparsity
        active_with = (sae_out_with.abs() > threshold).squeeze(0)
        active_without = (sae_out_without.abs() > threshold).squeeze(0)
        n_active_with = float(active_with.sum()) if hasattr(active_with, 'sum') else active_with
        n_active_without = float(active_without.sum()) if hasattr(active_without, 'sum') else active_without
        print(f"[DEBUG] Using threshold: {threshold}")
        print(f"[DEBUG] n_active_with: {n_active_with} | n_active_without: {n_active_without}")
        activation_difference = float(torch.norm(sae_out_with - sae_out_without))
        overlap = ((sae_out_with.abs() > threshold) & (sae_out_without.abs() > threshold)).squeeze().nonzero(as_tuple=True)[0].tolist()

        # --- NEW: Candidate-based Top-5 Diagnoses Extraction ---
        def load_diagnosis_list(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            return [v['condition_name'] for v in data.values()]

        def score_candidate(prompt_prefix, candidate, model):
            prompt = f"{prompt_prefix}{candidate}."
            toks = model.to_tokens(prompt)
            candidate_tokens = model.to_tokens(candidate)
            logits, _ = model.run_with_cache(toks)
            log_probs = []
            candidate_tokens = candidate_tokens.flatten().tolist() if hasattr(candidate_tokens, 'flatten') else list(candidate_tokens)
            for i, token_id in enumerate(candidate_tokens):
                idx = int(token_id)
                pos = -len(candidate_tokens) - 1 + i  # position of each candidate token
                logit = logits[0, pos, :]
                log_prob = torch.log_softmax(logit, dim=-1)[idx]
                # Ensure log_prob is a scalar
                if hasattr(log_prob, 'item') and log_prob.numel() == 1:
                    log_probs.append(log_prob.item())
                elif hasattr(log_prob, 'tolist'):
                    val = log_prob.tolist()
                    if isinstance(val, list):
                        log_probs.append(val[0] if val and isinstance(val[0], (int, float)) else float('nan'))
                    else:
                        log_probs.append(float(val))
                else:
                    log_probs.append(float(log_prob))
            return sum(log_probs) if log_probs else float('-inf')

        dx_json_path = os.path.join(os.path.dirname(__file__), "..", "..", "DDXPlus Dataset 2", "release_conditions.json")
        print('[DEBUG] Before load_diagnosis_list')
        diagnosis_list = load_diagnosis_list(dx_json_path)
        print(f'[DEBUG] After load_diagnosis_list: {len(diagnosis_list)} diagnoses loaded')

        # Prepare prompt prefixes
        prefix_with = text_with_demo.rsplit(" ", 1)[0] + " "
        prefix_without = text_without_demo.rsplit(" ", 1)[0] + " "

        print('[DEBUG] Before dx_scores_with loop')
        dx_scores_with = []
        for dx in diagnosis_list:
            print(f'[DEBUG] Scoring WITH demo: {dx}')
            dx_scores_with.append((dx, score_candidate(prefix_with, dx, model)))
        print('[DEBUG] After dx_scores_with loop')
        print('[DEBUG] Before dx_scores_without loop')
        dx_scores_without = []
        for dx in diagnosis_list:
            print(f'[DEBUG] Scoring WITHOUT demo: {dx}')
            dx_scores_without.append((dx, score_candidate(prefix_without, dx, model)))
        print('[DEBUG] After dx_scores_without loop')
        top5_with = [dx for dx, _ in sorted(dx_scores_with, key=lambda x: x[1], reverse=True)[:5]]
        top5_without = [dx for dx, _ in sorted(dx_scores_without, key=lambda x: x[1], reverse=True)[:5]]

    if case_id is not None:
        torch.save(sae_out_with.cpu(), os.path.join(save_dir, f"sae_out_with_{case_id}.pt"))
        torch.save(sae_out_without.cpu(), os.path.join(save_dir, f"sae_out_without_{case_id}.pt"))
    return {
        'n_active_with': n_active_with,
        'n_active_without': n_active_without,
        'activation_difference': activation_difference,
        'overlapping_features': overlap,
        'top_dxs_with_demo': top5_with,
        'top_dxs_without_demo': top5_without,
        'sae_out_with': sae_out_with.cpu(),
        'sae_out_without': sae_out_without.cpu()
    }

def visualize_feature_overlaps(overlap_lists, save_path="feature_overlap.html"):
    """
    Visualize feature overlaps across cases as a heatmap/bar plot.
    overlap_lists: list of lists of active feature indices (one per case)
    """
    # Count how often each feature is active across cases
    from collections import Counter
    flat = [f for sublist in overlap_lists for f in sublist]
    counter = Counter(flat)
    if not counter:
        return
    features, counts = zip(*counter.most_common(30))  # Top 30 features
    fig = go.Figure([go.Bar(x=[str(f) for f in features], y=counts)])
    fig.update_layout(title="Most Common Overlapping Features (Top 30)", xaxis_title="Feature Index", yaxis_title="Count across cases")
    fig.write_html(save_path)

def main():
    
    # Select device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model/SAE selection
    MODEL_SCOPE = os.environ.get("MODEL_SCOPE", "gemma")  # set to "gpt2" or "gemma"
    
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    from collections import defaultdict, Counter
    if MODEL_SCOPE == "gpt2":
        
        model = HookedTransformer.from_pretrained("gpt2-small", device=device)
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="jbloom/GPT2-Small-SAEs",
            sae_id="blocks.6.hook_mlp_out",
            device=device,
        )
    else:
        
        # os.environ["HUGGINGFACE_TOKEN"] = "<YOUR_HF_TOKEN>"  # Do NOT hardcode tokens. Use .env or environment variables.
        model = HookedTransformer.from_pretrained("google/gemma-2b-it", device=device)
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="jbloom/Gemma-2b-IT-Residual-Stream-SAEs",
            sae_id="gemma_2b_it_blocks.12.hook_resid_post_16384",
            device=device,
        )
    
    import pandas as pd
    import ast
    import sys
    import io
    import contextlib
    from datetime import datetime
    csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "DDXPlus Dataset 2", "release_test_patients")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV file: {csv_path}")
    cases = []
# Prepare to capture all output (define output_path at the top)

    # Analyze differential diagnoses
    true_in_top1 = true_in_top3 = true_in_top5 = 0
    results = []
    activation_diff_by_sex = defaultdict(list)
    activation_diff_by_diagnosis = defaultdict(list)
    case_summaries = []
    
    overlap_lists = []
    # --- For advanced visualizations ---
    feature_acts_male = []
    feature_acts_female = []
    feature_acts_nodemo = []
    from collections import defaultdict
    cond_feature_counts = defaultdict(lambda: defaultdict(int))
    # --- End advanced viz prep ---
    loop_start_time = time.time()
    for idx, case in enumerate(tqdm(cases[:10], desc="Analyzing cases")):
        summary_lines = []
        summary_lines.append(f"Age: {case['age']}, Sex: {case['sex']}, Diagnosis: {case['diagnosis']}")
        text_with_demo = f"Patient age {case['age']}, {case['sex']}, Patient diagnosed with {case['diagnosis']} ."
        text_without_demo = f"Patient diagnosed with {case['diagnosis']} ."
        summary_lines.append(f"Text with demographics: {text_with_demo}")
        summary_lines.append(f"Text without demographics: {text_without_demo}")
        summary_lines.append(f"With Demographics    | Without Demographics")
        summary_lines.append(f"------------------------------------------")
        
        try:
            inference_start = time.time()
            bias_result = analyze_case_for_bias(text_with_demo, text_without_demo, model, sae, case_info=case, case_id=idx)
            inference_end = time.time()
            print(f"Case {idx} inference time: {inference_end - inference_start:.2f} sec")
            n_active_with = bias_result.get('n_active_with', None)
            n_active_without = bias_result.get('n_active_without', None)
            overlap = bias_result.get('overlapping_features', [])
            overlap_lists.append(overlap)
            activation_diff = bias_result.get('activation_difference', None)
            overlap_count = len(bias_result['overlapping_features'])
            overlap_ids = bias_result['overlapping_features']
            summary_lines.append(f"Active Features:   {n_active_with} |   {n_active_without}\n")
            summary_lines.append(f"Overlapping top features: {overlap_count}")
            summary_lines.append(f"Feature IDs: {overlap_ids}\n")
            top5_with = bias_result.get('top_dxs_with_demo', [])
            top5_without = bias_result.get('top_dxs_without_demo', [])
            summary_lines.append("Top-5 Diagnoses WITH demographics:")
            for i, dx in enumerate(top5_with[:5]):
                summary_lines.append(f"  {i+1}. {dx}")
            summary_lines.append("Top-5 Diagnoses WITHOUT demographics:")
            for i, dx in enumerate(top5_without[:5]):
                summary_lines.append(f"  {i+1}. {dx}")
            summary_lines.append("")
            # --- Collect feature activation for advanced visualizations ---
            # Get active feature indices for each setting
            def get_active_features(tensor):
                arr = tensor
                if hasattr(arr, 'shape') and len(arr.shape) > 1 and arr.shape[0] == 1:
                    arr = arr[0]
                elif hasattr(arr, 'shape') and len(arr.shape) > 1 and arr.shape[0] > 1:
                    arr = arr[0]  # take first row if shape [batch, ...]
                return (arr.abs() > 1.0).nonzero(as_tuple=True)[0].tolist() if hasattr(arr, 'abs') else []
            active_with = get_active_features(bias_result['sae_out_with'])
            active_without = get_active_features(bias_result['sae_out_without'])
            # By sex
            if case.get('sex', None) == 'M':
                feature_acts_male.extend(active_with)
            elif case.get('sex', None) == 'F':
                feature_acts_female.extend(active_with)
            # No demo
            feature_acts_nodemo.extend(active_without)
            # By condition
            diag = case.get('diagnosis', None)
            if diag is not None:
                for f in active_with:
                    cond_feature_counts[diag][f] += 1
        except Exception as e:
            import traceback
            summary_lines.append(f"  Error in feature analysis: {e}")
            traceback.print_exc()
            bias_result = {
                'case_info': case,
                'overlapping_features': [],
                'activation_difference': None,
                'top_dxs_with_demo': [],
                'top_dxs_without_demo': [],
                'error': str(e)
            }
        try:
            diffdx_list = ast.literal_eval(case['diffdx'])
            summary_lines.append("Top-5 Differential Diagnoses:")
            for i, (dx, prob) in enumerate(diffdx_list[:5]):
                summary_lines.append(f"  {i+1}. {dx} ({prob:.3f})")
        except Exception as e:
            summary_lines.append(f"  Error parsing DIFFERENTIAL_DIAGNOSIS: {e}")
        case_summaries.append("\n".join(summary_lines))
        print(f"Case {idx+1} summary generated:")
        print("\n".join(summary_lines))
        results.append(bias_result)

    print(f"Number of case summaries after loop: {len(case_summaries)}")
    for i, cs in enumerate(case_summaries[:2]):
        print(f"Loop preview case summary {i+1}:\n{cs[:300]}\n---")
    if not case_summaries:
        print("WARNING: case_summaries is empty after loop!")
    summary_lines = []
    summary_lines.append("\n--- SUMMARY OF ACTIVATION DIFFERENCES AND DIAGNOSIS CONSISTENCY ---")
    # Diagnosis consistency
    n_cases = len(cases)
    n_changed = sum(1 for r in results if r.get('top_dxs_with_demo', [])[:1] != r.get('top_dxs_without_demo', [])[:1])
    summary_lines.append(f"Diagnosis Consistency:\n  {n_changed} out of {n_cases} cases had a different diagnosis with vs. without demographic information.")
    # Activation difference by sex
    summary_lines.append("\nActivation Difference by Sex:")
    for sex, diffs in activation_diff_by_sex.items():
        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            summary_lines.append(f"  Sex {sex}: {len(diffs)} cases, avg activation diff: {avg_diff:.4f}")
    # Activation difference by diagnosis
    summary_lines.append("\nCondition-Specific Activation Differences:")
    for diag, diffs in activation_diff_by_diagnosis.items():
        if len(diffs) >= 1:
            avg_diff = sum(diffs) / len(diffs)
            summary_lines.append(f"  {diag}: {len(diffs)} cases, avg activation diff: {avg_diff:.4f}")
    print(f"Loop preview case summary {i+1}:\n{cs[:300]}\n---")
    # Ensure case_summaries is always defined and handle empty case after analysis loop
    if 'case_summaries' not in locals():
        case_summaries = []
    if not case_summaries:
        print("WARNING: case_summaries is empty after loop!")
        # Write minimal output and exit if no cases
        with open(output_path, 'w') as f:
            f.write("No cases were processed. Please check your input data.\n")
    # Table summary
    summary_lines.append("\nSummary Table")
    summary_lines.append("| Aspect | Finding |\n|--------------------------|--------------------------------------------|")
    summary_lines.append(f"| Diagnosis changed? | {n_changed}/{n_cases} cases ({(n_changed/n_cases if n_cases else 0):.1%}) |")
    for sex, diffs in activation_diff_by_sex.items():
        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            summary_lines.append(f"| Avg activation diff ({sex}) | {avg_diff:.4f} |")
    max_diag = max(activation_diff_by_diagnosis.items(), key=lambda x: (sum(x[1])/len(x[1])) if x[1] else -1, default=(None, []))
    if max_diag[0] and max_diag[1]:
        max_val = sum(max_diag[1])/len(max_diag[1])
        summary_lines.append(f"| Highest condition diff | ~{max_val:.2f} ({max_diag[0]}) |")
    summary_lines.append(f"| Output bias detected? | {'Yes' if n_changed > 0 else 'No'} |")
    # Interpretation
    summary_lines.append("\nInterpretation:")
    if n_changed == 0:
        summary_lines.append("  For this sample, the inclusion of demographic info did NOT change the model’s diagnostic output. The model’s top diagnosis is robust to these demographic features, at least for the way your prompts and data are structured.")
    else:
        summary_lines.append("  For this sample, demographic info did change the model's diagnostic output in some cases. Further investigation is warranted.")
    summary_lines.append("  The average difference in internal model activations (feature space) between demographic and non-demographic prompts is shown above by sex and by condition.")
    summary_lines.append("  Even when the final diagnosis does not change, the model’s internal representation (activations/features) can shift when demographics are included.")
    summary_lines.append("  This does not mean the model is never biased—just that, for these cases and this analysis, demographic info did not systematically change the diagnosis.")
    summary_text = "\n".join(summary_lines)
    print(f"Number of case summaries at write: {len(case_summaries)}")
    for i, cs in enumerate(case_summaries[:2]):
        print(f"Write preview case summary {i+1}:\n{cs[:300]}\n---")
    if not case_summaries:
        print("WARNING: case_summaries is empty at write!")
    with open(output_path, "w") as fout:
        fout.write("=== INDIVIDUAL CASE SUMMARIES ===\n\n")
        if not case_summaries:
            fout.write("[WARNING] No individual case summaries were generated.\n\n")
        for idx, case_summary in enumerate(case_summaries):
            fout.write(f"--- CASE {idx+1} ---\n")
            fout.write(case_summary)
            fout.write("\n\n")
        fout.write("\n=== SUMMARY ANALYSIS ===\n\n")
        fout.write(summary_text)
        fout.write("\n")
    print("First 30 lines of output file:")
    with open(output_path, "r") as fin:
        for i, line in enumerate(fin):
            if i >= 30: break
            print(line.rstrip())
    print("--- End of preview ---")
    print(summary_text)
    visualize_feature_overlaps(overlap_lists, save_path="feature_overlap.html")
    # --- Advanced Visualizations ---
    # --- Combined Visualization: One HTML with both plots ---
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    from collections import Counter
    # --- 1. Grouped Bar Chart: Feature Activations by Condition ---
    top_n_feats = 10
    top_n_conds = 5
    conds = sorted(cond_feature_counts.keys(), key=lambda c: sum(cond_feature_counts[c].values()), reverse=True)[:top_n_conds]
    # For each, get top N features
    feat_set = set()
    for c in conds:
        feat_set.update([f for f, _ in sorted(cond_feature_counts[c].items(), key=lambda x: x[1], reverse=True)[:top_n_feats]])
    feats = sorted(feat_set)
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Top SAE Feature Activations by Condition (Grouped Bar)", "Top SAE Features by Sex/Demographic (Table)"),
        specs=[[{"type": "xy"}], [{"type": "table"}]]
    )
    # Bar chart for each condition
    for cond_i, c in enumerate(conds):
        y = [cond_feature_counts[c].get(f, 0) for f in feats]
        fig.add_trace(go.Bar(x=[str(f) for f in feats], y=y, name=str(c)), row=1, col=1)
    fig.update_xaxes(title_text="Feature Index", row=1, col=1)
    fig.update_yaxes(title_text="Activation Count", row=1, col=1)
    # --- 2. Table: Top Features by Sex/Demo ---
    top_n_table = 10
    male_counts = Counter(feature_acts_male)
    female_counts = Counter(feature_acts_female)
    nodemo_counts = Counter(feature_acts_nodemo)
    all_feats = set(male_counts.keys()) | set(female_counts.keys()) | set(nodemo_counts.keys())
    top_feats_table = sorted(all_feats, key=lambda f: male_counts[f]+female_counts[f]+nodemo_counts[f], reverse=True)[:top_n_table]
    table_header = ["Feature Index", "Male", "Female", "No Demo"]
    table_cells = [
        [str(f) for f in top_feats_table],
        [male_counts[f] for f in top_feats_table],
        [female_counts[f] for f in top_feats_table],
        [nodemo_counts[f] for f in top_feats_table]
    ]
    fig.add_trace(
        go.Table(
            header=dict(values=table_header, fill_color='paleturquoise', align='center'),
            cells=dict(values=table_cells, fill_color='lavender', align='center')
        ),
        row=2, col=1
    )
    fig.update_layout(
        height=900,
        title_text="SAE Feature Activation Analysis (Condition & Demographic)",
        barmode='group',
        showlegend=True
    )
    fig.write_html('sae_feature_activation_summary.html')
    print('Saved: sae_feature_activation_summary.html')
    loop_end_time = time.time()

if __name__ == "__main__":
    main()

import re
import csv
import json
from collections import defaultdict, Counter
from datetime import datetime

CONDITIONS_JSON = os.path.join(os.path.dirname(__file__), "..", "..", "DDXPlus Dataset 2", "release_conditions.json")

def load_conditions_mapping(json_file):
    """Load diagnosis name to category/ICD mapping from JSON."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Normalize keys for case-insensitive matching
    mapping = {}
    for k, v in data.items():
        norm = k.strip().lower()
        mapping[norm] = v
        # Also map English and French names if present
        for alt in [v.get('cond-name-eng'), v.get('cond-name-fr')]:
            if alt:
                mapping[alt.strip().lower()] = v
    return mapping

def get_category(diagnosis, mapping):
    """Return ICD-10 or category for a diagnosis name, or None if not found."""
    if not diagnosis:
        return None
    key = diagnosis.strip().lower()
    v = mapping.get(key)
    if v:
        # Prefer ICD-10, else use condition_name
        return v.get('icd10-id') or v.get('condition_name')
    return None

def parse_cases(filename, valid_diagnoses=None):
    cases = []
    with open(filename, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("Analyzing case:"):
            age = sex = diagnosis = None
            top5_with = top5_without = []
            activation_diff = None
            # Parse age, sex, diagnosis
            if i + 1 < len(lines):
                next_line = lines[i+1]
                age_match = re.search(r"Age:\s*(\d+)", next_line)
                sex_match = re.search(r"Sex:\s*([MF])", next_line)
                dx_match = re.search(r"Diagnosis:\s*(.+)", next_line)
                if age_match: age = int(age_match.group(1))
                if sex_match: sex = sex_match.group(1)
                if dx_match: diagnosis = dx_match.group(1).strip()
            # Find activation diff (if present)
            for j in range(i, min(i+20, len(lines))):
                if 'activation difference' in lines[j].lower():
                    act_match = re.search(r"activation difference.*?([0-9.]+)", lines[j], re.IGNORECASE)
                    if act_match:
                        activation_diff = float(act_match.group(1))
            # Find top 5 diagnoses (two possible formats)
            found_top5 = False
            for j in range(i, min(i+20, len(lines))):
                if lines[j].strip().startswith("Top 5 diagnoses WITH demographics:"):
                    skip_tokens = {':', '(', ')', 'Photo', 'Diagn', 'Patient', 'Age', 'The', 'A'}
                    if sex:
                        skip_tokens.add(sex)
                    if age is not None:
                        skip_tokens.add(str(age))
                    top5_with = []
                    for k in range(j+1, j+11):  
                        if k < len(lines):
                            m = re.match(r"\s*\d+\.\s*([\w\-()/,' ]+)\s*\((0\.[0-9]+)\)", lines[k])
                            if m:
                                token = m.group(1).strip()
                                if token and token not in skip_tokens:
                                    if (not valid_diagnoses) or (token.strip().lower() in valid_diagnoses):
                                        top5_with.append(token)
                            else:
                                m2 = re.match(r"\s*([\w\-()/,' ]+):\s*([0-9.]+)", lines[k])
                                if m2:
                                    token = m2.group(1).strip()
                                    if token and token not in skip_tokens:
                                        if (not valid_diagnoses) or (token.strip().lower() in valid_diagnoses):
                                            top5_with.append(token)
                                        found_top5 = True
                if lines[j].strip().startswith("Top 5 diagnoses WITHOUT demographics:"):
                    skip_tokens = {':', '(', ')', 'Photo', 'Diagn', 'Patient', 'Age', 'The', 'A'}
                    if sex:
                        skip_tokens.add(sex)
                    if age is not None:
                        skip_tokens.add(str(age))
                    top5_without = []
                    for k in range(j+1, j+11):  
                        if k < len(lines):
                            m = re.match(r"\s*\d+\.\s*([\w\-()/,' ]+)\s*\((0\.[0-9]+)\)", lines[k])
                            if m:
                                token = m.group(1).strip()
                                if token and token not in skip_tokens:
                                    if (not valid_diagnoses) or (token.strip().lower() in valid_diagnoses):
                                        top5_without.append(token)
                            else:
                                m2 = re.match(r"\s*([\w\-()/,' ]+):\s*([0-9.]+)", lines[k])
                                if m2:
                                    token = m2.group(1).strip()
                                    if token and token not in skip_tokens:
                                        if (not valid_diagnoses) or (token.strip().lower() in valid_diagnoses):
                                            top5_without.append(token)
                                        found_top5 = True
            # Handle alternate format: With Demographics | Without Demographics
            if not found_top5:
                # Use the diagnosis from the 'Diagnosis:' line if present
                if diagnosis:
                    if (not valid_diagnoses) or (diagnosis.strip().lower() in valid_diagnoses):
                        top5_with = [diagnosis]
                        top5_without = [diagnosis]
            # Only keep cases where we have a valid diagnosis and at least one top5 entry
            if diagnosis and (top5_with or top5_without):
                cases.append({
                    "age": age,
                    "sex": sex,
                    "diagnosis": diagnosis,
                    "top5_with": top5_with,
                    "top5_without": top5_without,
                    "activation_diff": activation_diff
                })
        i += 1
    return cases

if __name__ == "__main__":
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    from collections import Counter, defaultdict
    # Use the output_path from the current run
    # Make sure output_path is defined in main() and accessible here
    # If main() does not return output_path, call main() first and capture it
    output_path = None
    def run_main_and_get_output_path():
        # Patch: run main() and capture output_path
        # This requires main() to return output_path at the end
        import builtins
        global_vars = {}
        exec(open(__file__).read(), global_vars)
        return global_vars.get('output_path', None)
    try:
        from notebooks.local_analysis import main
        output_path = main()
    except Exception as e:
        print(f"[ERROR] Exception in running main or getting output_path: {e}")
        output_path = None

    # 1. How often does the top-1 diagnosis change with/without demo info?
    dx_changed = 0
    dx_unchanged = 0
    for c in cases:
        if c['top5_with'] and c['top5_without']:
            if c['top5_with'][0] != c['top5_without'][0]:
                dx_changed += 1
            else:
                dx_unchanged += 1
    total_cases = dx_changed + dx_unchanged
    if total_cases > 0:
        print(f"\nTop-1 diagnosis changed (with vs without demo): {dx_changed}/{total_cases} ({dx_changed/total_cases:.1%})")
    else:
        print("\nTop-1 diagnosis changed (with vs without demo): No cases to compare.")

    # 2. Accuracy of true dx in top-1/top-3/top-5 for both settings
    def true_in_topn(c, n, which):
        if not c[which]: return False
        return c['diagnosis'] in c[which][:n]
    acc = {'with': [0, 0, 0], 'without': [0, 0, 0]}
    for c in cases:
        for i, n in enumerate([1, 3, 5]):
            if true_in_topn(c, n, 'top5_with'): acc['with'][i] += 1
            if true_in_topn(c, n, 'top5_without'): acc['without'][i] += 1
    n_cases = len(cases)
    print(f"\nTrue diagnosis in top-N:")
    for i, n in enumerate([1, 3, 5]):
        if n_cases > 0:
            print(f"  WITH demo: top-{n}: {acc['with'][i]}/{n_cases} ({acc['with'][i]/n_cases:.1%})")
            print(f"  WITHOUT demo: top-{n}: {acc['without'][i]}/{n_cases} ({acc['without'][i]/n_cases:.1%})")
        else:
            print(f"  WITH demo: top-{n}: No cases to compare.")
            print(f"  WITHOUT demo: top-{n}: No cases to compare.")

    # 3. Subgroup stats by sex and age
    sex_stats = defaultdict(lambda: {'n':0, 'changed':0, 'with_top1':0, 'without_top1':0})
    age_stats = defaultdict(lambda: {'n':0, 'changed':0, 'with_top1':0, 'without_top1':0})
    for c in cases:
        sex = c['sex']
        age = c['age']
        sex_stats[sex]['n'] += 1
        age_stats[age]['n'] += 1
        if c['top5_with'] and c['top5_without'] and c['top5_with'][0] != c['top5_without'][0]:
            sex_stats[sex]['changed'] += 1
            age_stats[age]['changed'] += 1
        if true_in_topn(c, 1, 'top5_with'):
            sex_stats[sex]['with_top1'] += 1
            age_stats[age]['with_top1'] += 1
        if true_in_topn(c, 1, 'top5_without'):
            sex_stats[sex]['without_top1'] += 1
            age_stats[age]['without_top1'] += 1
    print("\nSubgroup analysis by sex:")
    for sex, stats in sex_stats.items():
        print(f"  Sex {sex}: {stats['n']} cases | dx changed: {stats['changed']} | with_top1: {stats['with_top1']} | without_top1: {stats['without_top1']}")
    print("\nSubgroup analysis by age (first 10 shown):")
    for age, stats in sorted(age_stats.items())[:10]:
        if stats['n'] > 0:
            print(f"  Age {age}: {stats['n']} cases | dx changed: {stats['changed']}/{stats['n']} ({stats['changed']/stats['n']:.1%}) | with_top1: {stats['with_top1']}/{stats['n']} ({stats['with_top1']/stats['n']:.1%}) | without_top1: {stats['without_top1']}/{stats['n']} ({stats['without_top1']/stats['n']:.1%})")
            # Only plot if there are cases and denominator is not zero
            try:
                fig = go.Figure(data=[go.Bar(
                    x=["changed", "with_top1", "without_top1"],
                    y=[stats['changed']/stats['n'], stats['with_top1']/stats['n'], stats['without_top1']/stats['n']],
                    marker_color=["#EF553B", "#636EFA", "#00CC96"]
                )])
                fig.update_layout(title=f"Age {age} subgroup stats", yaxis_title="Proportion", xaxis_title="Metric")
                fig.show()  # You can also use fig.write_html(f"age_{age}_subgroup_stats.html") to save as HTML
            except ZeroDivisionError:
                pass
        else:
            pass

    # 4. Visualizations
    # (A) Bar: Top-1 diagnosis changed vs unchanged
    fig1 = px.bar(x=["Changed", "Unchanged"], y=[dx_changed, dx_unchanged],
                  labels={'x':'Top-1 Diagnosis', 'y':'Count'},
                  title="Top-1 Diagnosis Change With vs Without Demographics")
    fig1.write_html("diagnosis_change_bar.html")
    # (B) Bar: True dx in top-N (with/without)
    if n_cases > 0:
        fig2 = go.Figure()
        for label, vals in acc.items():
            fig2.add_trace(go.Bar(name=f"{label.capitalize()} demographics",
                                  x=["Top-1", "Top-3", "Top-5"],
                                  y=[v/n_cases for v in vals]))
        fig2.update_layout(barmode='group',
                          title="True Diagnosis in Top-N",
                          yaxis_title="Proportion")
        fig2.write_html("true_dx_topn_bar.html")
    else:
        print("No cases to compare for true diagnosis in top-N bar plot. Skipping.")
    # (C) Subgroup: Sex
    if any(sex_stats[s]['n'] > 0 for s in sex_stats):
        fig3 = px.bar(x=list(sex_stats.keys()),
                      y=[sex_stats[s]['changed'] for s in sex_stats],
                      labels={'x':'Sex', 'y':'Diagnosis Changed Count'},
                      title="Diagnosis Change Count by Sex")
        fig3.write_html("dx_change_by_sex.html")
    else:
        print("No cases to compare for sex subgroup bar plot. Skipping.")
    print("\nSaved visualizations: diagnosis_change_bar.html, true_dx_topn_bar.html, dx_change_by_sex.html")


def analyze_cases(cases, conditions_mapping):
    changed_top1 = 0
    changed_top5 = 0
    subgroup_stats = defaultdict(lambda: Counter())
    max_activation = (None, -1)
    rare_diagnoses = Counter()
    changed_cases = []
    sex_counter = Counter()
    diag_counter = Counter()
    cat_counter = Counter()

    # Category-based stats
    category_stats = defaultdict(lambda: {'count': 0, 'changed_top5': 0})
    for c in cases:
        if not c["top5_with"] or not c["top5_without"]:
            continue
        sex_counter[c["sex"]] += 1
        diag_counter[c["diagnosis"]] += 1
        # Map diagnosis to category
        cat = get_category(c["diagnosis"], conditions_mapping)
        if cat:
            cat_counter[cat] += 1
            category_stats[cat]['count'] += 1
        # Top-1 change
        top1_changed = c["top5_with"][0] != c["top5_without"][0]
        # Top-5 change
        top5_changed = set(c["top5_with"]) != set(c["top5_without"])
        if top1_changed:
            changed_top1 += 1
        if top5_changed:
            changed_top5 += 1
            changed_cases.append(c)
        # Subgroup stats
        subgroup_stats[c["sex"]]["count"] += 1
        if top5_changed:
            subgroup_stats[c["sex"]]["changed_top5"] += 1
        subgroup_stats[c["age"]]["count"] += 1
        if top5_changed:
            subgroup_stats[c["age"]]["changed_top5"] += 1
        subgroup_stats[c["diagnosis"]]["count"] += 1
        if top5_changed:
            subgroup_stats[c["diagnosis"]]["changed_top5"] += 1
        if cat:
            if top5_changed:
                category_stats[cat]['changed_top5'] += 1
        # Max activation diff
        if c["activation_diff"] is not None and c["activation_diff"] > max_activation[1]:
            max_activation = (c, c["activation_diff"])
        # Rare diagnoses
        for dx in c["top5_with"] + c["top5_without"]:
            rare_diagnoses[dx] += 1


    now = datetime.now().strftime('%H%M_%y%m%d')
    out_txt = f"analysis_results_{now}.txt"


def main(output_path=None):
    conditions_mapping = load_conditions_mapping(CONDITIONS_JSON)
    # Build set of all valid diagnosis names (lowercase, including all aliases)
    valid_diagnoses = set()
    for k, v in conditions_mapping.items():
        valid_diagnoses.add(k.strip().lower())
        for alt in [v.get('cond-name-eng'), v.get('cond-name-fr')]:
            if alt:
                valid_diagnoses.add(alt.strip().lower())
if __name__ == "__main__":
    main()
