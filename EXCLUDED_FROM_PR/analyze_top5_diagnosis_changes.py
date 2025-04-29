import re
import csv
import json
from collections import defaultdict, Counter
from datetime import datetime

CONDITIONS_JSON = "/Users/ameliag/Downloads/alethia-main/DDXPlus Dataset 2/release_conditions.json"

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
        if lines[i].strip().startswith("Analyzing case:") or lines[i].strip().startswith("--- CASE"):
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
    # Prepare CSV rows
    csv_rows = []
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
        # Also store category in CSV
        csv_rows.append({
            'age': c['age'],
            'sex': c['sex'],
            'diagnosis': c['diagnosis'],
            'category': cat,
            'changed_top1': top1_changed,
            'changed_top5': top5_changed,
            'activation_diff': c['activation_diff'],
            'top5_with': '|'.join(c['top5_with']),
            'top5_without': '|'.join(c['top5_without'])
        })

    now = datetime.now().strftime('%H%M_%y%m%d')
    out_txt = f"analysis_results_{now}.txt"
    out_csv = f"analysis_results_{now}.csv"
    with open(out_txt, 'w') as f:
        f.write(f"Number of cases where TOP-1 diagnosis changed: {changed_top1}/{len(cases)}\n")
        f.write(f"Number of cases where TOP-5 set changed: {changed_top5}/{len(cases)}\n\n")
        f.write("Subgroup summary (sex):\n")
        for sex in ['M', 'F', None]:
            total = subgroup_stats[sex]['count']
            changed = subgroup_stats[sex]['changed_top5']
            f.write(f"  Sex {sex}: {changed}/{total} cases had a top-5 change ({(changed/total*100 if total else 0):.1f}%)\n")
        # Separate ages and diagnoses for correct sorting
        ages = [k for k in subgroup_stats if isinstance(k, int) and subgroup_stats[k]['count'] > 10]
        diags = [k for k in subgroup_stats if isinstance(k, str) and subgroup_stats[k]['count'] > 10]
        f.write("\nSubgroup summary (age): (showing ages with >10 cases)\n")
        for age in sorted(ages):
            total = subgroup_stats[age]['count']
            changed = subgroup_stats[age]['changed_top5']
            f.write(f"  Age {age}: {changed}/{total} top-5 changes\n")
        f.write("\nSubgroup summary (diagnosis): (showing diagnoses with >10 cases)\n")
        for diag in sorted(diags):
            total = subgroup_stats[diag]['count']
            changed = subgroup_stats[diag]['changed_top5']
            f.write(f"  Dx {diag}: {changed}/{total} top-5 changes\n")
        # Category summary
        f.write("\nSubgroup summary (category/ICD-10): (showing categories with >10 cases)\n")
        for cat, stats in sorted(category_stats.items()):
            if stats['count'] > 10:
                f.write(f"  Category {cat}: {stats['changed_top5']}/{stats['count']} top-5 changes\n")
        f.write("\nEdge case: Largest activation difference:\n")
        if max_activation[0]:
            c = max_activation[0]
            f.write(f"  Age: {c['age']}, Sex: {c['sex']}, Dx: {c['diagnosis']}, Activation diff: {c['activation_diff']}\n")
            f.write(f"  Top-5 WITH: {c['top5_with']}\n")
            f.write(f"  Top-5 WITHOUT: {c['top5_without']}\n")
        f.write("\n5 rarest diagnoses in top-5:\n")
        for dx, count in rare_diagnoses.most_common()[-5:]:
            f.write(f"  {dx}: {count} appearances in top-5\n")
        f.write("\n\nFull details of cases where top-5 changed:\n")
        for c in changed_cases:
            f.write(f"Age: {c['age']}, Sex: {c['sex']}, Dx: {c['diagnosis']}, Activation diff: {c['activation_diff']}\n")
            f.write(f"  Top-5 WITH: {c['top5_with']}\n")
            f.write(f"  Top-5 WITHOUT: {c['top5_without']}\n\n")
        # Debug: show sex and diagnosis counts
        f.write("\nSex count breakdown (all cases):\n")
        for k, v in sex_counter.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nDiagnosis count breakdown (all cases):\n")
        for k, v in diag_counter.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nCategory count breakdown (all cases):\n")
        for k, v in cat_counter.items():
            f.write(f"  {k}: {v}\n")
    # Write CSV for visualization
    with open(out_csv, 'w', newline='') as csvfile:
        fieldnames = ['age', 'sex', 'diagnosis', 'category', 'changed_top1', 'changed_top5', 'activation_diff', 'top5_with', 'top5_without']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"Analysis written to {out_txt} and {out_csv}")

import glob
import os

def get_latest_results_file():
    files = sorted(glob.glob(os.path.expanduser("~/Downloads/alethia-main/analysis_results_*.txt")), key=os.path.getmtime, reverse=True)
    for f in files:
        try:
            with open(f, 'r') as file:
                # Look for a line indicating case data
                for line in file:
                    if '--- CASE' in line or 'Analyzing case:' in line:
                        return f
        except Exception as e:
            continue
    raise FileNotFoundError("No analysis_results_*.txt files with case data found in ~/Downloads/alethia-main/")

def main():
    conditions_mapping = load_conditions_mapping(CONDITIONS_JSON)
    # Build set of all valid diagnosis names (lowercase, including all aliases)
    valid_diagnoses = set()
    for k, v in conditions_mapping.items():
        valid_diagnoses.add(k.strip().lower())
        for alt in [v.get('cond-name-eng'), v.get('cond-name-fr')]:
            if alt:
                valid_diagnoses.add(alt.strip().lower())
    latest_file = get_latest_results_file()
    print(f"[INFO] Using latest analysis results file: {latest_file}")
    cases = parse_cases(latest_file, valid_diagnoses=valid_diagnoses)
    # Limit to first 1500 cases
    cases = cases[:1500]
    analyze_cases(cases, conditions_mapping)

if __name__ == "__main__":
    main()

