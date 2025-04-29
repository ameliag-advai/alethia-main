import re
from collections import Counter, defaultdict

RESULTS_FILE = "1000caseslocalanalysisresults220425.txt"  # Change this to your results file

def parse_cases(filename):
    cases = []
    with open(filename, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("Analyzing case:"):
            age = sex = diagnosis = dx_with = dx_without = None
            # Parse age, sex, diagnosis from the NEXT line
            if i + 1 < len(lines):
                next_line = lines[i+1]
                age_match = re.search(r"Age:\s*(\d+)", next_line)
                sex_match = re.search(r"Sex:\s*([MF])", next_line)
                dx_match = re.search(r"Diagnosis:\s*(.+)", next_line)
                if age_match: age = int(age_match.group(1))
                if sex_match: sex = sex_match.group(1)
                if dx_match: diagnosis = dx_match.group(1).strip()
            # Scan forward for the next 'Text with demographics:' and 'Text without demographics:'
            j = i + 1
            while j < len(lines) and (dx_with is None or dx_without is None) and j < i + 20:
                if lines[j].strip().lower().startswith("text with demographics"):
                    dx_with_match = re.search(r"diagnosed with (.+?)\.", lines[j], re.IGNORECASE)
                    if dx_with_match:
                        dx_with = dx_with_match.group(1).strip()
                elif lines[j].strip().lower().startswith("text without demographics"):
                    dx_without_match = re.search(r"diagnosed with (.+?)\.", lines[j], re.IGNORECASE)
                    if dx_without_match:
                        dx_without = dx_without_match.group(1).strip()
                j += 1
            cases.append({
                "age": age,
                "sex": sex,
                "diagnosis_with": dx_with,
                "diagnosis_without": dx_without,
                "line": lines[i].strip()
            })
        i += 1
    return cases

def analyze_cases(cases):
    diff_cases = [c for c in cases if c["diagnosis_with"] != c["diagnosis_without"] and c["diagnosis_with"] and c["diagnosis_without"]]
    total = len(cases)
    diff = len(diff_cases)
    print(f"Number of cases where diagnosis was different: {diff}/{total}, {diff/total*100 if total else 0:.1f}%")
    # Sex split
    sex_counter = Counter(c["sex"] for c in diff_cases)
    print("Sex split of cases where diagnosis changed:", dict(sex_counter))
    # Age split
    over_65 = sum(1 for c in diff_cases if c["age"] is not None and c["age"] >= 65)
    under_45 = sum(1 for c in diff_cases if c["age"] is not None and c["age"] < 45)
    print(f"Age split: {over_65/diff*100 if diff else 0:.1f}% over 65, {under_45/diff*100 if diff else 0:.1f}% under 45")
    # Most common changes per demographic
    changes_by_sex = defaultdict(list)
    changes_by_age_group = defaultdict(list)
    for c in diff_cases:
        changes_by_sex[c["sex"]].append((c["diagnosis_without"], c["diagnosis_with"]))
        if c["age"] is not None:
            age_group = "under_45" if c["age"] < 45 else "45_to_64" if c["age"] < 65 else "65_plus"
            changes_by_age_group[age_group].append((c["diagnosis_without"], c["diagnosis_with"]))
    print("\nMost common changes by sex:")
    for sex, changes in changes_by_sex.items():
        counter = Counter(changes)
        most_common = counter.most_common(3)
        print(f"{sex}: {most_common}")
    print("\nMost common changes by age group:")
    for group, changes in changes_by_age_group.items():
        counter = Counter(changes)
        most_common = counter.most_common(3)
        print(f"{group}: {most_common}")

if __name__ == "__main__":
    cases = parse_cases(RESULTS_FILE)
    analyze_cases(cases)
