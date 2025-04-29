import re

RESULTS_FILE = "/Users/ameliag/Downloads/5000caseslocalanalysisresults250422.txt"

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

def print_diff_cases(cases):
    diff_cases = [c for c in cases if c["diagnosis_with"] and c["diagnosis_without"] and c["diagnosis_with"] != c["diagnosis_without"]]
    total = len(cases)
    diff = len(diff_cases)
    print(f"Number of cases where diagnosis was different: {diff}/{total} ({(diff/total*100 if total else 0):.1f}%)\n")
    if diff > 0:
        print("Cases where diagnosis is different with vs. without demographics:")
        for c in diff_cases:
            print(f"Age: {c['age']}, Sex: {c['sex']}, With demographics: {c['diagnosis_with']}, Without demographics: {c['diagnosis_without']}")
    print("\nFirst 10 parsed cases (for debugging):")
    for c in cases[:10]:
        print(f"Age: {c['age']}, Sex: {c['sex']}, With demographics: {c['diagnosis_with']}, Without demographics: {c['diagnosis_without']}")

if __name__ == "__main__":
    cases = parse_cases(RESULTS_FILE)
    print_diff_cases(cases)
