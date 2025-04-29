import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 8))

# Define box positions and labels
boxes = [
    (0.5, 7.5, 'Start: Run local_analysis.py'),
    (0.5, 6.5, 'Load Model & SAE'),
    (0.5, 5.5, 'Load Patient Cases from CSV'),
    (0.5, 4.5, 'For Each Case (Loop)'),
    (0.5, 3.5, 'Generate Prompts (with/without demo)'),
    (0.5, 2.5, 'Run Model & SAE: Extract Activations'),
    (0.5, 1.5, 'Analyze Features (overlap, activation diff)'),
    (0.5, 0.5, 'Get Top 5 Diagnoses'),
    (3, 4.5, 'Collect Results'),
    (3, 3.5, 'After Loop: Summary Stats'),
    (3, 2.5, 'Subgroup Analysis (sex, age)'),
    (3, 1.5, 'Condition-Specific Analysis'),
    (3, 0.5, 'Plot Results (interactive HTML)'),
    (5.5, 2.5, 'Save Output Files (TXT, HTML)'),
    (7, 2.5, 'End'),
]

# Draw boxes
for x, y, label in boxes:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle="round,pad=0.1", fc="lightblue", ec="navy", lw=2))
    ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

# Draw arrows
arrowprops = dict(arrowstyle="->", color='black', lw=1.5)
arrows = [
    ((0.5, 7.2), (0.5, 6.8)),
    ((0.5, 6.2), (0.5, 5.8)),
    ((0.5, 5.2), (0.5, 4.8)),
    ((0.5, 4.2), (0.5, 3.8)),
    ((0.5, 3.2), (0.5, 2.8)),
    ((0.5, 2.2), (0.5, 1.8)),
    ((0.5, 1.2), (0.5, 0.8)),
    ((0.5, 0.5), (3, 4.5)),
    ((3, 4.2), (3, 3.8)),
    ((3, 3.2), (3, 2.8)),
    ((3, 2.2), (3, 1.8)),
    ((3, 1.2), (3, 0.8)),
    ((3, 0.5), (5.5, 2.5)),
    ((5.9, 2.5), (6.7, 2.5)),
]
for start, end in arrows:
    ax.annotate('', xy=end, xytext=start, arrowprops=arrowprops)

ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.axis('off')
plt.tight_layout()
plt.savefig('/Users/ameliag/Downloads/alethia-main/local_analysis_pipeline.png', dpi=200)
