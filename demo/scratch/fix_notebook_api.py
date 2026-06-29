import json
from pathlib import Path

notebook_path = Path("d:/Python/AIN701_Group_04/demo/notebooks/04_LLM_Explainer.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        for i, line in enumerate(source):
            if "explain_loan_decision(result, top_features)" in line:
                source[i] = line.replace("explain_loan_decision(result, top_features)", "explain_loan_decision(result, top_features, raw_input)")
            if "build_explanation_prompt(r, tf)" in line:
                source[i] = line.replace("build_explanation_prompt(r, tf)", "build_explanation_prompt(r, tf, ri)")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
