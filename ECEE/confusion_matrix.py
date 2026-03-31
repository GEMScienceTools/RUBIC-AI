import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

input_file = "Comparision_RUBIC_AI_vs_MANUAL.csv"
feature = "roof_material"
# pred = feature+"_RUBIC_AI"
pred = feature+"_RUBIC_AI"
df = pd.read_csv(input_file)
output_file = feature+"_console.png"

# Drop rows where either label is missing
df = df.dropna(subset=[feature, pred])

y_true = df[feature]
y_pred = df[pred]

labels = sorted(df[feature].unique())

cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Raw counts
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=axes[0], colorbar=True, xticks_rotation=45, cmap="Blues")
axes[0].set_title("Console - Confusion Matrix")

# Normalized per row
disp_norm = ConfusionMatrixDisplay(confusion_matrix=np.round(cm_normalized, 2), display_labels=labels)
disp_norm.plot(ax=axes[1], colorbar=True, xticks_rotation=45, cmap="Blues")
axes[1].set_title("Console - Normalized Confusion Matrix")

plt.tight_layout()
plt.savefig(output_file, dpi=150)
print(f"Saved: {output_file}")
