import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Your confusion matrix values
cm = np.array([[3507, 60],
               [35, 4219]])

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")

plt.savefig("logistic_confusion.png")
plt.show()
