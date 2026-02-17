import matplotlib.pyplot as plt

# Model names
models = ['Logistic Regression', 'BERT']

# Replace with your actual values
accuracies = [98.78, 99.98]

plt.figure()
plt.bar(models, accuracies)

plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison of Models")
plt.ylim(95, 100)

plt.savefig("accuracy_comparison.png")
plt.show()
