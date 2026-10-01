import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data

df = pd.read_csv("AG_NO3_fill_cells_remove_NAN-2.csv")

# The assignment calls this the Nitrate column.
# In the provided CSV, the actual column name is NO3N.
x = df["NO3N"].to_numpy()

# Ground-truth labels contained in the CSV
actual = df["Student_Flag"].to_numpy().astype(int)

# 2. Parameters

W = 650
q = 90

predicted = np.zeros(len(x), dtype=int)
thresholds = np.full(len(x), np.nan)

# 3. First window

T1 = np.percentile(x[0:W], q, method="linear")

# Label every point in the first window
predicted[0:W] = (x[0:W] >= T1).astype(int)
thresholds[0:W] = T1

# 4. Remaining overlapping windows
#    step size = 1

for start in range(1, len(x) - W + 1):

    end = start + W

    # Current fixed-size window
    window = x[start:end]

    # Adaptive percentile threshold
    Ti = np.percentile(window, q, method="linear")

    # Only classify the newly added point
    new_index = end - 1

    predicted[new_index] = int(x[new_index] >= Ti)
    thresholds[new_index] = Ti

# 5. Confusion matrix

TP = np.sum((predicted == 1) & (actual == 1))
FP = np.sum((predicted == 1) & (actual == 0))
FN = np.sum((predicted == 0) & (actual == 1))
TN = np.sum((predicted == 0) & (actual == 0))

P = TP + FN
N = TN + FP

normal_accuracy = TN / N
anomaly_accuracy = TP / P

print("Window size W:", W)
print("Percentile q:", q)
print()

print("Actual anomaly rows:", P)
print("Actual normal rows:", N)
print()

print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("TN:", TN)
print()

print(f"Normal event detection accuracy: {normal_accuracy:.4f}")
print(f"Anomaly event detection accuracy: {anomaly_accuracy:.4f}")
print()

print(f"Normal accuracy: {normal_accuracy * 100:.2f}%")
print(f"Anomaly accuracy: {anomaly_accuracy * 100:.2f}%")

# 6. Plot

dates = pd.to_datetime(
    df["Date"],
    format="%m/%d/%Y %H:%M"
)

predicted_indices = np.where(predicted == 1)[0]
actual_indices = np.where(actual == 1)[0]

plt.figure(figsize=(14, 6))

# Original nitrate time series
plt.plot(
    dates,
    x,
    linewidth=0.7,
    color="steelblue",
    label="Nitrate (NO3N)"
)

# Sliding-window threshold
plt.plot(
    dates,
    thresholds,
    linewidth=1.2,
    color="orange",
    label=f"{q}th Percentile Threshold"
)

# Anomalies detected by our method
plt.scatter(
    dates.iloc[predicted_indices],
    x[predicted_indices],
    marker="x",
    s=18,
    color="red",
    label="Predicted Anomaly"
)

# Ground-truth anomalies from Student_Flag
plt.scatter(
    dates.iloc[actual_indices],
    x[actual_indices],
    facecolors="none",
    edgecolors="black",
    s=45,
    linewidths=1.2,
    label="Ground Truth"
)

plt.xlabel("Date")
plt.ylabel("NO3N")
plt.title(
    f"Nitrate Anomaly Detection Using Sliding Windows "
    f"(W={W}, q={q})"
)

plt.legend()
plt.tight_layout()

plt.savefig("anomaly_detection.png", dpi=300)
plt.show()