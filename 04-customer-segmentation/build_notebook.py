import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbclient import NotebookClient

nb = new_notebook()
cells = []

cells.append(new_markdown_cell(
"""# Customer Segmentation with K-Means Clustering (RFM Analysis)

**Goal:** Group customers into meaningful segments using Recency, Frequency
and Monetary (RFM) behavior, so marketing can target each group differently
— e.g. reward loyal customers, win back at-risk ones.

**Dataset:** `data/customers.csv` — 808 synthetic customers with:
- `Recency_Days` — days since their last order
- `Frequency_Orders` — number of orders placed
- `Monetary_Spend` — total amount spent

**Tools:** Python, Pandas, Scikit-learn (KMeans, StandardScaler), Matplotlib/Seaborn

**Sections:**
1. Load & clean the data
2. Feature scaling
3. Finding the right number of clusters (elbow method)
4. K-Means clustering
5. Interpreting the segments
6. Key findings
"""))

cells.append(new_markdown_cell("## 1. Load & Clean the Data"))
cells.append(new_code_cell(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

df = pd.read_csv("data/customers.csv")
print("Raw shape:", df.shape)
df.head()"""))

cells.append(new_code_cell(
"""print("Duplicates:", df.duplicated().sum())
print("Nulls:\\n", df.isnull().sum()[df.isnull().sum() > 0])"""))

cells.append(new_code_cell(
"""df = df.drop_duplicates()
df["Monetary_Spend"] = df["Monetary_Spend"].fillna(df["Monetary_Spend"].median())
print("Cleaned shape:", df.shape)
df[["Recency_Days", "Frequency_Orders", "Monetary_Spend"]].describe().round(1)"""))

cells.append(new_markdown_cell(
"""## 2. Feature Scaling

K-Means uses distance between points, so features need to be on the same
scale — otherwise Monetary_Spend (in thousands) would dominate Recency_Days
(in tens) just because of its larger numbers."""))

cells.append(new_code_cell(
"""features = ["Recency_Days", "Frequency_Orders", "Monetary_Spend"]
X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled[:5]"""))

cells.append(new_markdown_cell(
"""## 3. Finding the Right Number of Clusters (Elbow Method)

Running K-Means for a range of k values and plotting the within-cluster sum
of squares (inertia) — the "elbow" in the curve is a reasonable choice for k."""))

cells.append(new_code_cell(
"""inertias = []
k_range = range(1, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots()
ax.plot(list(k_range), inertias, marker="o")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
ax.set_title("Elbow Method for Optimal k")
plt.tight_layout()
plt.savefig("charts/elbow_method.png", dpi=120)
plt.show()"""))

cells.append(new_markdown_cell("The elbow flattens out around **k = 4**, so that's the value used below."))

cells.append(new_markdown_cell("## 4. K-Means Clustering"))
cells.append(new_code_cell(
"""kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
df["Cluster"].value_counts().sort_index()"""))

cells.append(new_markdown_cell("## 5. Interpreting the Segments"))
cells.append(new_code_cell(
"""cluster_profile = df.groupby("Cluster")[features].mean().round(1)
cluster_profile["Customer_Count"] = df["Cluster"].value_counts().sort_index()
cluster_profile"""))

cells.append(new_code_cell(
"""# Label clusters based on their profile (recency low = recent, monetary high = big spender)
def label_cluster(row):
    if row.Recency_Days > 60:
        return "At-Risk / Churned"
    elif row.Frequency_Orders >= 15:
        return "Loyal High-Spenders"
    elif row.Monetary_Spend >= 800 and row.Frequency_Orders < 5:
        return "New Big Spenders"
    else:
        return "Regular Customers"

cluster_profile["Segment_Label"] = cluster_profile.apply(label_cluster, axis=1)
cluster_profile"""))

cells.append(new_code_cell(
"""fig, axes = plt.subplots(1, 3, figsize=(16, 5))
pairs = [("Recency_Days", "Frequency_Orders"),
         ("Frequency_Orders", "Monetary_Spend"),
         ("Recency_Days", "Monetary_Spend")]
for ax, (x, y) in zip(axes, pairs):
    sns.scatterplot(data=df, x=x, y=y, hue="Cluster", palette="Set2", alpha=0.7, ax=ax, legend=(x==pairs[0][0]))
    ax.set_title(f"{x} vs {y}")
plt.tight_layout()
plt.savefig("charts/cluster_scatter_pairs.png", dpi=120)
plt.show()"""))

cells.append(new_code_cell(
"""fig, ax = plt.subplots()
cluster_profile[features].plot(kind="bar", ax=ax)
ax.set_title("Average RFM Values per Cluster")
ax.set_xlabel("Cluster")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("charts/cluster_profile_bar.png", dpi=120)
plt.show()"""))

cells.append(new_markdown_cell(
"""## 6. Key Findings

- K-Means with k=4 cleanly separates customers into **Loyal High-Spenders**,
  **Regular Customers**, **At-Risk/Churned**, and **New Big Spenders** —
  confirmed by the elbow method and the cluster profile table.
- **At-Risk customers** have high recency (haven't purchased recently) and
  low frequency — a clear list to target with a win-back campaign.
- **Loyal High-Spenders** are a small but valuable group — worth a loyalty
  or VIP program to retain them.
- **New Big Spenders** placed few orders but spent a lot per order — good
  candidates for a "welcome" cross-sell campaign to increase order frequency.
- Scaling the features before clustering was essential — without it,
  Monetary_Spend (largest numeric range) would have dominated the distance
  calculation and distorted the clusters.

*(Findings are illustrative, generated from a synthetic dataset built with
deliberately separated customer groups — the RFM + K-Means approach applies
directly to a real order-history export.)*
"""))

nb["cells"] = cells

client = NotebookClient(nb, timeout=120, kernel_name="python3")
client.execute()

with open("customer_segmentation.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook built and executed successfully.")
