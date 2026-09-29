# Customer Segmentation with K-Means Clustering (RFM Analysis)

Groups customers into behavioral segments using Recency, Frequency and
Monetary (RFM) values, so marketing and retention efforts can be targeted
differently per group instead of treating all customers the same.

## Problem
An e-commerce business wants to know: which customers are loyal and valuable,
which are at risk of churning, and which are new high-value customers worth
nurturing — using only order history, no manual tagging.

## Dataset
`data/customers.csv` — 808 customers with `Recency_Days` (days since last
order), `Frequency_Orders` (number of orders), and `Monetary_Spend` (total
spend). (Synthetic data, generated with `generate_data.py`, built from four
natural customer groups so the clustering has real structure to find.)

## Approach
1. **Clean** — removed duplicates, imputed missing spend values with the median.
2. **Scale** — standardized features with `StandardScaler`, since K-Means is
   distance-based and Monetary_Spend's larger numeric range would otherwise dominate.
3. **Elbow method** — ran K-Means for k = 1 to 8 and plotted inertia to choose k = 4.
4. **K-Means clustering** — fit the final model and labeled each cluster
   based on its average RFM profile.
5. **Visualization** — scatter plots across RFM pairs, bar chart of cluster averages.

## Key Findings
- Four clear segments emerge: **Loyal High-Spenders**, **Regular Customers**,
  **At-Risk/Churned**, and **New Big Spenders**.
- At-risk customers (high recency, low frequency) are a clear win-back target.
- Loyal high-spenders are a small but valuable group worth a retention program.
- Feature scaling was essential — without it, spend alone would have driven the clustering.

## Tools
Python · Pandas · Scikit-learn (KMeans, StandardScaler) · Matplotlib · Seaborn

## Files
- `customer_segmentation.ipynb` — full analysis notebook
- `generate_data.py` — synthetic dataset generator
- `data/customers.csv` — the dataset
- `charts/` — exported chart images

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook customer_segmentation.ipynb
```
