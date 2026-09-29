"""
Generates a synthetic e-commerce customer dataset for RFM-style segmentation:
Recency (days since last purchase), Frequency (number of orders),
Monetary (total spend) - the standard features used for customer clustering.
"""
import numpy as np
import pandas as pd

np.random.seed(11)
N = 800

# Build 4 natural customer groups, then shuffle so the clustering has to find them
segments_def = [
    # (n, recency_range, frequency_range, monetary_range, label for reference only)
    (200, (1, 20), (15, 40), (2000, 6000), "Loyal High-Spenders"),
    (220, (5, 40), (5, 15), (500, 2000), "Regular Customers"),
    (180, (60, 180), (1, 4), (50, 400), "At-Risk / Churned"),
    (200, (1, 30), (1, 3), (800, 3000), "New Big Spenders"),
]

rows = []
cust_id = 5000
for n, rec_range, freq_range, mon_range, _ in segments_def:
    recency = np.random.randint(rec_range[0], rec_range[1], n)
    frequency = np.random.randint(freq_range[0], freq_range[1], n)
    monetary = np.round(np.random.uniform(mon_range[0], mon_range[1], n), 2)
    for r, f, m in zip(recency, frequency, monetary):
        rows.append({
            "Customer_ID": f"CUST-{cust_id}",
            "Recency_Days": int(r),
            "Frequency_Orders": int(f),
            "Monetary_Spend": float(m),
        })
        cust_id += 1

df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)

# add a bit of noise/messiness
df.loc[df.sample(15, random_state=2).index, "Monetary_Spend"] = np.nan
dupe_rows = df.sample(8, random_state=3)
df = pd.concat([df, dupe_rows], ignore_index=True)

df.to_csv("data/customers.csv", index=False)
print(f"Generated {len(df)} rows -> data/customers.csv")
