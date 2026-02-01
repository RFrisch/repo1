# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 18:51:15 2025

@author: skarb
"""

import matplotlib.pyplot as plt
import pandas as pd

# Data extracted from the image
data = {
    "Ticker": ["AMSF", "AFG", "ACGL", "CB", "EG", "IFC", "IGIC", "KNSL", "ORI", "PLMR", "RLI", "SIGI", "ALL", "HIG", "PGR", "TRV", "WRB"],
    "ROE_FY": [20.17, 20.33, 22.01, 14.60, 10.14, 13.34, 22.61, 32.28, 14.18, 19.59, 23.56, 6.82, 23.59, 19.58, 35.41, 18.94, 22.10],
    "PB_LTM": [3.4, 2.2, 1.6, 1.9, 1.1, 3.0, 1.5, 7.0, 1.5, 5.1, 4.3, 1.7, 2.4, 1.9, 5.6, 2.3, 2.9]
}

df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(df["PB_LTM"], df["ROE_FY"], color='royalblue')

# Annotate points with tickers
for i in range(len(df)):
    plt.text(df["PB_LTM"][i] + 0.05, df["ROE_FY"][i], df["Ticker"][i], fontsize=9)

plt.xlabel("P/B (LTM)")
plt.ylabel("Return on Equity % (FY)")
plt.title("Return on Equity vs. P/B Ratio for Insurance Companies")
plt.grid(True)
plt.tight_layout()
plt.show()
