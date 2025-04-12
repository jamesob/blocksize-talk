import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the seaborn theme
sns.set_theme()

# Define the ranges - reduce number of points for cleaner display with annotations
blocksizes = np.array([4,  32,  64, 100,  200, 675])
feerates =   np.array([1, 2, 4, 128, 256, 1024, 5000, 10_000])

# Create a revenue dataframe (converting satoshis to BTC)
# 1 BTC = 100,000,000 satoshis
revenue_data = np.zeros((len(feerates), len(blocksizes)))
for i, fee in enumerate(feerates):
    for j, block in enumerate(blocksizes):
        # Calculate revenue: block size (MB) × fee rate (sat/vB) × 1,000,000 (bytes) / 100,000,000 (sats per BTC)
        revenue_data[i, j] = (block * fee * 1_000_000) / 100_000_000

# Create the figure
f, ax = plt.subplots(figsize=(9, 6))

# Create the heatmap with annotations
# Format with 2 decimal places for BTC values
sns.heatmap(revenue_data,
            annot=True,
            fmt=".1f",
            linewidths=.5,
            ax=ax,
            xticklabels=[int(x) for x in blocksizes],
            yticklabels=[int(y) for y in feerates],
            cmap="YlOrRd",
            )

# Add labels and title
ax.set_title('Bitcoin Miner Revenue Heatmap')
ax.set_xlabel('Block Size (MvB)')
ax.set_ylabel('Fee Rate (sat/vB)')

# Adjust the y-axis to have 0 at the bottom
ax.invert_yaxis()

# Add a colorbar label
cbar = ax.collections[0].colorbar
cbar.set_label('Miner Revenue (BTC)', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('bitcoin_miner_revenue_heatmap.png', dpi=300)
plt.show()
