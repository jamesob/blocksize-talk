import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Set the seaborn theme
sns.set_theme(style="whitegrid")

# Define the ranges with the updated parameters
blocksizes = np.linspace(1, 675, 200)  # Block sizes from 1MB to 675MB
feerates = np.linspace(1, 2000, 200)   # Fee rates from 1 to 2000 sat/vB

# Create a meshgrid
X, Y = np.meshgrid(blocksizes, feerates)

# Calculate revenue (in BTC)
# BTC = MB * sat/vB * 1M / 100M
revenue_btc = (X * Y * 1_000_000) / 100_000_000

# Create the figure - single plot
fig, ax = plt.subplots(figsize=(12, 9))

# Define target revenue levels in BTC - include more levels for the expanded range
revenue_levels = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]

# Create the contour plot
contour = ax.contour(X, Y, revenue_btc, levels=revenue_levels, colors='darkblue', linewidths=1.5)
ax.clabel(contour, inline=True, fontsize=10, fmt='%g BTC')

# Fill between contours with colors
contourf = ax.contourf(X, Y, revenue_btc, levels=revenue_levels, alpha=0.3, cmap='viridis')

# Add a colorbar
cbar = plt.colorbar(contourf, ax=ax)
cbar.set_label('Miner Revenue (BTC)', rotation=270, labelpad=20, fontsize=12)

# Add labels and title
ax.set_xlabel('Block Size (MB)', fontsize=14)
ax.set_ylabel('Fee Rate (sat/vB)', fontsize=14)
ax.set_title('Bitcoin Miner Revenue: Block Size vs Fee Rate', fontsize=16)
ax.grid(True, alpha=0.3)

# Mark important block sizes
fixed_blocksize = 4  # MB
important_blocksizes = [1, 4, 32, 100, 675]

for bs in important_blocksizes:
if bs == fixed_blocksize:
    # Highlight the 4MB block size with a distinct color
    ax.axvline(x=bs, color='red', linestyle='--', linewidth=2.5,
              label=f'Fixed Block Size ({bs}MB)')
else:
    ax.axvline(x=bs, color='gray', linestyle=':', linewidth=1)
    ax.text(bs, 50, f"{bs}MB", ha='center', va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

# Draw horizontal lines at important fee rates
important_feerates = [10, 50, 100, 500, 1000, 2000]
for fr in important_feerates:
if fr in [10, 100, 1000]:  # Only highlight some fee rates with labels
    ax.axhline(y=fr, color='gray', linestyle=':', linewidth=1)
    ax.text(650, fr, f"{fr} sat/vB", va='center', ha='right',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

# For the fixed 4MB block size, add dots and labels where it intersects key revenue contours
for rev in revenue_levels:
# Calculate the fee rate needed for this revenue at 4MB
fee_needed = (rev * 100_000_000) / (fixed_blocksize * 1_000_000)
if 1 <= fee_needed <= 2000:  # Only annotate if within our range
    ax.plot(fixed_blocksize, fee_needed, 'ro', markersize=6)

    # Only label some points to avoid overcrowding
    if rev in [1, 10, 100, 1000]:
        ax.annotate(f"{rev} BTC", (fixed_blocksize, fee_needed),
                    xytext=(fixed_blocksize+30, fee_needed+100),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# Add a legend
ax.legend(loc='upper right', fontsize=12)

# Add a text block explaining how to read the chart
text_explanation = (
"How to read this chart:\n"
"• Each curve represents a constant revenue level\n"
"• Moving along a curve maintains the same revenue\n"
"• The red dashed line shows fixed 4MB block size approach\n"
"• Red dots show revenue points achievable with 4MB blocks"
)
plt.figtext(0.15, 0.02, text_explanation, fontsize=11,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

# Adjust layout
plt.tight_layout()
plt.savefig('bitcoin_revenue_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
