# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
# ]
# ///
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime, timedelta
import matplotlib.dates as mdates

WINDOW_MIN = 60
BUCKET_SEC = 20

df = pd.read_csv('event_log5.csv', usecols=['start_time_ns', 'duration', 'label', 'metadata'])

def size_mb(metadata):
    if pd.isna(metadata):
        return np.nan
    match = re.search(r'size_bytes=(\d+)', metadata)
    if match:
        return int(match.group(1)) / (1000**2)
    return np.nan

def getmeta(meta):
    metadata = {}

    if not isinstance(meta, str) or len(meta) < 1:
        return None

    for item in meta.split('|'):
        if item:
            k, v = item.split('=')

            if k == 'txids':
                v = v.split(',')

            metadata[k] = v
    return metadata


def getblockhash(meta):
    if not meta:
        return None
    return meta.get('blockhash') or meta.get('new')


df['size_mb'] = df['metadata'].apply(size_mb)
df['timestamp'] = df['start_time_ns'].apply(int)
df['meta'] = df['metadata'].apply(getmeta)
df['blockhash'] = df['meta'].apply(getblockhash)


txid_to_idx = {}

tx_rows = df[df['label'] == 'TX']
for i, event in tx_rows.iterrows():
    txid_to_idx[event['meta']['tx']] = i


NUM_BLOCK_IN_FILE = 16


connect_blocks_idxs = df[df['label'] == 'ConnectBlock'].index.tolist()
latest_cb = connect_blocks_idxs[NUM_BLOCK_IN_FILE]

print(f"Num blocks in file: {len(connect_blocks_idxs)}")

cb_row = df.loc[latest_cb]
tx_row_idxs = []

for txid in cb_row['meta']['txids']:
    if txid not in txid_to_idx:
        print(f"MISSING {txid}")
        continue
    tx_row_idxs.append(txid_to_idx[txid])

blockevent_idxs = df[
    ((df['label'] == 'BT') | (df['label'] == 'CMPCTBLOCK')) &
    (df['blockhash'] == cb_row['blockhash'])
].index.tolist()

block_df = df.loc[tx_row_idxs + blockevent_idxs]

min_time = block_df['timestamp'].min()
block_df['timestamp'] = block_df['timestamp'].apply(lambda t: pd.to_datetime(t - min_time, unit='ns'))

abc = df[
    (df['label'] == 'ActivateBestChain') & (df['blockhash'] == cb_row['blockhash'])
].iloc[0]

height = abc['meta']['height']
blockhash = "..." + abc['meta']['new'][-16:]



def map_label(label) -> str:
    return {
        'BT': 'BLOCKTXNS',
    }.get(label, label)

block_df['label'] = block_df['label'].apply(map_label)

total_size = block_df['size_mb'].sum()

print(f"Height: {height}, blockhash: {blockhash}, total size: {total_size}, num tx: {len(tx_row_idxs)}")


def plot_vertical_lines(df, timestamp_col='timestamp', size_col='size_mb', label_col='label',
                         blockhash_col='blockhash', figsize=(12, 8), alpha=0.7,
                         default_size=1):
    # Ensure timestamp is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])


    plt.rcParams.update({'font.size': 16})

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique labels for color mapping
    unique_labels = df[label_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_color_map = dict(zip(unique_labels, colors))

    # Track which labels actually appear in the plot
    used_labels = set()

    # Plot vertical lines for each row
    for idx, row in df.iterrows():
        timestamp = row[timestamp_col]

        # Handle missing size values
        if size_col in row and pd.notna(row[size_col]):
            size = row[size_col]
        else:
            ax.axvline(x=timestamp, ymin=0, ymax=1, color='red', linestyle='dashed', alpha=0.5)
            continue

        label = row[label_col]
        used_labels.add(label)

        line_color = label_color_map[label]

        # Plot vertical line from 0 to size
        ax.vlines(x=timestamp, ymin=0, ymax=size,
                 color=line_color, alpha=alpha, linewidth=1.5)

    # Create legend based on used labels
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=label_color_map[label], lw=2, label=label)
                       for label in used_labels]

    legend = ax.legend(handles=legend_elements, loc='upper right')

    # Get the legend's position
    legend_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
    legend_bottom = legend_bbox.y0  # Bottom of legend in figure coordinates

    # Create annotation text
    annotation_text = f"Total rx'd: {total_size:.3f} MB"

    # Add text box under the legend
    plt.figtext(
        0.76,  # x position (same as legend)
        legend_bottom - 0.05,  # y position (below legend)
        annotation_text,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'),
        fontsize=18
    )

    # Format x-axis for dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)

    # Set labels and title
    ax.set_xlabel('Relative time')
    ax.set_ylabel('Size (MB)')
    ax.set_title(f'Bandwidth history for block {blockhash} (height {height})')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    return fig, ax


plot_vertical_lines(block_df)

