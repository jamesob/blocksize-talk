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
from matplotlib.lines import Line2D
from datetime import datetime, timedelta

WINDOW_MIN = 60
BUCKET_SEC = 20
OFFSET_MIN = 70

df = pd.read_csv('event_log3.csv', usecols=['start_time_ns', 'duration', 'label', 'metadata'])

def extract_size_bytes(metadata):
    if pd.isna(metadata):
        return np.nan
    match = re.search(r'size_bytes=(\d+)', metadata)
    if match:
        return int(match.group(1)) / (1000**2)
    return np.nan

df['size_mb'] = df['metadata'].apply(extract_size_bytes)
df['timestamp'] = pd.to_datetime(df['start_time_ns'], unit='ns')

def map_label(label) -> str:
    return {
        'BT': 'BLOCKTXNS',
    }.get(label, label)


latest_timestamp = df['timestamp'].max() - timedelta(minutes=OFFSET_MIN)
min_ago = latest_timestamp - timedelta(minutes=WINDOW_MIN) - timedelta(minutes=OFFSET_MIN)
df = df[df['timestamp'] < latest_timestamp]
df = df[df['timestamp'] >= min_ago]

df['time_bucket'] = df['timestamp'].dt.floor(f'{BUCKET_SEC}s')
df['label'] = df['label'].apply(map_label)

df_with_size = df.dropna(subset=['size_mb'])

plt.rcParams.update({'font.size': 16})


def getmeta(event) -> dict:
    metadata = {}
    for item in event['metadata'].split('|'):
        if item:
            k, v = item.split('=')
            metadata[k] = v
    return metadata


grouped = df_with_size.groupby(['time_bucket', 'label'])['size_mb'].sum().unstack()
grouped = grouped.fillna(0)

fig, ax = plt.subplots(figsize=(12, 8))

labels = df_with_size['label'].unique()
cmap = plt.cm.get_cmap('Set1', len(labels))
colors = {
    label: cmap(i) for i, label in
    enumerate(['BLOCKTXNS', 'CMPCTBLOCK', 'HEADERS', 'INV', 'TX'])
}

x_labels = [ts.strftime('%H:%M:%S') for ts in grouped.index]

bottom = np.zeros(len(grouped))
for label in grouped.columns:
    if label in colors:
        ax.bar(x_labels, grouped[label], bottom=bottom, label=label, color=colors[label])
        bottom += grouped[label].values

ax.set_xlabel(f'Time ({BUCKET_SEC}-second buckets)')
ax.set_ylabel('Data received (MB)')
ax.set_title(f"Stacked data rx'd by message type (last {WINDOW_MIN} min)")


def add_event_line(ax, event_time, grouped):
    # Convert event_time to pandas timestamp if it's not already
    if not isinstance(event_time, pd.Timestamp):
        event_time = pd.to_datetime(event_time)

    # Find the closest time bucket in your data
    closest_bucket = None
    min_diff = pd.Timedelta.max

    for bucket in grouped.index:
        diff = abs(bucket - event_time)
        if diff < min_diff:
            min_diff = diff
            closest_bucket = bucket

    if closest_bucket is not None:
        # Get the index position of this bucket in your x-labels
        x_labels = [ts.strftime('%H:%M:%S') for ts in grouped.index]
        position = x_labels.index(closest_bucket.strftime('%H:%M:%S'))

        # Now use this position for axvline
        ax.axvline(x=position, color='red', linestyle='dashed', label='_nolegend_', alpha=0.6)


abc_rows = df[df['label'] == 'ActivateBestChain']
print(f"found {len(abc_rows)} ACTIVATES")
if not abc_rows.empty:
    for _, event in abc_rows.iterrows():
        if 'new=' not in str(event['metadata']):
            # Not a new block
            continue
        metadata = getmeta(event)
        blockhash = metadata.get('new', 'unknown')[-12:]
        height = metadata.get('height', 'unknown')

        event_time = event['time_bucket']

        add_event_line(ax, event_time, grouped)

# Get the current handles and labels from the automatically created legend
handles, labels = ax.get_legend_handles_labels()

# Create a custom handle for the event line
event_line = Line2D([0], [0], color='red', linestyle='dashed', alpha=0.6)

# Add the custom handle to the list of handles and add a label
handles.append(event_line)
labels.append('Block received')

# Create a new legend with our custom entries included
ax.legend(handles=handles, labels=labels, title='Message type', loc='upper right')

# Set font size for axis tick labels
ax.tick_params(axis='both', which='major', labelsize=12)

# Show fewer x-axis labels by showing only every nth label
n = max(1, len(x_labels) // 10)  # Show approximately 10 labels
plt.xticks(range(0, len(x_labels), n), [x_labels[i] for i in range(0, len(x_labels), n)], rotation=45, ha='right')

ax.legend(title='Message type')
plt.tight_layout()

plt.show()
