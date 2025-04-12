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
import matplotlib.patches as patches
import json
import numpy as np

# Read the CSV file
df = pd.read_csv('event_log3.csv', usecols=['start_time_ns', 'duration', 'label', 'metadata'])

abc_rows = df[df['label'] == 'ActivateBestChain']

# Get one minute after the latest ABC
max_time = abc_rows['start_time_ns'].max() + 60 * 1e9

TIME_WINDOW_MIN = 10
window_ns = TIME_WINDOW_MIN * 60 * 1e9
min_time = max_time - window_ns
df = df[df['start_time_ns'] >= min_time]
df = df[df['start_time_ns'] < max_time]

# Convert start_time_ns and duration to numeric if they're not already
df['start_time_ns'] = pd.to_numeric(df['start_time_ns'])
df['duration'] = pd.to_numeric(df['duration'])

min_visible_duration = df['duration'].max() * 0.01
print(f"Assuming min visible duration as {min_visible_duration}")

small_events = df[df['duration'] < min_visible_duration]
if not small_events.empty:
    print(f"Upscaling {len(small_events)} events, which are too small to see")

df['visible_duration'] = df['duration'].apply(lambda x: max(x, 100_000))

# Find the minimum time to normalize timestamps (makes visualization easier)
min_time = df['start_time_ns'].min()
df['start_time_normalized'] = df['start_time_ns'] - min_time

# Group by label
grouped = df.groupby('label')

# Get unique labels and assign colors
labels = df['label'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
label_color_map = dict(zip(labels, colors))

label_map = {
    'BT': 'BLOCKTXN',
    'CIS': 'CheckInputScripts',
    'FS': 'FlushStateToDisk',
    'GH': 'GETHEADERS',
    'GD': 'GETDATA',
    'GBTXN': 'GETBLOCKTXN',
}

for label in labels:
    if label not in label_map:
        label_map[label] = label

total_time_range = df['start_time_normalized'].max() - df['start_time_normalized'].min()
time_density = len(df) / total_time_range  # Events per time unit

print(f"Total time range: {total_time_range} ns, Time density: {time_density:.8f} events/ns")

# Create figure and axis
fig, ax = plt.subplots(figsize=(20, 10))

# Calculate y positions for each label
y_positions = {label: i for i, label in enumerate(labels)}
y_ticks = list(y_positions.values())
y_labels = [label_map[label] for label in y_positions.keys()]

# Plot each label's events
for label, group in grouped:
    # Extract (start, duration) tuples for broken_barh
    bar_data = list(zip(group['start_time_normalized'], group['visible_duration']))

    mapped = label_map.get(label)
    if mapped in ['VERACK', 'VERSION', 'ADDRV2']:
        # Skip non-bandwidth-intensive messages
        continue

    # Plot the events for this label
    ax.broken_barh(
        bar_data,
        (y_positions[label] - 0.4, 0.8),  # (y-position, height)
        # facecolors=label_color_map[label],
        facecolors='black',
        alpha=0.8
    )

def getmeta(event) -> dict:
    metadata = {}
    for item in event['metadata'].split('|'):
        if item:
            k, v = item.split('=')
            metadata[k] = v
    return metadata

# Handle ConnectBlock events specially
abc_rows = df[df['label'] == 'ActivateBestChain']
if not abc_rows.empty:
    for _, event in abc_rows.iterrows():
        if 'new=' not in str(event['metadata']):
            # Not a new block
            continue
        metadata = getmeta(event)
        blockhash = metadata.get('new', 'unknown')[-12:]
        height = metadata.get('height', 'unknown')

        event_time = event['start_time_normalized']

        # Draw vertical line
        ax.axvline(x=event_time, ymin=0, ymax=2, color='red', linestyle='dashed', alpha=0.2)

        # Add block hash annotation
        ax.annotate(
            f'{height} (...{blockhash})',
            xy=(event_time, 0.2),
            xytext=(event_time, len(labels)),
            rotation=45,
            ha='right',
            fontsize=8,
        )

# Set the chart limits and labels
ax.set_ylim(-1, len(labels) + 1)  # Add space for ConnectBlock annotations
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels)

ax.set_xlabel('Time (relative)')

# Format nanoseconds as human readable time (h:m:s.ms)
def format_ns_to_human_time(ns, pos):
    # Convert nanoseconds to seconds
    seconds = ns / 1e9

    # Calculate hours, minutes, seconds, and milliseconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)

    # Format based on the size of the time
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m {secs:02d}s"
    elif secs > 0:
        return f"{secs}.{millisecs:03d}s"
    else:
        # For very small durations, show milliseconds or microseconds
        if millisecs > 0:
            return f"{millisecs}ms"
        else:
            microsecs = int((seconds % 0.001) * 1000000)
            return f"{microsecs}μs"


from matplotlib.ticker import FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(format_ns_to_human_time))

label_to_count = {label: 0 for label in labels}
label_to_total_ns = {label: 0 for label in labels}

legend_patches = []

for label in labels:
    withlabel = df[df['label'] == label]
    count = len(withlabel)

    # Events assumed to be non-overlapping
    total_ns = withlabel['duration'].sum()
    perc_time = (total_ns / total_time_range) * 100
    perc_str = f"{perc_time:.4f}%"

    legend_patches.append(
        patches.Patch(
            color=label_color_map[label],
            label=f"{label_map[label]} ({count}) ({perc_str} of time)",
        )
    )

# Convert back to real timestamp for title
def ns_to_time_str(ns):
    # Convert nanoseconds to seconds
    seconds = ns / 1e9
    return f"{seconds:.3f}s"

# Add chart information
ax.set_title(f'Node activity over {format_ns_to_human_time(total_time_range, None)}')

# Add note about minimum duration if applicable
small_events = df[df['duration'] < min_visible_duration]
if not small_events.empty:
    plt.figtext(
        0.8, 0.01,
        f"Note: {len(small_events)} events shorter than {format_ns_to_human_time(min_visible_duration, 0)} have been enlarged for visibility",
        ha="center", fontsize=8, style='italic')

# Add a legend for red lines
legend_patches.append(patches.Patch(color='red', label='New block'))

plt.legend(handles=legend_patches, loc='lower right')

# Add grid for better readability
ax.grid(True, axis='x', linestyle='--', alpha=0.3)

# Save the figure
plt.savefig('event_timeline.png', dpi=300)
print("wrote event_timeline.png")

# Show the figure
plt.show()

