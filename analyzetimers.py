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
df = pd.read_csv('event_log5.csv', usecols=['start_time_ns', 'duration', 'label', 'metadata'])


# Format nanoseconds as human readable time (h:m:s.ms)
def format_ns(ns):
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


# Convert start_time_ns and duration to numeric if they're not already
df['start_time_ns'] = pd.to_numeric(df['start_time_ns'])
df['duration'] = pd.to_numeric(df['duration'])

# Group by label
grouped = df.groupby('label')

# Get unique labels and assign colors
labels = df['label'].unique()

label_map = {
    'BT': 'BLOCKTXN',
    'CIS': 'CheckInputScripts',
    'FS': 'FlushStateToDisk',
    'GH': 'GETHEADERS',
    'GBTXN': 'GETBLOCKTXN',
}

for label in labels:
    if label not in label_map:
        label_map[label] = label

total_time_range = df['start_time_ns'].max() - df['start_time_ns'].min()
time_density = len(df) / total_time_range  # Events per time unit

print(f"Total time: {format_ns(total_time_range)}")

def getmeta(event) -> dict:
    metadata = {}
    for item in event['metadata'].split('|'):
        if item:
            k, v = item.split('=')
            metadata[k] = v
    return metadata


print(f"\n{'event type':<20} {'count':<20} {'total time':<20}\n")

label_to_lines = {}
total_time = 0.
total_perc_time = 0.

for label in labels:
    withlabel = df[df['label'] == label]
    count = len(withlabel)

    # Events assumed to be non-overlapping
    total_ns = withlabel['duration'].sum()
    perc_time = (total_ns / total_time_range) * 100
    c2 = f"{format_ns(total_ns)} total"
    c3 = f"{perc_time:.4f}% of total"

    total_time += total_ns
    total_perc_time += perc_time

    label_to_lines[label] = (
        total_ns, f"{label_map[label]:<20} {count:<20} {c2:<20} {c3:<20}")


for _, (_, line) in sorted(label_to_lines.items(), key=lambda v: -v[1][0]):
    print(line)

print()
print("-" * 50)
print()
print(f"Total time spent working: {format_ns(total_time)}")
print(f"Total time spent working (percent): {total_perc_time:.4f}%")

print()
print(f"Scaled linearly 100x: {total_perc_time * 100:.1f}%")
print(f"Scaled linearly 1000x: {total_perc_time * 1000:.1f}%")
