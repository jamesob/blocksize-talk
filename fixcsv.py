import csv
import pandas as pd

def fix_csv_quoting(input_file, output_file):
    # Read the file line by line
    with open(input_file, 'r', newline='') as infile:
        lines = infile.readlines()

    # Process header separately
    header = lines[0].strip()

    # Process each data line
    fixed_lines = [header]
    for line in lines[1:]:
        # Split only by the first three commas
        parts = line.strip().split(',', 3)

        # If we have at least 4 parts (the expected columns)
        if len(parts) >= 4:
            # Properly quote the fourth column (metadata)
            if not (parts[3].startswith('"') and parts[3].endswith('"')):
                # Replace any existing double quotes with escaped double quotes
                parts[3] = parts[3].replace('"', '""')
                # Wrap in double quotes
                parts[3] = f'"{parts[3]}"'

            # Reassemble the line
            fixed_line = f"{parts[0]},{parts[1]},{parts[2]},{parts[3]}"
            fixed_lines.append(fixed_line)
        else:
            print(f"Skipping malformed line: {line.strip()}")

    # Write the fixed data
    with open(output_file, 'w', newline='') as outfile:
        for line in fixed_lines:
            outfile.write(line + '\n')

    print(f"Fixed CSV written to {output_file}")
    print(f"Processed {len(lines)} lines, wrote {len(fixed_lines)} lines")

if __name__ == "__main__":
    input_file = input("Enter the path to your CSV file: ")
    output_file = input("Enter the path for the fixed CSV file: ")
    fix_csv_quoting(input_file, output_file)
