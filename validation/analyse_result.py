import pandas as pd
import ast

# Load the CSV
df = pd.read_csv('efficiency_report.csv')


def parse_list(s):
    try:
        return ast.literal_eval(s)
    except:
        return []


# Apply parsing
df['Detected_List'] = df['Detected'].apply(parse_list)
df['Expected_List'] = df['Expected'].apply(parse_list)

# Group by Folder to find the most common wrong detection
print(f"{'Folder':<8} | {'Expected':<35} | {'Most Common Detection (Count)':<40}")
print("-" * 90)

for folder in sorted(df['Folder'].unique()):
    subset = df[df['Folder'] == folder]
    expected = subset.iloc[0]['Expected']

    # Count most common detected patterns
    detected_patterns = subset['Detected'].value_counts().head(1)
    if not detected_patterns.empty:
        common_det = detected_patterns.index[0]
        count = detected_patterns.iloc[0]
        print(f"{folder:<8} | {str(expected):<35} | {str(common_det):<30} ({count})")
