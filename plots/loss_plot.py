import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import re

def extract_float(tensor_str):
    if isinstance(tensor_str, str):
        match = re.search(r'tensor\(\[([0-9eE\.\-]+)', tensor_str)
        if match:
            return float(match.group(1))
    try:
        return float(tensor_str)
    except:
        return float('nan')

def plot_fluency_sentiment(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    required_cols = ['Sentiment_Score', 'NLL']
    if not all(col in df.columns for col in required_cols):
        print(f"CSV must contain columns: {required_cols}")
        return

    # Clean tensor string columns
    for col in ['NLL', 'Sentiment_Score', 'Acceptance_rate']:
        if col in df.columns:
            df[col] = df[col].apply(extract_float)

    fluency = df['NLL']
    sentiment = df['Sentiment_Score'] * 100

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, sentiment, label='Sentiment Loss (upscaled 100)', color='green', linestyle='-', marker='o')
    plt.plot(df.index, fluency, label='Fluency Loss (NLL)', color='blue', linestyle='--', marker='x')

    plt.xlabel('Example Index')
    plt.ylabel('Score')
    plt.title('Sentiment Score and Fluency over Examples')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename_wo_ext = os.path.splitext(os.path.basename(csv_file))[0]
    output_file = f"{filename_wo_ext}_sentiment_fluency_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved as: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_fluency_sentiment.py <csv_file>")
    else:
        plot_fluency_sentiment(sys.argv[1])
