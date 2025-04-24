import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_lambda(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if 'Lambda' not in df.columns:
        print("The file does not contain a 'Lambda' column.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Lambda'], marker='o', linestyle='-', color='b', label='Lambda')
    plt.xlabel('Example Index')
    plt.ylabel('Lambda')
    plt.title('Lambda over Examples')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to file
    filename_wo_ext = os.path.splitext(os.path.basename(csv_file))[0]
    output_file = f"{filename_wo_ext}_lambda_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved as: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_lambda.py <csv_file>")
    else:
        plot_lambda(sys.argv[1])
