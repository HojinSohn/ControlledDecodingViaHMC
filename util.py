import datetime
import pandas as pd
import matplotlib.pyplot as plt
import csv

def write_rejected_text(file, rejected_text, nll, sentiment_score, PE, KE, lambda_energy, alpha_val):
    """
    Write rejected text with its NLL and sentiment score to a file.
    
    Args:
        file: Open file object to write to
        rejected_text: String of the rejected sequence
        nll: Negative log-likelihood (float)
        sentiment_score: Sentiment score (float)
    """
    # file.write(f"{rejected_text},{nll:.4f},{sentiment_score:.4f},{PE:.4f},{KE:.4f},{lambda_energy:.4f},0\n")
    # file.flush()  # Ensure immediate write to disk

    writer = csv.writer(file, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    # Write the data with rejected_text properly quoted
    writer.writerow([rejected_text, nll, sentiment_score, PE, KE, lambda_energy, alpha_val, 0])
    file.flush()  # Ensure immediate write to disk

def write_accepted_text(file, accepted_text, nll, sentiment_score, PE, KE, lambda_energy, alpha_val):
    """
    Write accepted text with its NLL and sentiment score to a file.
    
    Args:
        file: Open file object to write to
        accepted_text: String of the accepted sequence
        nll: Negative log-likelihood (float)
        sentiment_score: Sentiment score (float)
    """
    # file.write(f"{accepted_text},{nll:.4f},{sentiment_score:.4f},{PE:.4f},{KE:.4f},{lambda_energy:.4f},1\n")
    # file.flush()  # Ensure immediate write to disk

    writer = csv.writer(file, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    # Write the data with rejected_text properly quoted
    writer.writerow([accepted_text, nll, sentiment_score, PE, KE, lambda_energy, alpha_val, 1])
    file.flush()  # Ensure immediate write to disk

def write_file_header(file):
    file.write("Text,NLL,Sentiment_Score,Potential_Energy,Kinetic_Energy,Lambda,Acceptance_rate,Accepted\n")

def get_file_name(lambda_energy, epsilon, alpha, n_steps, std_dev, delta, num_leapfrog):
    """
    Generates a sampling log filename based on hyperparameters.

    Args:
        lambda_energy (float)
        epsilon (float)
        alpha (float)
        n_steps (int)
        std_dev (float)
        delta (float)
        num_leapfrog (int)

    Returns:
        str: Filename string
    """
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"log_data/sampling_log_LE{lambda_energy}_eps{epsilon}_a{alpha}_ns{n_steps}_"
        f"std{std_dev}_d{delta}_lf{num_leapfrog}_{timestamp}.csv"
    )
    return filename

def plot_energy_movement(file_name):
    # Load CSV
    df = pd.read_csv(file_name)

    # Compute Total Energy
    df['Total_Energy'] = df['Potential_Energy'] + df['Kinetic_Energy']

    # Plot
    plt.figure(figsize=(10, 5))
    
    # Plot each energy type
    plt.plot(df.index, df["Potential_Energy"], marker='o', label='Potential Energy')
    plt.plot(df.index, df["Kinetic_Energy"], marker='x', label='Kinetic Energy')
    plt.plot(df.index, df['Total_Energy'], marker='s', label='Total Energy')

    # Labeling
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.title("Energy over Sampling Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot with the same name as input CSV
    output_file = file_name.replace(".csv", "_energy_plot.png")
    plt.savefig(output_file, dpi=300)