import pandas as pd
import argparse
from Evaluators.Evaluators import PerplexityEvaluator, SentimentEvaluator
import os
import torch
import torch.nn.functional as F

def options():
    parser = argparse.ArgumentParser(description="Evaluate accepted texts using perplexity and sentiment.")
    parser.add_argument("--csv_file", required=True, help="CSV file name (in 'good_data/' directory)")
    parser.add_argument("--output_file", default="best_sample.csv", help="Output CSV file to save results")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()

def main():
    args = options()
    df = pd.read_csv(os.path.join("good_data", args.csv_file))
    accepted_texts = df[df["Accepted"] == 1]["Text"].tolist()

    if not accepted_texts:
        print("No accepted samples found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    p_eval = PerplexityEvaluator(device=device, debug=False)
    s_eval = SentimentEvaluator("textattack/roberta-base-SST-2", device=device, debug=False)

    results = []

    for text in accepted_texts:
        perplexity = p_eval.compute_perplexity(text)
        _, sentiment_probs = s_eval.predict_sentiment(text)
        sentiment_score = sentiment_probs[1]  # positive class prob

        if args.debug:
            print(f"Sentiment Score: {sentiment_score:.3f}, Perplexity: {perplexity:.2f}")

        results.append({
            "Text": text,
            "Sentiment Score": sentiment_score,
            "Perplexity": perplexity
        })

    output_path = os.path.join("evaluation_result", args.output_file)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
