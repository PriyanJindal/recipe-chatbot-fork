#!/usr/bin/env python3
"""Evaluate the LLM judge performance on the test set.

This script evaluates the finalized LLM judge on the test set to get
unbiased estimates of TPR and TNR for use with judgy.
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rich.console import Console
from dotenv import load_dotenv
import re
import phoenix as px
import os
import litellm
from phoenix.experiments import run_experiment
import requests
from collections import defaultdict
from sklearn.metrics import confusion_matrix

load_dotenv()

# Set up Phoenix tracing
from phoenix.otel import register
tracer_provider = register(project_name="recipe-agent", batch=True, auto_instrument=True)

console = Console()

def load_data_split(csv_path: str) -> pd.DataFrame:
    """Load a data split from CSV file."""
    df = pd.read_csv(csv_path)
    return df

def load_judge_prompt(prompt_path: str) -> str:
    """Load the judge prompt from file."""
    with open(prompt_path, 'r') as f:
        return f.read()

def generate_eval_prompt(input, metadata, base_prompt):
    """Generate evaluation prompt for a single example."""
    formatted_prompt = base_prompt.replace("{attributes.query}", str(input.get("attributes.query")))
    formatted_prompt = formatted_prompt.replace("{attributes.dietary_restriction}", str(metadata.get("attributes.dietary_restriction")))
    formatted_prompt = formatted_prompt.replace("{attributes.output}", str(metadata.get("attributes.output.value")))

    return formatted_prompt

def create_task_function(base_prompt):
    """Create a task function that uses the provided base prompt."""
    def task(input, metadata):
        eval_prompt = generate_eval_prompt(input, metadata, base_prompt)
        completion = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": eval_prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(completion.choices[0].message.content)
    
    return task

def eval_tp(metadata, output):
    """Evaluate true positive."""
    label = output.get("label")
    tp = (metadata["ground_truth_label"] == "PASS") & (label.lower() == "pass")
    return tp

def eval_tn(metadata, output):
    """Evaluate true negative."""
    label = output.get("label")
    tn = (metadata["ground_truth_label"] == "FAIL") & (label.lower() == "fail")
    return tn

def eval_fp(metadata, output):
    """Evaluate false positive."""
    label = output.get("label")
    fp = (metadata["ground_truth_label"] == "FAIL") & (label.lower() == "pass")
    return fp

def eval_fn(metadata, output):
    """Evaluate false negative."""
    label = output.get("label")
    fn = (metadata["ground_truth_label"] == "PASS") & (label.lower() == "fail")
    return fn

def accuracy(metadata, output):
    """Evaluate accuracy."""
    label = output.get("label")
    accuracy = (metadata["ground_truth_label"].lower() == label.lower())
    return accuracy

def evaluate_judge_on_test(judge_prompt: str, test_traces: pd.DataFrame) -> Tuple[float, float, pd.DataFrame]:
    """Evaluate the judge prompt on the test set using Phoenix experiments."""
    
    console.print(f"[yellow]Evaluating judge on {len(test_traces)} test traces with Phoenix experiments...")
    
    # Set up Phoenix client
    phoenix_client = px.Client()
    
    # Upload test dataset to Phoenix
    test_dataset = phoenix_client.upload_dataset(
        dataframe=test_traces,
        dataset_name="test_set",
        input_keys=["attributes.query"],
        output_keys=[],
        metadata_keys=["attributes.output.value", "ground_truth_label", "ground_truth_explanation", "attributes.dietary_restriction", "attributes.trace_num"],
    )
    
    # Create task function with the judge prompt
    task = create_task_function(judge_prompt)
    
    # Run the experiment
    experiment = run_experiment(
        dataset=test_dataset, 
        task=task, 
        evaluators=[eval_tp, eval_tn, eval_fp, eval_fn, accuracy], 
        concurrency=3
    )
    experiment_id = experiment.id
    
    # Get results via API
    base_url = "http://localhost:6006"
    url = f"{base_url}/v1/experiments/{experiment_id}/json"
    response = requests.get(url)
    results = response.json()
    
    # Process results to get metrics
    metrics_count = defaultdict(int)
    for entry in results:
        for ann in entry['annotations']:
            if ann['name'] in ('eval_tp', 'eval_tn', 'eval_fp', 'eval_fn') and ann['label'] == 'True':
                metrics_count[ann['name']] += 1
    
    # Extract counts
    TP = metrics_count['eval_tp']
    TN = metrics_count['eval_tn']
    FP = metrics_count['eval_fp']
    FN = metrics_count['eval_fn']
    
    # Compute metrics
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    balanced_acc = (TPR + TNR) / 2
    
    # Build predictions dataframe for analysis
    predictions_data = []
    for idx, entry in enumerate(results):
        # Extract prediction and ground truth
        prediction = entry.get('output', {})
        test_data = test_traces.iloc[idx]
        
        predictions_data.append({
            'ground_truth_label': test_data.get('ground_truth_label'),
            'llm_as_judge_label': prediction.get('label'),
            'explanation': prediction.get('explanation'),
            'attributes.query': test_data.get('attributes.query'),
            'attributes.dietary_restriction': test_data.get('attributes.dietary_restriction'),
            'attributes.output.value': test_data.get('attributes.output.value'),
        })
    
    predictions = pd.DataFrame(predictions_data)
    
    console.print(f"[green]Completed labeling of {len(predictions)} traces")
    
    console.print("[green]Completed LLM-as-Judge Evaluation, logged to Phoenix")
    
    return TPR, TNR, predictions


def save_results(tpr: float, tnr: float, predictions: pd.DataFrame, 
                results_dir: Path) -> None:
    """Save evaluation results."""
    
    # Save performance metrics
    performance = {
        "test_set_performance": {
            "true_positive_rate": float(tpr),
            "true_negative_rate": float(tnr),
            "balanced_accuracy": float((tpr + tnr) / 2),
            "total_predictions": int(len(predictions)),
            "correct_predictions": int((predictions["ground_truth_label"] == predictions["llm_as_judge_label"]).sum()),
            "accuracy": float((predictions["ground_truth_label"] == predictions["llm_as_judge_label"]).mean())
        }
    }
    
    performance_path = results_dir / "judge_performance.json"
    with open(performance_path, 'w') as f:
        json.dump(performance, f, indent=2)
    console.print(f"[green]Saved performance metrics to {performance_path}")
    
    # Save detailed predictions
    predictions_path = results_dir / "test_predictions.json"
    predictions.to_json(predictions_path)
    console.print(f"[green]Saved test predictions to {predictions_path}")
    
    # Save predictions in format for judgy
    test_labels = [1 if label == "PASS" else 0 for label in predictions["ground_truth_label"]]
    test_preds = [1 if label == "PASS" else 0 for label in predictions["llm_as_judge_label"]]
    
    judgy_data = {
        "test_labels": test_labels,
        "test_preds": test_preds,
        "description": "Test set labels and predictions for judgy evaluation"
    }
    
    judgy_path = results_dir / "judgy_test_data.json"
    with open(judgy_path, 'w') as f:
        json.dump(judgy_data, f, indent=2)
    console.print(f"[green]Saved judgy test data to {judgy_path}")

def main():
    """Main function to evaluate the judge on test set."""
    console.print("[bold blue]LLM Judge Test Set Evaluation")
    console.print("=" * 50)
    
    # Set up paths
    script_dir = Path(__file__).parent
    hw3_dir = script_dir.parent
    data_dir = hw3_dir / "data"
    results_dir = hw3_dir / "results"
    
    # Load test set
    test_path = data_dir / "test_set.csv"
    if not test_path.exists():
        console.print("[red]Error: Test set not found!")
        console.print("[yellow]Please run split_data.py first.")
        return
    
    test_traces = load_data_split(str(test_path))
    console.print(f"[green]Loaded {len(test_traces)} test traces")
    
    # Load judge prompt
    prompt_path = results_dir / "judge_prompt.txt"
    if not prompt_path.exists():
        console.print("[red]Error: Judge prompt not found!")
        console.print("[yellow]Please run develop_judge.py first.")
        return
    
    judge_prompt = load_judge_prompt(str(prompt_path))
    console.print("[green]Loaded judge prompt")
    
    # Evaluate judge on test set
    console.print("[yellow]Evaluating judge on test set... This may take a while.")
    tpr, tnr, predictions = evaluate_judge_on_test(judge_prompt, test_traces)
    
    # Print results
    console.print(f"\n[bold]Judge Performance on Test Set:")
    console.print(f"True Positive Rate (TPR): {tpr:.3f}")
    console.print(f"True Negative Rate (TNR): {tnr:.3f}")
    console.print(f"Balanced Accuracy: {(tpr + tnr) / 2:.3f}")
    console.print(f"Overall Accuracy: {(predictions['ground_truth_label'] == predictions['llm_as_judge_label']).mean():.3f}")
    
    # Save results
    save_results(tpr, tnr, predictions, results_dir)
    
    console.print("\n[bold green]Test set evaluation completed!")
    console.print("[blue]Results saved for use with judgy in the final evaluation step.")

if __name__ == "__main__":
    main() 