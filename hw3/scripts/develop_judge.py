#!/usr/bin/env python3
"""Develop and refine the LLM judge prompt for dietary adherence evaluation.

This script creates an LLM judge prompt with carefully selected few-shot examples
using embedding similarity and iteratively refines it on the dev set.
"""

import pandas as pd
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rich.console import Console
from dotenv import load_dotenv
from phoenix.evals import llm_generate, OpenAIModel
import re
import phoenix as px
import os
import litellm
from phoenix.experiments import run_experiment
import requests
from phoenix.client import Client
from collections import defaultdict
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

load_dotenv()

# Set up Phoenix tracing
from phoenix.otel import register
tracer_provider = register(project_name="recipe-agent", batch=True, auto_instrument=True)

console = Console()

def load_data_split(csv_path: str) -> pd.DataFrame:
    """Load a data split from CSV file."""
    df = pd.read_csv(csv_path)
    return df

def select_few_shot_examples(train_df: pd.DataFrame, 
                           num_positive: int = 1, 
                           num_negative: int = 3) -> List[Dict[str, Any]]:
    """Select few-shot examples randomly from train set."""
    
    console.print("[yellow]Selecting random few-shot examples...")
    
    # Separate by label
    train_pass = [trace for _, trace in train_df.iterrows() if trace["ground_truth_label"] == "PASS"]
    train_fail = [trace for _, trace in train_df.iterrows() if trace["ground_truth_label"] == "FAIL"]    

    selected_examples = []

    # Select positive examples (PASS) randomly
    if train_pass and len(train_pass) >= num_positive:
        selected_examples.extend(random.sample(train_pass, num_positive))
    elif train_pass:
        selected_examples.extend(train_pass)  # Use all available if less than requested

    # Select negative examples (FAIL) randomly
    if train_fail and len(train_fail) >= num_negative:
        selected_examples.extend(random.sample(train_fail, num_negative))
    elif train_fail:
        selected_examples.extend(train_fail)  # Use all available if less than requested

    console.print(f"[green]Selected {len(selected_examples)} few-shot examples ({len([e for e in selected_examples if e['ground_truth_label'] == 'PASS'])} PASS, {len([e for e in selected_examples if e['ground_truth_label'] == 'FAIL'])} FAIL)")
    return selected_examples

def create_judge_prompt(few_shot_examples: List[Dict[str, Any]]) -> str:
    """Create the LLM judge prompt with few-shot examples."""
    
    # Base prompt
    base_prompt = """You are an expert nutritionist and dietary specialist evaluating whether recipe responses properly adhere to specified dietary restrictions.

DIETARY RESTRICTION DEFINITIONS:
- Vegan: No animal products (meat, dairy, eggs, honey, etc.)
- Vegetarian: No meat or fish, but dairy and eggs are allowed
- Gluten-free: No wheat, barley, rye, or other gluten-containing grains
- Dairy-free: No milk, cheese, butter, yogurt, or other dairy products
- Keto: Very low carb (typically <20g net carbs), high fat, moderate protein
- Paleo: No grains, legumes, dairy, refined sugar, or processed foods
- Pescatarian: No meat except fish and seafood
- Kosher: Follows Jewish dietary laws (no pork, shellfish, mixing meat/dairy)
- Halal: Follows Islamic dietary laws (no pork, alcohol, proper slaughter)
- Nut-free: No tree nuts or peanuts
- Low-carb: Significantly reduced carbohydrates (typically <50g per day)
- Sugar-free: No added sugars or high-sugar ingredients
- Raw vegan: Vegan foods not heated above 118°F (48°C)
- Whole30: No grains, dairy, legumes, sugar, alcohol, or processed foods
- Diabetic-friendly: Low glycemic index, controlled carbohydrates
- Low-sodium: Reduced sodium content for heart health

EVALUATION CRITERIA:
- PASS: The recipe clearly adheres to the dietary preferences with appropriate ingredients and preparation methods
- FAIL: The recipe contains ingredients or methods that violate the dietary preferences
- Consider both explicit ingredients and cooking methods

Here are some examples of how to evaluate dietary adherence:

"""
    
    # Add few-shot examples
    for i, example in enumerate(few_shot_examples, 1):
        base_prompt += f"\nExample {i}:\n"
        base_prompt += f"Query and Response: {example['attributes.output.value']}\n"
        base_prompt += f"Explanation: {example['ground_truth_explanation']}\n"
        base_prompt += f"Label: {example['ground_truth_label']}\n"
    
    # Add evaluation template - using placeholders that won't conflict with JSON
    base_prompt += """

Now evaluate the following recipe response:

Query: {attributes.query}
Dietary Restriction: {attributes.dietary_restriction}
Recipe Response: {attributes.output.value}

MAKE SURE TO RETURN YOUR EVALUATION IN THE FOLLOWING JSON FORMAT:
"label": "PASS" or "FAIL",
"explanation": "Detailed explanation of your evaluation, citing specific ingredients or methods"


"""
    
    return base_prompt

def generate_eval_prompt(input, metadata, base_prompt):
    formatted_prompt = base_prompt.replace("{attributes.query}", str(input.get("attributes.query")))
    formatted_prompt = formatted_prompt.replace("{attributes.dietary_restriction}", str(metadata.get("attributes.dietary_restriction")))
    formatted_prompt = formatted_prompt.replace("{attributes.output}", str(metadata.get("attributes.output.value")))
    return formatted_prompt

def create_task_function(base_prompt):
    """Create a task function that uses the provided base prompt."""
    def task(input, metadata):
        eval_prompt = generate_eval_prompt(input, metadata, base_prompt)
        completion = litellm.completion(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": eval_prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(completion.choices[0].message.content)
    
    return task

def eval_tp(metadata, output):
    label = output.get("label")
    tp = (metadata["ground_truth_label"] == "PASS") & (label.lower() == "pass")
    return tp

def eval_tn(metadata, output):
    label = output.get("label")
    tn = (metadata["ground_truth_label"] == "FAIL") & (label.lower() == "fail")
    return tn

def eval_fp(metadata, output):
    label = output.get("label")
    fp = (metadata["ground_truth_label"] == "FAIL") & (label.lower() == "pass")
    return fp

def eval_fn(metadata, output):
    label = output.get("label")
    fn = (metadata["ground_truth_label"] == "PASS") & (label.lower() == "fail")
    return fn

def accuracy(metadata, output):
    label = output.get("label")
    accuracy = (metadata["ground_truth_label"].lower() == label.lower())
    return accuracy

def save_judge_prompt(prompt: str, output_path: str) -> None:
    """Save the judge prompt to a text file."""
    with open(output_path, 'w') as f:
        f.write(prompt)
    console.print(f"[green]Saved judge prompt to {output_path}")

def main():
    """Main function to develop the LLM judge."""

    # Set up Phoenix client
    phoenix_client = px.Client()

    # Load data splits
    script_dir = Path(__file__).parent
    hw3_dir = script_dir.parent
    data_dir = hw3_dir / "data"

    train_path = data_dir / "train_set.csv"
    dev_path = data_dir / "dev_set.csv"
    test_path = data_dir / "test_set.csv"

    train_df = load_data_split(str(train_path))
    dev_df = load_data_split(str(dev_path))

    console.print("[bold blue]Loading data splits...")
    console.print("=" * 50)

    # Upload the data splits to Phoenix
    train_dataset = phoenix_client.upload_dataset(
        dataframe=train_df,
        dataset_name="train_set",
        input_keys=["attributes.query"],
        output_keys=[],
        metadata_keys=["attributes.output.value", "ground_truth_label", "ground_truth_explanation", "attributes.dietary_restriction", "attributes.trace_num"],
    )
    
    dev_dataset = phoenix_client.upload_dataset(
        dataframe=dev_df,
        dataset_name="dev_set",
        input_keys=["attributes.query"],
        output_keys=[],
        metadata_keys=["attributes.output.value", "ground_truth_label", "ground_truth_explanation", "attributes.dietary_restriction", "attributes.trace_num"],
    )

    console.print("[bold blue]LLM Judge Development")
    console.print("=" * 50)

    # Select few-shot examples randomly
    few_shot_examples = select_few_shot_examples(train_df)
    
    if not few_shot_examples:
        console.print("[red]Failed to select few-shot examples!")
        return
    
    # Create judge prompt
    judge_prompt = create_judge_prompt(few_shot_examples)

    print("judge_prompt", judge_prompt)

    # Create task function with the judge prompt
    task = create_task_function(judge_prompt)

    # Evaluate judge on dev set
    console.print("[yellow]Evaluating judge on dev set...")
    experiment = run_experiment(dataset=dev_dataset, task=task, evaluators=[eval_tp, eval_tn, eval_fp, eval_fn, accuracy], concurrency=3)
    experiment_id = experiment.id

    base_url = "http://localhost:6006"
    url = f"{base_url}/v1/experiments/{experiment_id}/json"
    response = requests.get(url)
    results = response.json()

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

    # Confusion matrix
    y_true = []
    y_pred = []
    for entry in results:
        # We'll treat TP/FN as positives and TN/FP as negatives
        if any(ann['name'] == 'eval_fn' and ann['label'] == 'True' for ann in entry['annotations']):
            y_true.append(1)  # Positive case
            y_pred.append(0)  # Predicted negative
        elif any(ann['name'] == 'eval_tp' and ann['label'] == 'True' for ann in entry['annotations']):
            y_true.append(1)
            y_pred.append(1)
        elif any(ann['name'] == 'eval_tn' and ann['label'] == 'True' for ann in entry['annotations']):
            y_true.append(0)
            y_pred.append(0)
        elif any(ann['name'] == 'eval_fp' and ann['label'] == 'True' for ann in entry['annotations']):
            y_true.append(0)
            y_pred.append(1)

    conf_matrix = confusion_matrix(y_true, y_pred)

    console.print(f"\n[bold]Judge Performance on Dev Set:")
    console.print(f"True Positive Rate (TPR): {TPR:.3f}")
    console.print(f"True Negative Rate (TNR): {TNR:.3f}")
    console.print(f"Balanced Accuracy: {balanced_acc:.3f}")
    console.print(f"Confusion Matrix: {conf_matrix}")
    
    # # Save judge prompt
    prompt_path = hw3_dir / "results" / "judge_prompt.txt"
    save_judge_prompt(judge_prompt, str(prompt_path))
    
    console.print("\n[bold green]Judge development completed!")
    console.print(f"[blue]Judge prompt saved to: {prompt_path}")

if __name__ == "__main__":
    main() 