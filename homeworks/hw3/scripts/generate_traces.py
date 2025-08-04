#!/usr/bin/env python3
"""Generate Recipe Bot traces for dietary adherence evaluation.

This script sends dietary preference queries to the Recipe Bot and collects
the responses to create a dataset for LLM-as-Judge evaluation.
"""

import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up Phoenix tracing
from phoenix.otel import register
tracer_provider = register(project_name="recipe-agent", batch=True, auto_instrument=True)
tracer = tracer_provider.get_tracer(__name__)

# Add the backend to the path so we can import the Recipe Bot
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.utils import get_agent_response

MAX_WORKERS = 32

console = Console()

def load_dietary_queries(csv_path: str) -> List[Dict[str, Any]]:
    """Load dietary preference queries from CSV file."""
    df = pd.read_csv(csv_path)
    return df.to_dict('records')
        

def generate_trace_with_id(args: tuple):
    """Wrapper function for parallel processing."""
    with tracer.start_as_current_span(
        "Query_Information", openinference_span_kind="chain",
    ) as span:
        try:
            query_data, trace_num = args
            query = query_data["query"]
            dietary_restriction = query_data["dietary_restriction"]
            
            span.set_input(query)
            span.set_attribute("query", query)
            span.set_attribute("id", query_data["id"])
            span.set_attribute("trace_num", trace_num)
            span.set_attribute("dietary_restriction", dietary_restriction)
            span.set_attribute("success", True)
            
            messages = [{"role": "user", "content": query}]
            response = get_agent_response(messages)
            output = response[-1]["content"]
            span.set_output(output)
            return span
        
        except Exception as e:
            console.print(f"[red]Error generating trace for query: {query}")
            console.print(f"[red]Error: {str(e)}")
            span.set_attribute("error", str(e))
            span.set_attribute("success", False)
            return span

def generate_multiple_traces_per_query(queries: List[Dict[str, Any]], 
                                     traces_per_query: int = 40,
                                     max_workers: int = MAX_WORKERS) -> List[Dict[str, Any]]:
    """Generate multiple traces for each query using parallel processing."""
    
    # Create all the tasks
    tasks = []
    for query_data in queries:
        for i in range(traces_per_query):
            tasks.append((query_data, i + 1))
    
    all_traces = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and collect results
        futures = [executor.submit(generate_trace_with_id, task) for task in tasks]
        
        # Collect all results
        for future in as_completed(futures):
            trace = future.result()
            all_traces.append(trace)
    
    console.print(f"[green]Completed parallel generation of {len(all_traces)} traces")
    return all_traces

def main():
    """Main function to generate Recipe Bot traces."""
    console.print("[bold blue]Recipe Bot Trace Generation")
    console.print("=" * 50)
    
    # Set up paths
    script_dir = Path(__file__).parent
    hw3_dir = script_dir.parent
    data_dir = hw3_dir / "data"
    
    # Load dietary queries
    queries_path = data_dir / "dietary_queries.csv"
    if not queries_path.exists():
        console.print(f"[red]Error: {queries_path} not found!")
        return
    
    queries = load_dietary_queries(str(queries_path))
    console.print(f"[green]Loaded {len(queries)} dietary queries")

    
    # Generate traces (40 traces per query)
    console.print("[yellow]Generating traces... This may take a while as we are making many LLM calls.")
    traces = generate_multiple_traces_per_query(queries, traces_per_query=40)
    
    # Filter successful traces
    successful_traces = [t for t in traces if t.attributes["success"]]
    failed_traces = [t for t in traces if not t.attributes["success"]]
    
    console.print(f"[green]Successfully generated {len(successful_traces)} traces")
    if failed_traces:
        console.print(f"[yellow]Failed to generate {len(failed_traces)} traces")
    
    # Print summary statistics
    console.print("\n[bold]Summary Statistics:")
    console.print(f"Total traces generated: {len(successful_traces)}")
    
    # Count by dietary restriction
    restriction_counts = {}
    for trace in successful_traces:
        restriction = trace.attributes["dietary_restriction"]
        restriction_counts[restriction] = restriction_counts.get(restriction, 0) + 1
    
    console.print("\nTraces per dietary restriction:")
    for restriction, count in sorted(restriction_counts.items()):
        console.print(f"  {restriction}: {count}")

if __name__ == "__main__":
    main() 