#!/usr/bin/env python3
"""
Evaluation Utilities for Recipe Retrieval Systems

Provides reusable evaluation classes and functions for measuring retrieval performance
with standard IR metrics like Recall@k and MRR.
"""

import json
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
from tqdm import tqdm


class BaseRetrievalEvaluator:
    """Base evaluator for retrieval performance using standard IR metrics."""
    
    def __init__(self, retriever, query_processor: Optional[Callable] = None):
        """
        Initialize evaluator.
        
        Args:
            retriever: Object with retrieve_bm25(query, top_k) method
            query_processor: Optional function to process queries before retrieval
        """
        self.retriever = retriever
        self.query_processor = query_processor
        self.results = []
    
    def calculate_recall_at_k(self, retrieved_ids: List[int], target_id: int, k: int) -> float:
        """Calculate recall@k for a single query."""
        if target_id in retrieved_ids[:k]:
            return 1.0
        return 0.0
    
    def calculate_reciprocal_rank(self, retrieved_ids: List[int], target_id: int) -> float:
        """Calculate reciprocal rank for a single query."""
        try:
            rank = retrieved_ids.index(target_id) + 1  # 1-indexed
            return 1.0 / rank
        except ValueError:
            return 0.0  # Target not found
    
    def evaluate_single_query(self, query_data: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """Evaluate retrieval for a single query."""
        original_query = query_data['query']
        target_recipe_id = query_data['source_recipe_id']
        
        # Apply query processing if provided
        if self.query_processor:
            processed_result = self.query_processor(original_query)
            search_query = processed_result.get('processed_query', original_query)
            processing_strategy = processed_result.get('strategy', 'none')
        else:
            search_query = original_query
            processing_strategy = 'none'
        
        # Retrieve top-k results (use max of top_k and 10 for recall@10)
        max_k = max(top_k, 10)
        results = self.retriever.retrieve_bm25(search_query, top_k=max_k)
        retrieved_ids = [recipe['id'] for recipe in results]
        
        # Calculate metrics
        recall_1 = self.calculate_recall_at_k(retrieved_ids, target_recipe_id, 1)
        recall_3 = self.calculate_recall_at_k(retrieved_ids, target_recipe_id, 3)
        recall_5 = self.calculate_recall_at_k(retrieved_ids, target_recipe_id, 5)
        recall_10 = self.calculate_recall_at_k(retrieved_ids, target_recipe_id, 10)
        reciprocal_rank = self.calculate_reciprocal_rank(retrieved_ids, target_recipe_id)
        
        # Find actual rank of target recipe
        target_rank = None
        if target_recipe_id in retrieved_ids:
            target_rank = retrieved_ids.index(target_recipe_id) + 1
        
        evaluation_result = {
            "original_query": original_query,
            "search_query": search_query,
            "processing_strategy": processing_strategy,
            "target_recipe_id": target_recipe_id,
            "target_recipe_name": query_data['source_recipe_name'],
            "salient_fact": query_data['salient_fact'],
            "retrieved_ids": retrieved_ids[:top_k],  # Only save the requested top_k
            "retrieved_names": [recipe['name'] for recipe in results[:top_k]],
            "target_rank": target_rank,
            "recall_1": recall_1,
            "recall_3": recall_3,
            "recall_5": recall_5,
            "recall_10": recall_10,
            "reciprocal_rank": reciprocal_rank,
            "bm25_scores": [recipe.get('bm25_score', 0.0) for recipe in results[:top_k]]
        }
        
        return evaluation_result
    
    def evaluate_all_queries(self, queries: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Evaluate retrieval for all queries."""
        results = []
        for query_data in tqdm(queries, desc="Evaluating queries"):
            result = self.evaluate_single_query(query_data, top_k)
            results.append(result)
        
        self.results = results
        return results
    
    def calculate_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate metrics across all queries."""
        if not results:
            return {}
        
        recall_1_scores = [r['recall_1'] for r in results]
        recall_3_scores = [r['recall_3'] for r in results]
        recall_5_scores = [r['recall_5'] for r in results]
        recall_10_scores = [r['recall_10'] for r in results]
        reciprocal_ranks = [r['reciprocal_rank'] for r in results]
        
        # Calculate found ranks for additional analysis
        found_ranks = [r['target_rank'] for r in results if r['target_rank'] is not None]
        
        metrics = {
            "recall_at_1": statistics.mean(recall_1_scores),
            "recall_at_3": statistics.mean(recall_3_scores),
            "recall_at_5": statistics.mean(recall_5_scores),
            "recall_at_10": statistics.mean(recall_10_scores),
            "mean_reciprocal_rank": statistics.mean(reciprocal_ranks),
            "total_queries": len(results),
            "queries_found": len(found_ranks),
            "queries_not_found": len(results) - len(found_ranks),
            "average_rank_when_found": statistics.mean(found_ranks) if found_ranks else None,
            "median_rank_when_found": statistics.median(found_ranks) if found_ranks else None
        }
        
        return metrics
    
    def print_detailed_results(self, results: List[Dict[str, Any]], show_failures: bool = True, max_examples: int = 10):
        """Print detailed results for analysis."""
        metrics = self.calculate_aggregate_metrics(results)
        
        print(f"\n--- Retrieval Evaluation Results ---")
        print(f"Total queries evaluated: {metrics['total_queries']}")
        print(f"Recall@1: {metrics['recall_at_1']:.3f}")
        print(f"Recall@3: {metrics['recall_at_3']:.3f}")
        print(f"Recall@5: {metrics['recall_at_5']:.3f}")
        print(f"Recall@10: {metrics['recall_at_10']:.3f}")
        print(f"Mean Reciprocal Rank (MRR): {metrics['mean_reciprocal_rank']:.3f}")
        print(f"Queries where target was found: {metrics['queries_found']}/{metrics['total_queries']}")
        
        if metrics['average_rank_when_found']:
            print(f"Average rank when found: {metrics['average_rank_when_found']:.2f}")
            print(f"Median rank when found: {metrics['median_rank_when_found']:.1f}")
        
        # Show successful examples
        successful = [r for r in results if r['recall_1'] == 1.0]
        if successful:
            print(f"\n--- Successful Retrievals (Recall@1) [{min(len(successful), max_examples)}] ---")
            for i, result in enumerate(successful[:max_examples]):
                query_info = f"Original: '{result['original_query']}'"
                if result['search_query'] != result['original_query']:
                    query_info += f"\n   Search: '{result['search_query']}' ({result['processing_strategy']})"
                
                print(f"\n{i+1}. {query_info}")
                print(f"   Target: {result['target_recipe_name']}")
                print(f"   Retrieved #1: {result['retrieved_names'][0] if result['retrieved_names'] else 'None'}")
                print(f"   Salient Fact: {result['salient_fact']}")
        
        # Show failure examples
        if show_failures:
            failed = [r for r in results if r['recall_5'] == 0.0]
            if failed:
                print(f"\n--- Failed Retrievals (Not in Top 5) [{min(len(failed), max_examples)}] ---")
                for i, result in enumerate(failed[:max_examples]):
                    query_info = f"Original: '{result['original_query']}'"
                    if result['search_query'] != result['original_query']:
                        query_info += f"\n   Search: '{result['search_query']}' ({result['processing_strategy']})"
                    
                    print(f"\n{i+1}. {query_info}")
                    print(f"   Target: {result['target_recipe_name']}")
                    print(f"   Salient Fact: {result['salient_fact']}")
                    print(f"   Top Retrieved:")
                    for j, name in enumerate(result['retrieved_names'][:3]):
                        print(f"     {j+1}. {name}")
    
    def save_results(self, results: List[Dict[str, Any]], output_path: Path, experiment_name: str = "baseline") -> None:
        """Save detailed results to JSON file."""
        metrics = self.calculate_aggregate_metrics(results)
        
        output_data = {
            "experiment_name": experiment_name,
            "aggregate_metrics": metrics,
            "detailed_results": results,
            "evaluation_summary": {
                "total_queries": len(results),
                "recall_at_1": metrics['recall_at_1'],
                "recall_at_3": metrics['recall_at_3'],
                "recall_at_5": metrics['recall_at_5'],
                "recall_at_10": metrics['recall_at_10'],
                "mrr": metrics['mean_reciprocal_rank']
            }
        }
        
        print(f"Saving evaluation results to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(output_data, file, indent=2, ensure_ascii=False)
        
        print("Results saved successfully")


def compare_retrieval_systems(results_baseline: List[Dict[str, Any]], 
                            results_enhanced: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare two retrieval systems and calculate improvement metrics."""
    
    # Calculate metrics for both systems
    evaluator = BaseRetrievalEvaluator(None)  # No retriever needed for this
    metrics_baseline = evaluator.calculate_aggregate_metrics(results_baseline)
    metrics_enhanced = evaluator.calculate_aggregate_metrics(results_enhanced)
    
    # Calculate improvements
    improvements = {}
    for metric in ['recall_at_1', 'recall_at_3', 'recall_at_5', 'recall_at_10', 'mean_reciprocal_rank']:
        baseline_val = metrics_baseline[metric]
        enhanced_val = metrics_enhanced[metric]
        improvement = enhanced_val - baseline_val
        improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
        
        improvements[metric] = {
            'baseline': baseline_val,
            'enhanced': enhanced_val,
            'absolute_improvement': improvement,
            'relative_improvement_pct': improvement_pct
        }
    
    return {
        'baseline_metrics': metrics_baseline,
        'enhanced_metrics': metrics_enhanced,
        'improvements': improvements
    }


def print_comparison_results(comparison: Dict[str, Any]):
    """Print formatted comparison between two retrieval systems."""
    
    print(f"\n{'='*80}")
    print("RETRIEVAL SYSTEM COMPARISON")
    print(f"{'='*80}")
    
    improvements = comparison['improvements']
    
    print(f"📊 Performance Comparison:")
    for metric, data in improvements.items():
        metric_name = metric.replace('_', ' ').title()
        baseline = data['baseline']
        enhanced = data['enhanced']
        abs_imp = data['absolute_improvement']
        rel_imp = data['relative_improvement_pct']
        
        direction = "📈" if abs_imp > 0 else "📉" if abs_imp < 0 else "➡️"
        
        print(f"   {metric_name}:")
        print(f"     Baseline: {baseline:.3f} → Enhanced: {enhanced:.3f}")
        print(f"     {direction} {abs_imp:+.3f} ({rel_imp:+.1f}%)")
        print()
    
    # Overall assessment
    recall5_improvement = improvements['recall_at_5']['relative_improvement_pct']
    mrr_improvement = improvements['mean_reciprocal_rank']['relative_improvement_pct']
    
    print(f"💡 Overall Assessment:")
    if recall5_improvement > 5 and mrr_improvement > 5:
        print(f"   • Significant improvement in both recall and ranking quality")
    elif recall5_improvement > 5:
        print(f"   • Good improvement in recall, moderate ranking improvement")
    elif mrr_improvement > 5:
        print(f"   • Good improvement in ranking quality, moderate recall improvement") 
    elif recall5_improvement > 0 or mrr_improvement > 0:
        print(f"   • Modest improvements observed")
    else:
        print(f"   • No significant improvement detected")
    
    print(f"{'='*80}")


def load_queries(queries_path: Path) -> List[Dict[str, Any]]:
    """Load synthetic queries from JSON file."""
    print(f"Loading queries from {queries_path}")
    
    with open(queries_path, 'r', encoding='utf-8') as file:
        queries = json.load(file)
    
    # Handle both direct list format and metadata format
    if isinstance(queries, dict) and 'queries' in queries:
        queries = queries['queries']
    
    print(f"Loaded {len(queries)} queries")
    return queries 