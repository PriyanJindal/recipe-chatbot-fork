#!/usr/bin/env python3
"""
Query Rewrite Agent for Improved Recipe Retrieval

Uses LLM to rewrite natural language queries into more effective search terms
for BM25 retrieval, focusing on extracting key cooking terms and techniques.
"""

import litellm
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()


class QueryRewriteAgent:
    """LLM-powered agent for optimizing retrieval queries."""
    
    def __init__(self, model: str = "gpt-4.1-nano", max_workers: int = 32):
        self.model = model
        self.max_workers = max_workers
    
    def extract_search_keywords(self, query: str) -> str:
        """
        Extract the most important search keywords from a natural language query.
        """
        prompt = f"""
You are a search optimization expert for a recipe database. Given a natural language cooking query, extract the most important keywords that would help find relevant recipes in a BM25 search.

Focus on:
1. **Cooking methods** (air fry, bake, grill, sauté, etc.)
2. **Equipment/appliances** (air fryer, oven, pressure cooker, etc.)
3. **Key ingredients** (chicken, vegetables, pasta, etc.)
4. **Cooking specifics** (temperature, time, texture, etc.)
5. **Food types** (appetizer, dessert, main dish, etc.)

Remove filler words and focus on terms that would appear in recipe instructions or ingredients.

Query: "{query}"

Important search keywords (space-separated):
"""
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100
            )
            keywords = response.choices[0].message.content.strip()
            return keywords
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return query  # Fallback to original query
    
    def rewrite_for_search(self, query: str) -> str:
        """
        Rewrite a natural language query to be more effective for BM25 search.
        """
        prompt = f"""
You are optimizing a cooking query for recipe search. Rewrite the query to be more effective for finding relevant recipes, focusing on terms that would appear in recipe titles, ingredients, and instructions.

Guidelines:
1. Use specific cooking terms instead of vague language
2. Include equipment names if mentioned
3. Add related cooking techniques
4. Use ingredient names that commonly appear in recipes
5. Keep it concise but descriptive
6. Remove question words (what, how, when) and focus on content

Original query: "{query}"

Optimized search query:
"""
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            rewritten = response.choices[0].message.content.strip()
            return rewritten
        except Exception as e:
            print(f"Error rewriting query: {e}")
            return query  # Fallback to original query
    
    def expand_query_with_synonyms(self, query: str) -> str:
        """
        Expand a query with cooking-related synonyms and related terms.
        """
        prompt = f"""
Expand this cooking query by adding relevant synonyms and related cooking terms that might appear in recipes. This helps catch more relevant results in recipe search.

Add terms that are:
1. Synonyms for cooking methods mentioned
2. Alternative ingredient names
3. Related cooking techniques
4. Equipment alternatives

Keep the expansion focused and avoid unrelated terms.

Original: "{query}"

Expanded query with synonyms:
"""
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=200
            )
            expanded = response.choices[0].message.content.strip()
            return expanded
        except Exception as e:
            print(f"Error expanding query: {e}")
            return query  # Fallback to original query
    
    def process_query(self, query: str, strategy: str = "rewrite") -> Dict[str, str]:
        """
        Process a query using the specified strategy.
        
        Args:
            query: Original natural language query
            strategy: "keywords", "rewrite", or "expand"
            
        Returns:
            Dictionary with original and processed query
        """
        if strategy == "keywords":
            processed = self.extract_search_keywords(query)
        elif strategy == "rewrite":
            processed = self.rewrite_for_search(query)
        elif strategy == "expand":
            processed = self.expand_query_with_synonyms(query)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return {
            "original_query": query,
            "processed_query": processed,
            "strategy": strategy
        }
    
    def _process_query_with_retry(self, query: str, strategy: str, max_retries: int = 3) -> Dict[str, str]:
        """Process a query with retry logic for robustness."""
        for attempt in range(max_retries):
            try:
                return self.process_query(query, strategy)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to process query after {max_retries} attempts: {query[:50]}...")
                    return {
                        "original_query": query,
                        "processed_query": query,  # Fallback to original
                        "strategy": strategy
                    }
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
    def batch_process_queries(self, queries: List[str], strategy: str = "rewrite") -> List[Dict[str, str]]:
        """
        Process multiple queries in parallel using ThreadPoolExecutor.
        
        Args:
            queries: List of query strings to process
            strategy: Processing strategy to use for all queries
            
        Returns:
            List of processed query dictionaries
        """
        if not queries:
            return []
        
        print(f"Processing {len(queries)} queries with {strategy} strategy using {self.max_workers} workers...")
        
        results = [None] * len(queries)  # Pre-allocate to maintain order
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._process_query_with_retry, query, strategy): i
                for i, query in enumerate(queries)
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_index), total=len(queries), desc=f"Processing {strategy}"):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    print(f"Error processing query {index}: {e}")
                    # Fallback result
                    results[index] = {
                        "original_query": queries[index],
                        "processed_query": queries[index],
                        "strategy": strategy
                    }
        
        return results
    
    def batch_process_multiple_strategies(self, queries: List[str], strategies: List[str] = None) -> Dict[str, List[Dict[str, str]]]:
        """
        Process queries with multiple strategies in parallel.
        
        Args:
            queries: List of query strings
            strategies: List of strategies to test (default: all three)
            
        Returns:
            Dictionary mapping strategy name to list of results
        """
        if strategies is None:
            strategies = ["keywords", "rewrite", "expand"]
        
        print(f"Processing {len(queries)} queries with {len(strategies)} strategies...")
        
        strategy_results = {}
        
        # Process each strategy in parallel
        with ThreadPoolExecutor(max_workers=len(strategies)) as strategy_executor:
            strategy_futures = {
                strategy_executor.submit(self.batch_process_queries, queries, strategy): strategy
                for strategy in strategies
            }
            
            for future in as_completed(strategy_futures):
                strategy = strategy_futures[future]
                try:
                    results = future.result()
                    strategy_results[strategy] = results
                    print(f"✅ Completed {strategy} strategy")
                except Exception as e:
                    print(f"❌ Failed {strategy} strategy: {e}")
                    strategy_results[strategy] = []
        
        return strategy_results


def compare_query_strategies(agent: QueryRewriteAgent, query: str) -> Dict[str, str]:
    """
    Compare all query processing strategies for a single query.
    """
    strategies = ["keywords", "rewrite", "expand"]
    results = {
        "original": query
    }
    
    for strategy in strategies:
        processed = agent.process_query(query, strategy)
        results[strategy] = processed["processed_query"]
    
    return results


def main():
    """Example usage and testing."""
    agent = QueryRewriteAgent(max_workers=5)  # Use 5 workers for demo
    
    # Test queries
    test_queries = [
        "What air fryer settings for frozen chicken tenders?",
        "How long to marinate beef for Korean bulgogi?",
        "What's the exact temperature for crispy roasted vegetables?",
        "How do I get the right consistency for homemade pasta dough?",
        "What oven temp for perfectly baked chocolate chip cookies?",
        "How to make crispy bacon in the oven?",
        "What's the best way to cook salmon fillets?",
        "How long to bake a whole chicken at 375?"
    ]
    
    print("=== Query Rewrite Agent Testing ===\n")
    
    # Demo 1: Single query comparison
    print("1. Single Query Strategy Comparison:")
    query = test_queries[0]
    print(f"Query: {query}")
    
    comparisons = compare_query_strategies(agent, query)
    
    print(f"  Keywords:  {comparisons['keywords']}")
    print(f"  Rewritten: {comparisons['rewrite']}")
    print(f"  Expanded:  {comparisons['expand']}")
    print()
    
    # Demo 2: Batch processing with single strategy
    print("2. Batch Processing (Rewrite Strategy):")
    start_time = time.time()
    batch_results = agent.batch_process_queries(test_queries, "rewrite")
    elapsed = time.time() - start_time
    
    print(f"Processed {len(test_queries)} queries in {elapsed:.2f} seconds")
    for result in batch_results[:3]:  # Show first 3
        print(f"  Original: {result['original_query']}")
        print(f"  Rewritten: {result['processed_query']}")
        print()
    
    # Demo 3: Multiple strategies in parallel
    print("3. Multiple Strategies Processing:")
    start_time = time.time()
    multi_results = agent.batch_process_multiple_strategies(test_queries[:5])  # Use fewer for demo
    elapsed = time.time() - start_time
    
    print(f"Processed {len(test_queries[:5])} queries with 3 strategies in {elapsed:.2f} seconds")
    
    for strategy, results in multi_results.items():
        if results:
            print(f"\n{strategy.upper()} Results:")
            for result in results[:2]:  # Show first 2
                print(f"  {result['original_query']} → {result['processed_query']}")


if __name__ == "__main__":
    main() 