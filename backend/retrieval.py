#!/usr/bin/env python3
"""
BM25 Recipe Retrieval Engine for HW4 RAG Evaluation

Provides BM25-based recipe retrieval functionality with indexing and search capabilities.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class RecipeRetriever:
    """BM25-based recipe retrieval system."""
    
    def __init__(self):
        self.recipes: List[Dict[str, Any]] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.recipe_id_to_index: Dict[int, int] = {}
        self.index_to_recipe_id: Dict[int, int] = {}
        self.is_indexed = False
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing."""
        if not text:
            return []
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into tokens and remove empty strings
        tokens = [token.strip() for token in text.split() if token.strip()]
        
        return tokens
    
    def load_recipes(self, recipes_path: Path) -> None:
        """Load processed recipes from JSON file."""
        print(f"Loading recipes from {recipes_path}")
        
        with open(recipes_path, 'r', encoding='utf-8') as file:
            self.recipes = json.load(file)
        
        # Create mapping between recipe IDs and list indices
        for idx, recipe in enumerate(self.recipes):
            recipe_id = recipe['id']
            self.recipe_id_to_index[recipe_id] = idx
            self.index_to_recipe_id[idx] = recipe_id
        
        print(f"Loaded {len(self.recipes)} recipes")
    
    def build_index(self) -> None:
        """Build BM25 index from loaded recipes."""
        if not self.recipes:
            raise ValueError("No recipes loaded. Call load_recipes() first.")
        
        print("Building BM25 index...")
        
        # Prepare documents for indexing
        documents = []
        for recipe in tqdm(self.recipes, desc="Preprocessing recipes"):
            # Combine all searchable text fields
            searchable_parts = [
                recipe.get('name', ''),
                recipe.get('description', ''),
                ' '.join(recipe.get('ingredients', [])),
                ' '.join(recipe.get('steps', [])),
                ' '.join(recipe.get('tags', []))
            ]
            
            full_text = ' '.join(filter(None, searchable_parts))
            tokens = self.preprocess_text(full_text)
            documents.append(tokens)
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(documents)
        self.is_indexed = True
        
        print(f"BM25 index built for {len(documents)} recipes")
    
    def save_index(self, index_path: Path) -> None:
        """Save the BM25 index to disk."""
        if not self.is_indexed:
            raise ValueError("No index to save. Call build_index() first.")
        
        index_data = {
            'bm25_index': self.bm25_index,
            'recipe_id_to_index': self.recipe_id_to_index,
            'index_to_recipe_id': self.index_to_recipe_id
        }
        
        with open(index_path, 'wb') as file:
            pickle.dump(index_data, file)
        
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_path: Path) -> None:
        """Load a pre-built BM25 index from disk."""
        print(f"Loading BM25 index from {index_path}")
        
        with open(index_path, 'rb') as file:
            index_data = pickle.load(file)
        
        self.bm25_index = index_data['bm25_index']
        self.recipe_id_to_index = index_data['recipe_id_to_index']
        self.index_to_recipe_id = index_data['index_to_recipe_id']
        self.is_indexed = True
        
        print("BM25 index loaded successfully")
    
    def retrieve_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant recipes using BM25.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of recipe dictionaries with relevance scores
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call build_index() or load_index() first.")
        
        if not self.recipes:
            raise ValueError("No recipes loaded. Call load_recipes() first.")
        
        # Preprocess query
        query_tokens = self.preprocess_text(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Build results with scores
        results = []
        for idx in top_indices:
            if idx < len(self.recipes):
                recipe = self.recipes[idx].copy()
                recipe['bm25_score'] = float(scores[idx])
                recipe['rank'] = len(results) + 1
                results.append(recipe)
        
        return results
    
    def search_by_recipe_id(self, recipe_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific recipe by ID."""
        if recipe_id in self.recipe_id_to_index:
            idx = self.recipe_id_to_index[recipe_id]
            return self.recipes[idx]
        return None
    
    def get_recipe_rank(self, query: str, target_recipe_id: int, top_k: int = 100) -> Optional[int]:
        """
        Get the rank of a specific recipe for a given query.
        
        Args:
            query: Search query
            target_recipe_id: ID of the recipe to find
            top_k: Maximum rank to search (for efficiency)
            
        Returns:
            Rank (1-indexed) of the recipe, or None if not found in top_k
        """
        results = self.retrieve_bm25(query, top_k=top_k)
        
        for i, recipe in enumerate(results):
            if recipe['id'] == target_recipe_id:
                return i + 1  # 1-indexed rank
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded recipes and index."""
        if not self.recipes:
            return {"error": "No recipes loaded"}
        
        stats = {
            "total_recipes": len(self.recipes),
            "indexed": self.is_indexed,
            "avg_ingredients": sum(r.get('n_ingredients', 0) for r in self.recipes) / len(self.recipes),
            "avg_steps": sum(r.get('n_steps', 0) for r in self.recipes) / len(self.recipes),
            "avg_cooking_time": sum(r.get('minutes', 0) for r in self.recipes if r.get('minutes', 0) > 0) / len([r for r in self.recipes if r.get('minutes', 0) > 0])
        }
        
        return stats


def create_retriever(recipes_path: Path, index_path: Optional[Path] = None, rebuild_index: bool = False) -> RecipeRetriever:
    """
    Factory function to create and initialize a RecipeRetriever.
    
    Args:
        recipes_path: Path to processed recipes JSON file
        index_path: Path to save/load BM25 index
        rebuild_index: Whether to rebuild index even if it exists
        
    Returns:
        Initialized RecipeRetriever instance
    """
    retriever = RecipeRetriever()
    retriever.load_recipes(recipes_path)
    
    # Try to load existing index
    if index_path and index_path.exists() and not rebuild_index:
        try:
            retriever.load_index(index_path)
            print("Using existing BM25 index")
        except Exception as e:
            print(f"Failed to load index: {e}")
            print("Building new index...")
            retriever.build_index()
            if index_path:
                retriever.save_index(index_path)
    else:
        # Build new index
        retriever.build_index()
        if index_path:
            retriever.save_index(index_path)
    
    return retriever


# Convenience function for backward compatibility
def retrieve_bm25(query: str, corpus: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Legacy function for compatibility with existing code.
    
    Note: This creates a new index each time, which is inefficient.
    Use RecipeRetriever class for better performance.
    """
    # Create temporary retriever
    retriever = RecipeRetriever()
    retriever.recipes = corpus
    
    # Build mapping
    for idx, recipe in enumerate(corpus):
        recipe_id = recipe.get('id', idx)
        retriever.recipe_id_to_index[recipe_id] = idx
        retriever.index_to_recipe_id[idx] = recipe_id
    
    # Build index
    retriever.build_index()
    
    # Retrieve results
    return retriever.retrieve_bm25(query, top_k=top_n)


def main():
    """Example usage and testing."""
    from pathlib import Path
    
    # Paths
    base_path = Path(__file__).parent.parent / "homeworks" / "hw4"
    recipes_path = base_path / "data" / "processed_recipes.json"
    index_path = base_path / "data" / "bm25_index.pkl"
    
    if not recipes_path.exists():
        print(f"Recipes file not found: {recipes_path}")
        print("Run process_recipes.py first")
        return
    
    # Create retriever
    retriever = create_retriever(recipes_path, index_path)
    
    # Print stats
    stats = retriever.get_stats()
    print(f"\n--- Retriever Stats ---")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test some queries
    test_queries = [
        "air fryer chicken",
        "vegan pasta",
        "gluten free bread",
        "chocolate chip cookies",
        "stir fry vegetables"
    ]
    
    print(f"\n--- Test Retrieval ---")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = retriever.retrieve_bm25(query, top_k=3)
        
        for i, recipe in enumerate(results):
            print(f"  {i+1}. {recipe['name']} (Score: {recipe['bm25_score']:.3f})")


if __name__ == "__main__":
    main() 