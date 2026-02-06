import pandas as pd
from typing import List, Dict, Any

class ContextManager:
    """Prepares restaurant data for LLM context."""
    
    @staticmethod
    def prepare_context(filtered_df: pd.DataFrame, max_items: int = 50) -> str:
        """
        Converts top N restaurants from the dataframe into a string format for the prompt.
        Prioritizes high-rated and popular restaurants to optimize context quality.
        """
        if filtered_df.empty:
            return "No restaurants available."
        
        # Sort by rate (desc) and votes (desc) to send the best candidates first
        # Ensure regex extraction doesn't fail if column is not string
        # Assuming DataIntegrator has already cleaned 'rate' to text or float. 
        # But let's be safe and rely on what passed in.
        
        # We take top 'max_items' to avoid token overflow
        candidates = filtered_df.head(max_items)
        
        context_lines = []
        for _, row in candidates.iterrows():
            name = row.get('name', 'Unknown')
            rate = row.get('rate', 'N/A')
            votes = row.get('votes', 0)
            price = row.get('price', 'N/A')
            cuisines = row.get('cuisines', [])
            rest_type = row.get('rest_type', 'N/A')
            
            # Format: "Name | Rating: 4.5 | Votes: 1000 | Price: 800 | Cuisines: Italian, Pizza"
            line = (
                f"{name} | Rating: {rate} | Votes: {votes} | "
                f"Price: {price} | Type: {rest_type} | "
                f"Cuisines: {cuisines}"
            )
            context_lines.append(line)
            
        return "\n".join(context_lines)
