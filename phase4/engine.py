import os
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from groq import Groq

from .prompts import PromptDesign
from .context import ContextManager
from .parser import ResponseParser

logger = logging.getLogger(__name__)

class GroqEngine:
    """
    Orchestrates the recommendation process using Groq LLM.
    Follows Step 4 of Architecture: Prompt -> Call -> Parse.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.client = None
        
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info("Groq client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
        else:
            logger.warning("GROQ_API_KEY not found. LLM features will be disabled.")

    def generate_recommendations(self, 
                               filtered_df: pd.DataFrame, 
                               user_input: Dict[str, Any], 
                               top_k: int = 10) -> pd.DataFrame:
        """
        Generates recommendations using LLM if available, otherwise falls back to rule-based.
        """
        if self.client and not filtered_df.empty:
            try:
                logger.info("Generating recommendations via Groq LLM...")
                
                # 1. Prepare Context
                context_str = ContextManager.prepare_context(filtered_df, max_items=50)
                
                # 2. Build Prompts
                system_prompt = PromptDesign.build_system_prompt(top_k)
                user_prompt = PromptDesign.build_user_prompt(user_input, context_str, top_k)
                
                # 3. Call API
                completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3, # Low temp for consistent formatting
                    max_tokens=1024
                )
                
                response_content = completion.choices[0].message.content
                
                # 4. Parse Response
                recommendations_data = ResponseParser.parse_response(response_content)
                
                if recommendations_data:
                    return self._merge_recommendations(filtered_df, recommendations_data)
                else:
                    logger.warning("Parsing failed or empty response. Using fallback.")
                    
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                
        # Fallback
        return self._fallback_ranking(filtered_df, top_k)

    def _merge_recommendations(self, original_df: pd.DataFrame, rec_data: List[Dict]) -> pd.DataFrame:
        """
        Merges LLM reasoning and ranking back into the original dataframe.
        """
        # Create a dataframe from the recommendations
        rec_df = pd.DataFrame(rec_data)
        
        # We need to map back to original rows to get full details (address, url, etc.)
        # We'll match on 'name'. This assumes names are relatively unique or exact matches.
        # Ideally, we'd pass IDs to LLM, but names are more token-efficient for context.
        
        merged_results = []
        
        for _, rec in rec_df.iterrows():
            name = rec.get('name')
            # Find matching row in original_df (case insensitive)
            match = original_df[original_df['name'].str.lower() == name.lower()]
            
            if not match.empty:
                # Take the first match
                row = match.iloc[0].copy()
                row = match.iloc[0].copy()
                reason = rec.get('reason', 'Recommended by AI')
                # Append a subtle tag to confirm it comes from the LLM
                row['reason'] = f"{reason}" 
                row['llm_rank'] = rec.get('rank', 99)
                
                # Deduplication check
                if row['name'] not in [r['name'] for r in merged_results]:
                    merged_results.append(row)
                
        if not merged_results:
            logger.warning("No matching restaurants found after merge. Returning fallback.")
            return self._fallback_ranking(original_df, 10)
            
        result_df = pd.DataFrame(merged_results)
        return result_df

    def _fallback_ranking(self, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """Rule-based sorting as described in architecture."""
        logger.info("Using fallback rule-based ranking.")
        
        # Sort by rate (desc) -> votes (desc) -> price (asc)
        # Ensure correct types for sorting
        df_copy = df.copy()
        
        # Convert rate to numeric if needed (assuming it might be string)
        # In Phase 2 data_integration, it's already engineered, but let's be safe
        
        sort_cols = ['rate', 'votes', 'price']
        ascending = [False, False, True]
        
        # filter available columns
        actual_cols = [c for c in sort_cols if c in df_copy.columns]
        actual_asc = [ascending[i] for i, c in enumerate(sort_cols) if c in df_copy.columns]
        
        if actual_cols:
            df_sorted = df_copy.sort_values(by=actual_cols, ascending=actual_asc)
        else:
            df_sorted = df_copy
            
        df_sorted['reason'] = "Best match based on rating and popularity."
        return df_sorted.head(top_k)
