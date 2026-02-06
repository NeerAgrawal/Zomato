from typing import Dict, Any

class PromptDesign:
    """Manages prompt construction for the Groq LLM."""
    
    SYSTEM_PROMPT = """
    You are an expert restaurant recommendation assistant for Zomato.
    Your goal is to recommend the best restaurants from the provided list based on the user's city and budget.
    
    Rules:
    1. Select the top {{top_k}} restaurants that best match the criteria.
    2. Rank them from 1 to {{top_k}}.
    3. Provide a **UNIQUE** and **SPECIFIC** reason for EACH recommendation.
    4. You MUST mention specific attributes from the data (e.g., "Best for North Indian," "High rating of 4.5/5," "Great value at 500," "Known for Biryani").
    5. **DO NOT** use generic phrases like "Good restaurant" or "Fits criteria" for all items. varied reasoning is required.
    6. Output STRICT JSON format only. No markdown, no explanations outside the JSON.
    
    Output Format:
    [
        {
            "name": "Restaurant Name",
            "rank": 1,
            "reason": "Specific reason citing rating, cuisine, or popularity found in context."
        },
        ...
    ]
    """
    
    USER_PROMPT_TEMPLATE = """
    User Request:
    - City: {{city}}
    - Maximum Budget (for two): {{price}}
    
    Candidate Restaurants:
    {{context}}
    
    Please provide the top {{top_k}} recommendations in the specified JSON format.
    """
    
    @staticmethod
    def build_system_prompt(top_k: int = 10) -> str:
        """Returns the formatted system prompt."""
        return PromptDesign.SYSTEM_PROMPT.replace("{{top_k}}", str(top_k))
    
    @staticmethod
    def build_user_prompt(user_input: Dict[str, Any], context: str, top_k: int = 10) -> str:
        """Returns the formatted user prompt with context."""
        prompt = PromptDesign.USER_PROMPT_TEMPLATE
        prompt = prompt.replace("{{city}}", str(user_input.get('city', 'Unknown')))
        prompt = prompt.replace("{{price}}", str(user_input.get('price', 'N/A')))
        prompt = prompt.replace("{{context}}", context)
        prompt = prompt.replace("{{top_k}}", str(top_k))
        return prompt
