import json
import logging
from typing import List, Dict, Any
import re

logger = logging.getLogger(__name__)

class ResponseParser:
    """Parses and validates the response from Groq LLM."""
    
    @staticmethod
    def parse_response(response_text: str) -> List[Dict[str, Any]]:
        """
        Extracts JSON list from the LLM response text.
        Handles cases where LLM wraps JSON in markdown code blocks.
        """
        try:
            cleaned_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if "```json" in cleaned_text:
                pattern = r"```json(.*?)```"
                match = re.search(pattern, cleaned_text, re.DOTALL)
                if match:
                    cleaned_text = match.group(1).strip()
            elif "```" in cleaned_text:
                pattern = r"```(.*?)```"
                match = re.search(pattern, cleaned_text, re.DOTALL)
                if match:
                    cleaned_text = match.group(1).strip()
            
            # Parse JSON
            data = json.loads(cleaned_text)
            
            if not isinstance(data, list):
                logger.warning("LLM response is not a list.")
                return []
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON content: {e}")
            logger.debug(f"Raw response: {response_text}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in parsing: {e}")
            return []
