"""
Perplexity API client for web search functionality in MLE-STAR.
"""
import os
import requests
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("mlestar")

class PerplexityClient:
    """Client for Perplexity API to perform web searches."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Perplexity client.
        
        Args:
            api_key: Perplexity API key. If None, will try to get from PERPLEXITY_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get('PERPLEXITY_API_KEY')
        if not self.api_key:
            raise ValueError("Perplexity API key must be provided or set as PERPLEXITY_API_KEY environment variable")
        
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def search(self, query: str, model: str = "llama-3.1-sonar-large-128k-online") -> Dict[str, Any]:
        """
        Perform a web search using Perplexity API.
        
        Args:
            query: Search query string
            model: Perplexity model to use (default: llama-3.1-sonar-large-128k-online for web search)
            
        Returns:
            Dictionary containing search results with citations
        """
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides accurate information with citations."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": 0.2,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(self.base_url, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract the content and citations
            choices = result.get('choices', [])
            if choices and len(choices) > 0:
                content = choices[0].get('message', {}).get('content', '')
            else:
                content = result.get('content', '')
            
            citations = result.get('citations', [])
            
            return {
                'content': content,
                'citations': citations,
                'raw_response': result
            }
        except requests.exceptions.Timeout:
            logger.error("Perplexity API request timed out")
            return {
                'content': "Error: Request timed out",
                'citations': [],
                'raw_response': None
            }
        except requests.exceptions.HTTPError as e:
            logger.error(f"Perplexity API HTTP error: {e.response.status_code if hasattr(e, 'response') else 'Unknown'}")
            return {
                'content': f"Error: HTTP {e.response.status_code if hasattr(e, 'response') else 'Unknown'}",
                'citations': [],
                'raw_response': None
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Perplexity API error: {e}")
            return {
                'content': f"Error performing search: {str(e)}",
                'citations': [],
                'raw_response': None
            }
        except Exception as e:
            logger.error(f"Unexpected error in Perplexity search: {e}")
            return {
                'content': f"Unexpected error: {str(e)}",
                'citations': [],
                'raw_response': None
            }
    
    def search_models(self, task_description: str, num_models: int = 5) -> List[Dict[str, str]]:
        """
        Search for recent effective models for a given task.
        Uses MLE-STAR prompt 1.
        
        Args:
            task_description: Description of the ML task/competition
            num_models: Number of models to retrieve
            
        Returns:
            List of dictionaries with 'model_name' and 'example_code' keys
        """
        prompt = f"""# Competition
{task_description}

# Your task
- List {num_models} recent effective models and their example codes to win the above competition.

# Requirement
- The example code should be concise and simple.
- You must provide an example code, i.e., do not just mention GitHubs or papers.

Use this JSON schema:
Model = {{'model_name': str, 'example_code': str}}
Return: list[Model]"""
        
        result = self.search(prompt)
        # Parse the JSON response to extract models
        # This is a simplified version - in practice, you'd want more robust parsing
        return self._parse_models_from_response(result['content'], num_models)
    
    def _parse_models_from_response(self, content: str, num_models: int) -> List[Dict[str, str]]:
        """
        Parse model information from Perplexity response.
        This is a simplified parser - you may want to enhance it.
        """
        models = []
        import json
        import re
        
        if not content:
            logger.warning("Empty response from Perplexity")
            return self._create_placeholder_models(num_models)
        
        # Try to extract JSON from the response
        json_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list) and len(parsed) > 0:
                    # Validate structure
                    valid_models = []
                    for item in parsed:
                        if isinstance(item, dict) and 'model_name' in item and 'example_code' in item:
                            valid_models.append({
                                'model_name': str(item['model_name']),
                                'example_code': str(item['example_code'])
                            })
                    if valid_models:
                        return valid_models[:num_models]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.debug(f"JSON parsing failed: {e}")
        
        # Fallback: try to extract model names and code blocks
        code_blocks = re.findall(r'```(?:python)?\n(.*?)```', content, re.DOTALL)
        model_names = re.findall(r'model[_\s]*name[:\s]*["\']?([^"\'\n]+)["\']?', content, re.IGNORECASE)
        
        # Also try to find model names in markdown lists or numbered lists
        if not model_names:
            model_names = re.findall(r'(?:^|\n)\s*(?:[-*]|\d+\.)\s*([A-Za-z][^:\n]+?)(?::|model|approach)', content, re.MULTILINE | re.IGNORECASE)
        
        for i, (name, code) in enumerate(zip(model_names[:num_models], code_blocks[:num_models])):
            models.append({
                'model_name': name.strip() if name else f"Model_{i+1}",
                'example_code': code.strip() if code else f"# Code for {name or f'Model_{i+1}'}"
            })
        
        # If we have code blocks but no names, use generic names
        if code_blocks and not model_names:
            for i, code in enumerate(code_blocks[:num_models]):
                models.append({
                    'model_name': f"Model_{i+1}",
                    'example_code': code.strip()
                })
        
        # If we still don't have models, create placeholder entries
        return models[:num_models] if models else self._create_placeholder_models(num_models)
    
    def _create_placeholder_models(self, num_models: int) -> List[Dict[str, str]]:
        """Create placeholder model entries when parsing fails."""
        return [{
            'model_name': f"Model_{i+1}",
            'example_code': f"# Model {i+1} code not found in search results\n# Please implement based on task description"
        } for i in range(num_models)]

