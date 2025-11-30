"""
MLE-STAR: Machine Learning Engineering - Search, Target, and Refine

This module implements the MLE-STAR agent that inherits from KaggleAgent
and adds MLE-STAR specific functionality for automated machine learning.
"""

import os
import json
import yaml
import logging
import time
import re
import pickle
import asyncio
import tiktoken
import aiohttp
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
from dataclasses import asdict, dataclass
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from mledo.chat import ChatClient, ModelSettings
from google import genai
from google.genai import types
from mledojo.agent.mleagent.agent import KaggleAgent, LLMConfig

from .prompt import prompts
from .config import MLEStarConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Container for search results."""
    title: str
    url: str
    snippet: str
    source: str = "web"
    metadata: Dict = None


class MLEStarAgent(KaggleAgent):
    """
    MLE-STAR: Machine Learning Engineering - Search, Target, and Refine agent.
    
    This agent implements the MLE-STAR methodology for automated machine learning,
    combining search-based optimization with targeted refinement of ML pipelines.
    """
    
    def __init_web_clients(self):
        """Initialize the web search client."""
        self.web_client = None
        
        # Initialize Perplexity client if API key is available
        perplexity_key = os.getenv('PERPLEXITY_API_KEY')
        if perplexity_key:
            try:
                self.web_client = OpenAI(
                    api_key=perplexity_key,
                    base_url="https://api.perplexity.ai"
                )
                self.logger.info("Initialized Perplexity web client")
            except Exception as e:
                self.logger.error(f"Failed to initialize Perplexity client: {str(e)}")
                self.web_client = None
        
        if not self.web_client:
            self.logger.warning("No web search client available. Set PERPLEXITY_API_KEY environment variable.")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            aiohttp.ClientError,
            asyncio.TimeoutError,
            Exception
        ))
    )
    async def search_web(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Perform web search using Perplexity API.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return (max 20)
            
        Returns:
            List of SearchResult objects
            
        Raises:
            ValueError: If search provider is not configured
            Exception: For any other errors during search
        """
        if not hasattr(self, 'web_client'):
            self.__init_web_clients()
        
        if not self.web_client:
            raise ValueError("Web search is not configured. Set PERPLEXITY_API_KEY environment variable.")
        
        try:
            max_results = min(max(1, max_results), 20)  # Ensure between 1-20
            
            # Make the API request with timeout
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.web_client.chat.completions.create(
                    model="sonar-medium-online",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that performs web searches."},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=2000,
                    temperature=0.7,
                )
            )
            
            # Process the response
            if not response or not response.choices:
                self.logger.warning("Empty response from Perplexity API")
                return []
                
            content = response.choices[0].message.content
            if not content:
                return []
            
            # Return a single result with the full content
            return [SearchResult(
                title=query,
                url="https://perplexity.ai",
                snippet=content[:5000],  # Limit snippet length
                source="perplexity"
            )]
            
        except Exception as e:
            self.logger.error(f"Web search failed: {str(e)}")
            return []
    
    async def _search_perplexity(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search using Perplexity API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects
            
        Raises:
            Exception: If the search fails
        """
        try:
            # Make the API request with timeout
            async with asyncio.timeout(30):  # 30 second timeout
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.web_clients['perplexity'].chat.completions.create(
                        model="sonar-medium-online",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that performs web searches."},
                            {"role": "user", "content": query}
                        ],
                        max_tokens=2000,
                        temperature=0.7,
                    )
                )
            
            # Process the response
            if not response or not response.choices:
                self.logger.warning("Empty response from Perplexity API")
                return []
                
            content = response.choices[0].message.content
            if not content:
                return []
            
            # Parse the response into structured results
            results = []
            for i, line in enumerate(content.split('\n')):
                if i >= max_results:
                    break
                if line.strip():
                    results.append(SearchResult(
                        title=f"Result {i+1}",
                        url=f"#result-{i+1}",
                        snippet=line.strip(),
                        source="perplexity"
                    ))
            
            return results
            
        except asyncio.TimeoutError:
            return []
            
        except Exception as e:
            self.logger.error(f"Web search failed: {str(e)}")
            raise
    
    async def _search_default(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Default search implementation as fallback.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects with limited functionality
        """
        self.logger.warning("Using default search implementation - results may be limited")
        return [
            SearchResult(
                title=f"Result {i+1}",
                url=f"#default-{i+1}",
                snippet=f"Search result for: {query}",
                source="default"
            )
            for i in range(min(3, max_results))
        ]
    
    async def research_topic(self, topic: str, depth: str = "moderate") -> Dict[str, Any]:
        """
        Conduct research on a given topic with configurable depth.
        
        Args:
            topic: The topic to research
            depth: Research depth (quick, moderate, deep)
            
        Returns:
            Dictionary containing:
            - topic: The researched topic
            - sources: List of source information
            - summary: AI-generated summary of findings
            - status: Success/failure status
        """
        try:
            # Generate search queries based on depth
            queries = self._generate_research_queries(topic, depth)
            
            # Execute searches sequentially to avoid rate limiting
            search_results = []
            for query in queries:
                try:
                    results = await self.search_web(query, max_results=1)
                    if results:
                        search_results.extend(results)
                except Exception as e:
                    self.logger.warning(f"Search failed for query '{query}': {str(e)}")
            
            if not search_results:
                return {
                    "topic": topic,
                    "sources": [],
                    "summary": "No results found.",
                    "status": "no_results"
                }
            
            # Analyze and summarize the research
            analysis = await self._analyze_research(topic, search_results)
            
            return {
                "topic": topic,
                "sources": [{"title": r.title, "url": r.url} for r in search_results],
                "summary": analysis.get("summary", "No summary available."),
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Research failed: {str(e)}", exc_info=True)
            return {
                "topic": topic,
                "sources": [],
                "summary": f"Research failed: {str(e)}",
                "status": "error"
            }
    
    async def act(self, obs: Any, action_left: int, time_left: int) -> Tuple[str, Dict]:
        """
        Main method called by the environment to get the agent's action.
        
        Args:
            obs: Current observation from the environment
            action_left: Number of actions remaining
            time_left: Time remaining in seconds
            
        Returns:
            Tuple of (action_type, action_params)
        """
        try:
            # Initialize task on first action
            if not self.task_metadata:
                await self._initialize_task(obs)
            
            # Choose action based on current state
            if not hasattr(self, 'current_phase'):
                self.current_phase = 'research'
            
            if self.current_phase == 'research':
                return await self._research_phase(obs, action_left, time_left)
            elif self.current_phase == 'initial_solution':
                return await self._initial_solution_phase(obs, action_left, time_left)
            elif self.current_phase == 'refinement':
                return await self._refinement_phase(obs, action_left, time_left)
            elif self.current_phase == 'ensembling':
                return await self._ensembling_phase(obs, action_left, time_left)
            else:
                # Default action if phase is unknown
                return "noop", {}
            
        except asyncio.CancelledError:
            self.logger.warning("Research task was cancelled")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse observation: {str(e)}")
            return "request_info", {"info_type": "competition_status"}
            
        except Exception as e:
            self.logger.error(f"Error in act(): {str(e)}", exc_info=True)
            return "request_info", {"info_type": "competition_status"}
        finally:
            self.iteration += 1
        return "request_info", {"info_type": "competition_rules"}
    
    def _extract_competition_slug(self, obs: Any) -> str:
        """Extract the competition slug from the observation."""
        # This is a simplified implementation
        # In practice, you'd need to parse the observation to get the competition slug
        return self.config.kaggle.get("competition", "titanic")
    
    def _save_solution(self, solution: Dict, prefix: str = "") -> str:
        """Save the solution to a file."""
        if not prefix:
            prefix = "solution"
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"
        filepath = self.work_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(solution, f, indent=2)
            
        logger.info(f"Solution saved to {filepath}")
        return str(filepath)
    
    def _init_llm_client(self) -> Any:
        """Initialize the LLM client based on configuration."""
        llm_config = self.llm_config
        
        if llm_config.model_name.startswith("gemini"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.gemini_client = genai
                self.is_experimental_gemini = True
                logger.info(f"Initialized Gemini client with model: {llm_config.model_name}")
                return self.gemini_client
            except ImportError:
                logger.error("Google Generative AI library not installed. Please install with: pip install google-generativeai")
                raise
        else:
            # Use the parent class's LLM client
            return super()._init_llm_client()
    
    def save_state(self, path: Optional[str] = None) -> str:
        """Save the agent's state to a file.
        
        Args:
            path: Path to save the state to. If None, uses a default path.
            
        Returns:
            Path to the saved state file
        """
        if path is None:
            path = self.work_dir / f"mle_star_state_{self.iteration}.pkl"
        
        state = {
            'iteration': self.iteration,
            'best_score': self.best_score,
            'metrics_history': self.metrics_history,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config,
            'conversation_history': self.conversation_history
        }
        
        with open(path, 'wb') as f:
            import pickle
            pickle.dump(state, f)
            
        logger.info(f"Saved agent state to {path}")
        return str(path)
    
    @classmethod
    def load_state(cls, path: str, api_idx: int, api_key: str) -> 'MLEStarAgent':
        """Load an agent from a saved state.
        
        Args:
            path: Path to the saved state file
            api_idx: API index for the LLM client
            api_key: API key for the LLM service
            
        Returns:
            Loaded MLEStarAgent instance
        """
        with open(path, 'rb') as f:
            import pickle
            state = pickle.load(f)
        
        # Create a new agent with the saved config
        config = state.get('config', {})
        llm_config = LLMConfig(**config.get('llm', {}))
        
        agent = cls(
            api_idx=api_idx,
            api_key=api_key,
            llm_config=llm_config,
            config=config
        )
        
        # Restore state
        agent.iteration = state.get('iteration', 0)
        agent.best_score = state.get('best_score', -float('inf'))
        agent.metrics_history = state.get('metrics_history', [])
        agent.conversation_history = state.get('conversation_history', [])
        
        logger.info(f"Loaded agent state from {path}")
        return agent
    
    def _init_openai_client(self) -> ChatClient:
        """Initialize OpenAI-compatible chat client."""
        return ChatClient(
            model_name=self.config.llm.model_name,
            model_category=self.config.llm.get('model_category', 'openai'),
            api_key=self.config.llm.get('api_key', ''),
            port=self.config.llm.get('port', 8000)
        )
    
    def analyze_task(self, task_description: str, data_summary: Dict) -> Dict:
        """Analyze the ML task and generate a plan.
        
        Args:
            task_description: Description of the ML task
            data_summary: Dictionary containing data statistics and metadata
            
        Returns:
            Dictionary containing the analysis and plan
        """
        prompt = prompts.get_prompt(
            'task_analysis_prompt',
            task_description=task_description,
            data_summary=yaml.dump(data_summary, default_flow_style=False)
        )
        
        response = self._query_llm(prompt)
        return {
            'analysis': response,
            'plan': self._extract_plan_from_analysis(response)
        }
    
    def generate_code(self, task: Dict, context: Optional[Dict] = None) -> Dict:
        """Generate code for a given task.
        
        Args:
            task: Dictionary containing task details
            context: Additional context for code generation
            
        Returns:
            Dictionary containing the generated code and metadata
        """
        if context is None:
            context = {}
            
        prompt = prompts.get_prompt(
            'code_generation_prompt',
            task_description=task.get('description', ''),
            requirements='\n'.join([f"- {req}" for req in task.get('requirements', [])]),
            context=json.dumps(context, indent=2)
        )
        
        response = self._query_llm(prompt)
        return {
            'code': self._extract_code_blocks(response),
            'explanation': self._extract_explanation(response),
            'metadata': {
                'task': task,
                'context': context
            }
        }
    
    def refine_code(self, code: str, feedback: str) -> Dict:
        """Refine existing code based on feedback.
        
        Args:
            code: The original code to refine
            feedback: Feedback or issues to address
            
        Returns:
            Dictionary containing the refined code and explanation
        """
        prompt = prompts.get_prompt(
            'refinement_prompt',
            code=code,
            feedback=feedback
        )
        
        response = self._query_llm(prompt)
        return {
            'refined_code': self._extract_code_blocks(response),
            'explanation': self._extract_explanation(response),
            'original_code': code,
            'feedback': feedback
        }
    
    def analyze_error(self, error_message: str, code_context: str) -> Dict:
        """Analyze an error and suggest fixes.
        
        Args:
            error_message: The error message to analyze
            code_context: The relevant code context
            
        Returns:
            Dictionary containing the analysis and suggested fixes
        """
        prompt = prompts.get_prompt(
            'error_analysis_prompt',
            error_message=error_message,
            code_context=code_context
        )
        
        response = self._query_llm(prompt)
        return {
            'analysis': response,
            'suggested_fixes': self._extract_suggested_fixes(response)
        }
    
    def _query_llm(self, prompt: str, **kwargs) -> str:
        """Query the LLM with the given prompt."""
        if isinstance(self.llm_client, genai.Client):
            return self._query_gemini(prompt, **kwargs)
        else:
            return self._query_openai(prompt, **kwargs)
    
    def _query_gemini(self, prompt: str, **kwargs) -> str:
        """Query the Gemini model."""
        try:
            response = self.llm_client.models.generate_content(
                model=self.config.llm.model_name,
                contents=prompt,
                **kwargs
            )
            return response.text
        except Exception as e:
            logger.error(f"Error querying Gemini: {e}")
            raise
    
    def _query_openai(self, prompt: str, **kwargs) -> str:
        """Query an OpenAI-compatible model."""
        try:
            # Add system message if not present
            messages = [{"role": "system", "content": prompts.system_prompt}]
            messages.append({"role": "user", "content": prompt})
            
            response = self.llm_client.chat_completion(
                messages=messages,
                **self.config.llm.to_dict()
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error querying OpenAI-compatible model: {e}")
            raise
    
    def _extract_code_blocks(self, text: str) -> List[Dict]:
        """Extract code blocks from markdown text."""
        import re
        code_blocks = []
        pattern = r'```(?:\w+)?\n(.*?)\n```'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            code_blocks.append({
                'language': 'python',  # Default to Python
                'code': match.group(1).strip()
            })
            
        return code_blocks
    
    def _extract_explanation(self, text: str) -> str:
        """Extract explanation text from LLM response."""
        # Simple implementation - can be enhanced
        return text.split('```')[0].strip()
    
    def _extract_plan_from_analysis(self, analysis: str) -> Dict[str, Any]:
        """Extract structured plan from analysis text.
        
        Args:
            analysis: Raw analysis text from LLM
            
        Returns:
            Dictionary with structured plan containing:
            - problem_understanding: str
            - data_preprocessing: List[str]
            - modeling_approach: List[str]
            - evaluation: List[str]
            - next_steps: List[str]
        """
        try:
            # Try to parse as JSON first
            if analysis.strip().startswith('{') and analysis.strip().endswith('}'):
                return json.loads(analysis)
                
            # Otherwise, extract sections using markdown headers
            plan = {
                'problem_understanding': "",
                'data_preprocessing': [],
                'modeling_approach': [],
                'evaluation': [],
                'next_steps': []
            }
            
            current_section = None
            
            for line in analysis.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Check for section headers
                if line.lower().startswith('## '):
                    section = line[3:].lower().replace(' ', '_')
                    if section in plan:
                        current_section = section
                    continue
                        
                # Add content to current section
                if current_section:
                    if current_section == 'problem_understanding':
                        plan[current_section] += ' ' + line
                    elif line and line not in plan[current_section]:
                        plan[current_section].append(line)
            
            return plan
            
        except Exception as e:
            logger.warning(f"Failed to parse plan from analysis: {e}")
            return {
                'problem_understanding': analysis,
                'data_preprocessing': [],
                'modeling_approach': [],
                'evaluation': [],
                'next_steps': []
            }
    
    def _extract_suggested_fixes(self, analysis: str) -> List[Dict[str, str]]:
        """Extract suggested fixes from error analysis.
        
        Args:
            analysis: Raw error analysis text from LLM
            
        Returns:
            List of dictionaries containing:
            - description: str - Description of the fix
            - priority: str - Priority level (high/medium/low)
            - code: Optional[str] - Code snippet if provided
        """
        try:
            # Try to parse as JSON first
            if analysis.strip().startswith('[') and analysis.strip().endswith(']'):
                return json.loads(analysis)
                
            # Otherwise, extract fixes using markdown formatting
            fixes = []
            current_fix = None
            
            for line in analysis.split('\n'):
                line = line.strip()
                
                # Look for numbered list items
                match = re.match(r'^(\d+)[.)]\s*(.*?)(?:\s*\(([^)]+)\))?$', line)
                if match:
                    if current_fix:
                        fixes.append(current_fix)
                    
                    priority = 'medium'
                    if match.group(3):
                        prio_text = match.group(3).lower()
                        if 'high' in prio_text:
                            priority = 'high'
                        elif 'low' in prio_text:
                            priority = 'low'
                    
                    current_fix = {
                        'description': match.group(2).strip(),
                        'priority': priority,
                        'code': ''
                    }
                # Look for code blocks
                elif line.startswith('```'):
                    if current_fix and '```' in line[3:]:  # Inline code
                        current_fix['code'] = line[3:-3].strip()
                elif current_fix and line and not line.startswith('-'):
                    # Add to current fix description
                    current_fix['description'] += ' ' + line
            
            # Add the last fix if exists
            if current_fix:
                fixes.append(current_fix)
                
            return fixes if fixes else [{
                'description': analysis,
                'priority': 'high',
                'code': ''
            }]
            
        except Exception as e:
            logger.warning(f"Failed to parse suggested fixes: {e}")
            return [{
                'description': analysis or 'An unknown error occurred',
                'priority': 'high',
                'code': ''
            }]
    
    def save_state(self, path: Optional[str] = None) -> str:
        """Save the agent's state to a file.
        
        Args:
            path: Path to save the state file. If None, generates a default path.
            
        Returns:
            Path to the saved state file
        """
        if path is None:
            path = str(self.work_dir / f'mle_star_state_{int(time.time())}.pkl')
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        state = {
            'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else self.config,
            'conversation_history': self.conversation_history,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'iteration': self.iteration,
            'metrics_history': self.metrics_history,
            'class_name': self.__class__.__name__,
            'llm_config': asdict(self.llm_config) if hasattr(self.llm_config, '__dataclass_fields__') else self.llm_config,
            'api_idx': self.api_idx,
            'api_key': '***'  # Don't save API key in state
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        logger.info(f"Saved agent state to {path}")
        return path

    @classmethod
    def load_state(cls, path: str, api_key: Optional[str] = None) -> 'MLEStarAgent':
        """Load an agent's state from a file.
        
        Args:
            path: Path to the saved state file
            api_key: API key for the LLM service. If None, tries to get from environment.
            
        Returns:
            Loaded MLEStarAgent instance
            
        Raises:
            FileNotFoundError: If the state file doesn't exist
            ValueError: If the state file is invalid or corrupted
        """
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
                
            # Get API key from argument or environment
            api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
            if not api_key:
                raise ValueError("API key is required. Either pass it as an argument or set PERPLEXITY_API_KEY environment variable.")
                
            # Create agent instance
            llm_config = LLMConfig(**(state.get('llm_config') or {}))
            config = state.get('config', {})
            
            agent = cls(
                api_idx=state.get('api_idx', 0),
                api_key=api_key,
                llm_config=llm_config,
                config=config
            )
            
            # Restore state
            agent.conversation_history = state.get('conversation_history', [])
            agent.best_model = state.get('best_model')
            agent.best_score = state.get('best_score', -float('inf'))
            agent.iteration = state.get('iteration', 0)
            agent.metrics_history = state.get('metrics_history', [])
            
            logger.info(f"Loaded agent state from {path}")
            return agent
            
        except FileNotFoundError:
            logger.error(f"State file not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load agent state: {e}")
            raise ValueError(f"Invalid or corrupted state file: {e}")
