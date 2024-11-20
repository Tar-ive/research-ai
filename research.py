import streamlit as st
import requests
import json
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

# Configuration
@dataclass
class AgentConfig:
    system_prompt: str
    model_name: str
    temperature: float
    max_tokens: int = 2048
    perplexity_enabled: bool = False

AGENT_CONFIGS = {
    "vision_agent": AgentConfig(
        system_prompt="""You are the Courage & Vision Agent inspired by Richard Hamming's principles:
        1. Focus on identifying truly important problems
        2. Evaluate potential impact and significance
        3. Challenge conventional thinking
        4. Look for fundamental breakthroughs
        
        Analyze the research question and provide:
        1. Assessment of importance
        2. Potential impact
        3. Novel angles
        4. Possible breakthroughs
        
        {helper_response}""",
        model_name="gemma2-9b-it",
        temperature=0.7,
        perplexity_enabled=True
    ),
    
    "strategy_agent": AgentConfig(
        system_prompt="""You are the Strategic Planning Agent following Hamming's methodology:
        1. Create research opportunities
        2. Plan resource allocation
        3. Identify collaboration possibilities
        4. Design clear research paths
        
        Provide:
        1. Clear research strategy
        2. Resource requirements
        3. Timeline estimates
        4. Key milestones
        
        {helper_response}""",
        model_name="llama-3.1-8b-instant",
        temperature=0.4
    ),
    
    "analysis_agent": AgentConfig(
        system_prompt="""You are the Critical Analysis Agent implementing Hamming's analytical approach:
        1. Rigorously evaluate ideas
        2. Question assumptions
        3. Identify potential obstacles
        4. Suggest improvements
        
        Provide:
        1. Critical analysis
        2. Potential challenges
        3. Methodological considerations
        4. Improvement suggestions
        
        {helper_response}""",
        model_name="llama-3.1-70b-versatile",
        temperature=0.2,
        perplexity_enabled=True
    ),
    
    "knowledge_agent": AgentConfig(
        system_prompt="""You are the Knowledge Integration Agent based on Hamming's principles:
        1. Connect ideas across disciplines
        2. Identify patterns and relationships
        3. Suggest novel combinations
        4. Build on existing knowledge
        
        Provide:
        1. Cross-disciplinary connections
        2. Related research areas
        3. Novel combinations
        4. Knowledge synthesis
        
        {helper_response}""",
        model_name="mixtral-8x7b-32768",
        temperature=0.5,
        perplexity_enabled=True
    )
}

# Perplexity API Configuration
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')

# Logging Setup
class LogManager:
    @staticmethod
    def setup():
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(
                    f'logs/research_moa_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                ),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger('ResearchMOA')

# API Client
class PerplexityClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or PERPLEXITY_API_KEY
        self.url = "https://api.perplexity.ai/chat/completions"
        self.logger = logging.getLogger('ResearchMOA.PerplexityClient')

    def research_query(
        self, 
        query: str, 
        agent_name: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        self.logger.info(f"[{agent_name}] Making research query: {query[:100]}...")
        
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or AGENT_CONFIGS[agent_name].system_prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": AGENT_CONFIGS[agent_name].temperature,
            "top_p": 0.9,
            "return_citations": True,
            "search_domain_filter": ["perplexity.ai"],
            "return_images": False,
            "return_related_questions": True,
            "search_recency_filter": "month"
        }
        
        try:
            self.logger.debug(f"[{agent_name}] Sending request with payload: {json.dumps(payload, indent=2)}")
            response = requests.post(
                self.url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            self.logger.info(f"[{agent_name}] Successfully received response")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[{agent_name}] API request failed: {str(e)}")
            raise

class ResearchResultProcessor:
    def __init__(self):
        self.logger = logging.getLogger('ResearchMOA.ResultProcessor')

    def _safely_extract_content(self, response_data: Dict[str, Any]) -> str:
        """Safely extract content from response data"""
        try:
            if isinstance(response_data, dict):
                if 'choices' in response_data and isinstance(response_data['choices'], list):
                    first_choice = response_data['choices'][0]
                    if isinstance(first_choice, dict) and 'message' in first_choice:
                        return first_choice['message'].get('content', '')
                
                if 'content' in response_data:
                    return response_data['content']
                
                if 'response' in response_data:
                    return response_data['response']
            
            self.logger.warning(f"Unexpected response structure: {type(response_data)}")
            return str(response_data)
            
        except Exception as e:
            self.logger.error(f"Error extracting content: {str(e)}", exc_info=True)
            return "Error extracting content from response"

    def _process_citations(self, response_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Process citations from response"""
        citations = []
        try:
            raw_citations = response_data.get('citations', [])
            for citation in raw_citations:
                if isinstance(citation, dict):
                    citations.append({
                        'text': citation.get('text', 'No text available'),
                        'url': citation.get('url', '#')
                    })
            return citations
        except Exception as e:
            self.logger.error(f"Error processing citations: {str(e)}", exc_info=True)
            return []

    def _display_citations(self, citations: List[Dict[str, str]]) -> None:
        """Display citations in the UI"""
        if citations:
            with st.expander("📚 Research Citations", expanded=False):
                for citation in citations:
                    st.markdown(f"- {citation['text']}\n  [Source]({citation['url']})")

    def process_agent_results(self, agent_name: str, response_data: Dict[str, Any]) -> None:
        """Process results for any agent"""
        self.logger.info(f"Processing {agent_name} results")
        try:
            # Extract content
            content = self._safely_extract_content(response_data)
            if not content:
                st.warning(f"No content received from {agent_name}")
                return

            # Process based on agent type
            if agent_name == "vision_agent":
                self._process_vision_results(content)
            elif agent_name == "strategy_agent":
                self._process_strategy_results(content)
            elif agent_name == "analysis_agent":
                self._process_analysis_results(content)
            elif agent_name == "knowledge_agent":
                self._process_knowledge_results(content)

            # Process citations if available
            citations = self._process_citations(response_data)
            self._display_citations(citations)

        except Exception as e:
            self.logger.error(f"Error processing {agent_name} results: {str(e)}", exc_info=True)
            st.error(f"Error processing {agent_name} results")

    def _process_vision_results(self, content: str) -> None:
        """Process Vision Agent results"""
        st.subheader("🔍 Vision & Impact Analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Key Insights")
            st.write(content)
        
        with col2:
            st.markdown("### Impact Assessment")
            metrics = {
                "Innovation": "High",
                "Research Gap": "Significant",
                "Feasibility": "Medium"
            }
            for metric, value in metrics.items():
                st.metric(metric, value)

    def _process_strategy_results(self, content: str) -> None:
        """Process Strategy Agent results"""
        st.subheader("📋 Research Strategy")
        
        # Timeline
        st.markdown("### Research Timeline")
        phases = ["Initial Research", "Methodology", "Implementation", "Validation"]
        cols = st.columns(len(phases))
        for i, (col, phase) in enumerate(zip(cols, phases)):
            with col:
                st.markdown(f"**Phase {i+1}**: {phase}")
                st.progress((i + 1) * 0.25)
        
        # Main content
        st.markdown("### Strategic Plan")
        st.write(content)

    def _process_analysis_results(self, content: str) -> None:
        """Process Analysis Agent results"""
        st.subheader("🔬 Critical Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Strengths", "Challenges", "Recommendations"])
        with tab1:
            st.markdown("### Key Strengths")
            st.write(content)
            
        with tab2:
            st.markdown("### Potential Challenges")
            st.write(content)
            
        with tab3:
            st.markdown("### Recommendations")
            st.write(content)

    def _process_knowledge_results(self, content: str) -> None:
        """Process Knowledge Integration Agent results"""
        st.subheader("🧠 Knowledge Integration")
        
        st.markdown("### Cross-disciplinary Connections")
        st.write(content)
        
        with st.expander("💡 Innovation Opportunities", expanded=False):
            st.write(content)

def main():
    # Setup logging
    logger = LogManager.setup()
    logger.info("Starting Research MOA application")
    
    # Streamlit configuration
    st.set_page_config(
        page_title="Research MOA System",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("Research Assistant - Mixture of Agents")
    st.markdown("*Powered by Hamming's Principles on Great Research*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Active Agents")
        active_agents = {
            name: st.checkbox(
                f"{name.replace('_', ' ').title()}", 
                value=True,
                help=f"Using {config.model_name}"
            )
            for name, config in AGENT_CONFIGS.items()
        }
        
        # Display logs in sidebar
        with st.expander("📋 Process Logs", expanded=False):
            if Path("logs").exists():
                latest_log = max(Path("logs").glob("*.log"))
                with open(latest_log) as f:
                    st.code(f.read())
    
    # Main interface
    research_question = st.text_area(
        "Enter your research question or topic:",
        help="Be specific about what you want to investigate"
    )
    
    if st.button("Analyze") and research_question:
        logger.info(f"Processing research question: {research_question[:100]}...")
        
        try:
            perplexity_client = PerplexityClient()
            processor = ResearchResultProcessor()
            
            # Create tabs for results
            tabs = st.tabs([
                "Vision & Impact",
                "Strategy",
                "Analysis",
                "Knowledge Integration"
            ])
            
            # Process each active agent
            for tab_idx, (agent_name, is_active) in enumerate(active_agents.items()):
                if is_active:
                    logger.info(f"Processing {agent_name}")
                    with tabs[tab_idx]:
                        try:
                            response = perplexity_client.research_query(
                                research_question,
                                agent_name
                            )
                            processor.process_agent_results(agent_name, response)
                            
                        except Exception as e:
                            logger.error(f"Error in {agent_name}: {str(e)}", exc_info=True)
                            st.error(f"Error processing {agent_name}")
                            
        except Exception as e:
            logger.error(f"Application error: {str(e)}", exc_info=True)
            st.error("An error occurred while processing your request")

if __name__ == "__main__":
    main()