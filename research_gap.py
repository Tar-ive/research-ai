import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Any
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
from research import PerplexityClient

@dataclass
class AgentConfig:
    system_prompt: str
    model_name: str
    temperature: float
    max_tokens: int = 2048
    perplexity_enabled: bool = False

@dataclass
class ResearchStep:
    name: str
    agent: str
    description: str
    output_format: Dict[str, Any]

RESEARCH_WORKFLOW = [
    ResearchStep(
        name="Literature Collection",
        agent="strategy_agent",
        description="Plan and execute systematic literature search",
        output_format={
            "search_strategy": str,
            "collected_papers": List[Dict],
            "search_metrics": Dict
        }
    ),
    ResearchStep(
        name="Content Analysis",
        agent="analysis_agent",
        description="Analyze papers for themes and methodologies",
        output_format={
            "themes": List[str],
            "methodologies": List[str],
            "findings": List[Dict]
        }
    ),
    ResearchStep(
        name="Gap Identification",
        agent="vision_agent",
        description="Identify important unexplored areas",
        output_format={
            "research_gaps": List[Dict],
            "impact_assessment": Dict
        }
    ),
    ResearchStep(
        name="Interdisciplinary Connections",
        agent="knowledge_agent",
        description="Suggest cross-disciplinary research opportunities",
        output_format={
            "connections": List[Dict],
            "novel_approaches": List[str]
        }
    )
]

class ResearchGapAnalyzer:
    def __init__(self):
        self.logger = self._setup_logging()
        self.perplexity_client = PerplexityClient()

    def _setup_logging(self):
        Path("logs").mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(f'logs/gap_analysis_{datetime.now():%Y%m%d_%H%M%S}.log'),
                logging.StreamHandler()
            ]
        )
    
    def execute_workflow(self, research_area: str, discipline: str) -> Dict[str, Any]:
        results = {}
        for step in RESEARCH_WORKFLOW:
            self.logger.info(f"Executing {step.name}")
            
            # Configure agent for discipline
            agent_config = self._get_discipline_config(step.agent, discipline)
            
            try:
                query = self._format_query(research_area, step)
                response = self.perplexity_client.research_query(
                    query=query,
                    agent_name=step.agent,
                    system_prompt=agent_config.system_prompt
                )
                results[step.name] = self._process_response(response, step.output_format)
                
            except Exception as e:
                self.logger.error(f"Error in {step.name}: {str(e)}")
                results[step.name] = {"error": str(e)}
                
        return results

    def _format_query(self, research_area: str, step: ResearchStep) -> str:
        query_templates = {
            "Literature Collection": f"Conduct a systematic literature search for research in {research_area}.",
            "Content Analysis": f"Analyze the key themes and methodologies in {research_area} research.",
            "Gap Identification": f"Identify important unexplored areas in {research_area}.",
            "Interdisciplinary Connections": f"Suggest cross-disciplinary connections for {research_area}."
        }
        return query_templates[step.name]

    def _process_response(self, response: Dict[str, Any], output_format: Dict[str, Any]) -> Dict[str, Any]:
        # Extract content from response and format according to output_format
        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        # Basic processing - you might want to enhance this based on your needs
        processed_results = {
            "themes": [line.strip() for line in content.split('\n') if line.strip()],
            "methodologies": [],
            "findings": []
        }
        
        return processed_results

    def _get_discipline_config(self, agent_name: str, discipline: str) -> AgentConfig:
        # Customize agent parameters based on discipline
        base_config = AGENT_CONFIGS[agent_name]
        if discipline == "computer_science":
            base_config.temperature *= 1.2
        elif discipline == "biology":
            base_config.max_tokens *= 1.5
        return base_config

class ResearchGapUI:
    def __init__(self):
        self.analyzer = ResearchGapAnalyzer()

    def render(self):
        st.title("Research Gap Analyzer")
        
        with st.sidebar:
            discipline = st.selectbox(
                "Select Research Discipline",
                ["computer_science", "biology", "physics", "social_science"]
            )
            
        research_area = st.text_area(
            "Enter your research area of interest:",
            help="Describe the specific field or topic you want to analyze"
        )
        
        if st.button("Analyze Gaps"):
            self._run_analysis(research_area, discipline)

    def _run_analysis(self, research_area: str, discipline: str):
        with st.spinner("Analyzing research gaps..."):
            results = self.analyzer.execute_workflow(research_area, discipline)
            self._display_results(results)

    def _display_results(self, results: Dict[str, Any]):
        # Create tabs for each analysis step
        tabs = st.tabs([step.name for step in RESEARCH_WORKFLOW])
        
        for tab, step in zip(tabs, RESEARCH_WORKFLOW):
            with tab:
                step_results = results.get(step.name, {})
                if "error" in step_results:
                    st.error(f"Error in {step.name}: {step_results['error']}")
                    continue
                
                self._render_step_results(step, step_results)

    def _render_step_results(self, step: ResearchStep, results: Dict[str, Any]):
        if step.name == "Literature Collection":
            self._render_literature_collection(results)
        elif step.name == "Content Analysis":
            self._render_content_analysis(results)
        elif step.name == "Gap Identification":
            self._render_gap_identification(results)
        elif step.name == "Interdisciplinary Connections":
            self._render_interdisciplinary_connections(results)

    def _render_literature_collection(self, results):
        st.subheader("Search Strategy")
        st.write(results["search_strategy"])
        
        st.subheader("Collected Papers")
        df = pd.DataFrame(results["collected_papers"])
        st.dataframe(df)
        
        with st.expander("Search Metrics"):
            for metric, value in results["search_metrics"].items():
                st.metric(metric, value)

    def _render_content_analysis(self, results):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Major Themes")
            for theme in results["themes"]:
                st.write(f"• {theme}")
                
        with col2:
            st.subheader("Methodologies")
            for method in results["methodologies"]:
                st.write(f"• {method}")
        
        st.subheader("Key Findings")
        for finding in results["findings"]:
            with st.expander(finding["title"]):
                st.write(finding["description"])
                st.markdown(f"*Source: {finding['citation']}*")

    def _render_gap_identification(self, results):
        st.subheader("Research Gaps")
        
        for gap in results["research_gaps"]:
            with st.container():
                st.markdown(f"### {gap['title']}")
                st.write(gap['description'])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Impact", gap['impact_score'])
                col2.metric("Feasibility", gap['feasibility'])
                col3.metric("Novelty", gap['novelty_score'])

    def _render_interdisciplinary_connections(self, results):
        st.subheader("Cross-disciplinary Opportunities")
        
        for conn in results["connections"]:
            with st.expander(conn["field"]):
                st.write(conn["description"])
                st.markdown("**Potential Approaches:**")
                for approach in conn["approaches"]:
                    st.write(f"• {approach}")

def main():
    ui = ResearchGapUI()
    ui.render()

if __name__ == "__main__":
    main()