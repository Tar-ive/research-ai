import streamlit as st
from exa_search import ExaSearch
from citation_analyzer import CitationAnalyzer
from moa import MOASystem
from data_analyzer import DataAnalyzer
from hypothesis_validator import HypothesisValidator
from config import EXA_API_KEY, GROQ_API_KEY
import logging
from typing import List, Dict
import json
import numpy as np
import pandas as pd

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger('ResearchApp')

def initialize_session_state():
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'papers' not in st.session_state:
        st.session_state.papers = None
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'test_results' not in st.session_state:
        st.session_state.test_results = {}
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = {}

def convert_exa_to_moa_format(papers: List[Dict]) -> List[Dict]:
    return [{
        'title': p.title,
        'highlights': p.highlights,
        'summary': p.summary,
        'citations': len(getattr(p, 'citations', [])),
        'paper_id': p.paper_id
    } for p in papers]

def main():
    logger = setup_logging()
    st.set_page_config(page_title="Research Analysis Dashboard", layout="wide")
    initialize_session_state()

    # Initialize components
    exa_search = ExaSearch(EXA_API_KEY)
    citation_analyzer = CitationAnalyzer()
    moa = MOASystem(GROQ_API_KEY)
    data_analyzer = DataAnalyzer()
    hypothesis_validator = HypothesisValidator()

    # Header
    st.title("📚 Research Analysis Dashboard")
    
    # Query Input
    with st.container():
        query = st.text_input("Enter research topic or question", key="query_input")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            analyze_button = st.button("🔍 Analyze Research", use_container_width=True)

    # Main Analysis Process
    if analyze_button and query:
        try:
            with st.spinner("Analyzing research topic..."):
                # Search papers
                papers = exa_search.search_papers(query)
                if not papers:
                    st.warning("No papers found. Please try a different query.")
                    return
                
                # Convert and analyze
                moa_papers = convert_exa_to_moa_format(papers)
                analysis = moa.analyze_research_landscape(moa_papers, query)
                
                # Store in session state
                st.session_state.papers = papers
                st.session_state.analysis = analysis
                st.session_state.analysis_complete = True
                
                st.success("Analysis complete! View results in the tabs below.")
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            st.error(f"Error during analysis: {str(e)}")
            return

    # Only show tabs after analysis is complete
    if st.session_state.analysis_complete:
        papers_tab, analysis_tab, stats_tab, validation_tab, debug_tab = st.tabs([
            "📄 Papers", "🔬 Analysis", "📊 Statistics", "✅ Validation", "🐛 Debug"
        ])

        # Papers Tab
        with papers_tab:
            for i, paper in enumerate(st.session_state.papers, 1):
                with st.expander(f"Paper {i}: {paper.title}", expanded=(i == 1)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("### Summary")
                        st.write(paper.summary)
                        
                        st.markdown("### Key Highlights")
                        for highlight, score in zip(paper.highlights, paper.highlight_scores):
                            st.write(f"- 💡 ({score:.2f}) {highlight}")
                    
                    with col2:
                        if paper.paper_id:
                            st.markdown("### Citation Analysis")
                            if st.button("Analyze Citations", key=f"cite_{i}"):
                                try:
                                    with st.spinner("Analyzing citations..."):
                                        citations = citation_analyzer.get_citations(paper.paper_id)
                                        references = citation_analyzer.get_references(paper.paper_id)
                                        
                                        # Display metrics
                                        m1, m2 = st.columns(2)
                                        m1.metric("Citations", len(citations))
                                        m2.metric("References", len(references))
                                        
                                        # Create and display graph
                                        graph = citation_analyzer.create_citation_graph(
                                            paper.paper_id, citations, references)
                                        st.plotly_chart(graph, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error analyzing citations: {str(e)}")

        # Analysis Tab
        with analysis_tab:
            st.markdown("### Research Analysis Results")
            
            # Research Gaps
            st.markdown("#### Research Gaps")
            gaps = st.session_state.analysis.get('gaps', [])
            if gaps:
                gaps_table = moa.create_research_gaps_table(gaps)
                st.plotly_chart(gaps_table, use_container_width=True)
            else:
                st.info("No research gaps identified.")

            # Hypotheses
            st.markdown("#### Research Hypotheses")
            hypotheses = st.session_state.analysis.get('hypotheses', [])
            if hypotheses:
                hypotheses_table = moa.create_hypothesis_table(hypotheses)
                st.plotly_chart(hypotheses_table, use_container_width=True)
            else:
                st.info("No hypotheses generated.")

        # Statistics Tab
        with stats_tab:
            st.markdown("### Statistical Analysis")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                test_type = st.selectbox(
                    "Select Test Type",
                    options=['t_test', 'correlation'],
                    key='stat_test_type'
                )
                
                sample_size = st.number_input(
                    "Sample Size",
                    min_value=10,
                    max_value=1000,
                    value=100
                )

                if test_type == 't_test':
                    st.markdown("#### Group 1")
                    g1_mean = st.number_input("Mean", value=0.0, key='g1_mean')
                    g1_std = st.number_input("Std Dev", min_value=0.1, value=1.0, key='g1_std')
                    
                    st.markdown("#### Group 2")
                    g2_mean = st.number_input("Mean", value=2.0, key='g2_mean')
                    g2_std = st.number_input("Std Dev", min_value=0.1, value=1.0, key='g2_std')
                    
                    if st.button("Run T-Test", use_container_width=True):
                        try:
                            with st.spinner("Running t-test..."):
                                group1 = data_analyzer.generate_synthetic_data(g1_mean, g1_std, sample_size)
                                group2 = data_analyzer.generate_synthetic_data(g2_mean, g2_std, sample_size)
                                result = data_analyzer.perform_t_test(group1, group2)
                                st.session_state.current_result = result
                        except Exception as e:
                            st.error(f"Error in t-test: {str(e)}")
                else:
                    st.markdown("#### Correlation Parameters")
                    var1_mean = st.number_input("Variable 1 Mean", value=0.0, key='var1_mean')
                    var1_std = st.number_input("Variable 1 Std Dev", min_value=0.1, value=1.0, key='var1_std')
                    var2_mean = st.number_input("Variable 2 Mean", value=2.0, key='var2_mean')
                    var2_std = st.number_input("Variable 2 Std Dev", min_value=0.1, value=1.0, key='var2_std')
                    
                    if st.button("Run Correlation", use_container_width=True):
                        try:
                            with st.spinner("Running correlation analysis..."):
                                var1 = data_analyzer.generate_synthetic_data(var1_mean, var1_std, sample_size)
                                var2 = data_analyzer.generate_synthetic_data(var2_mean, var2_std, sample_size)
                                result = data_analyzer.perform_correlation(var1, var2)
                                st.session_state.current_result = result
                        except Exception as e:
                            st.error(f"Error in correlation: {str(e)}")
            
            with col2:
                if 'current_result' in st.session_state:
                    result = st.session_state.current_result
                    
                    # Display visualization
                    if hasattr(result, 'visualization'):
                        st.plotly_chart(result.visualization, use_container_width=True)
                    
                    # Display statistics
                    st.markdown("#### Results")
                    stats_df = pd.DataFrame({
                        'Metric': ['Test Statistic', 'P-value', 'Effect Size'],
                        'Value': [
                            f"{result.statistic:.3f}",
                            f"{result.p_value:.3f}",
                            f"{result.effect_size:.3f}"
                        ]
                    })
                    st.table(stats_df)
                    
                    if hasattr(result, 'interpretation'):
                        st.markdown("#### Interpretation")
                        st.markdown(result.interpretation)
                else:
                    st.info("Run a statistical test to see results")

        # Validation Tab
        with validation_tab:
            st.markdown("### Hypothesis Validation")
            
            hypotheses = st.session_state.analysis.get('hypotheses', [])
            if not hypotheses:
                st.info("No hypotheses available for validation.")
            else:
                for i, hyp in enumerate(hypotheses):
                    with st.expander(f"Hypothesis {i+1}: {hyp['statement']}", expanded=(i == 0)):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown("#### Validation Configuration")
                            test_type = st.selectbox(
                                "Test Type",
                                ["t_test", "correlation"],
                                key=f"val_test_{i}"
                            )
                            
                            data_params = {
                                'group1': {'mean': 0, 'std': 1, 'size': 100},
                                'group2': {'mean': 2, 'std': 1, 'size': 100}
                            }
                            
                            if st.button("Validate", key=f"val_btn_{i}"):
                                try:
                                    with st.spinner("Validating hypothesis..."):
                                        config = {
                                            'statement': hyp['statement'],
                                            'data_params': data_params,
                                            'required_tests': [test_type]
                                        }
                                        result = hypothesis_validator.validate_hypothesis(config)
                                        st.session_state.validation_results[i] = result
                                except Exception as e:
                                    st.error(f"Validation error: {str(e)}")
                        
                        with col2:
                            if i in st.session_state.validation_results:
                                result = st.session_state.validation_results[i]
                                st.markdown(f"**Status:** {result.validation_status}")
                                st.markdown(f"**Confidence:** {result.confidence:.2%}")
                                
                                if result.visualization:
                                    st.plotly_chart(result.visualization, use_container_width=True)
                                
                                st.markdown("#### Recommendations")
                                for rec in result.recommendations:
                                    st.markdown(f"- {rec}")
                            else:
                                st.info("Run validation to see results")

        # Debug Tab
        with debug_tab:
            st.markdown("### Debug Information")
            
            # Session State
            st.markdown("#### Session State")
            st.json({k: str(v) for k, v in st.session_state.items()})
            
            # Analysis Results
            if st.session_state.analysis:
                st.markdown("#### Analysis Results")
                st.json(st.session_state.analysis)

if __name__ == "__main__":
    main()