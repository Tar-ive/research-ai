import streamlit as st
from exa_search import ExaSearch
from citation_analyzer import CitationAnalyzer
from moa import MOASystem
from config import EXA_API_KEY, GROQ_API_KEY
import logging
from typing import List, Dict
import json

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger('ResearchApp')

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
    st.set_page_config(page_title="Research Analysis Dashboard", page_icon="📚", layout="wide")
    st.title("📚 Research Analysis Dashboard")
    
    exa_search = ExaSearch(EXA_API_KEY)
    citation_analyzer = CitationAnalyzer()
    moa = MOASystem(GROQ_API_KEY)
    
    query = st.text_input("Enter research topic or question")
    
    if st.button("Analyze") and query:
        try:
            with st.spinner("🔍 Searching papers..."):
                papers = exa_search.search_papers(query)
                if not papers:
                    st.warning("No papers found. Try modifying your search query.")
                    return
                st.success(f"Found {len(papers)} papers")
            
            papers_tab, analysis_tab, debug_tab = st.tabs(["📄 Papers", "🔬 Analysis", "🐛 Debug"])
            
            with papers_tab:
                for i, paper in enumerate(papers, 1):
                    with st.expander(f"Paper {i}: {paper.title}", expanded=i==1):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown("### Summary")
                            st.write(paper.summary)
                            st.markdown("### Key Highlights")
                            for highlight, score in zip(paper.highlights, paper.highlight_scores):
                                st.write(f"- 💡 ({score:.2f}) {highlight}")
                        
                        if paper.paper_id:
                            with col2:
                                with st.spinner("Analyzing citations..."):
                                    citations = citation_analyzer.get_citations(paper.paper_id)
                                    references = citation_analyzer.get_references(paper.paper_id)
                                    
                                    st.markdown("### Citation Metrics")
                                    m1, m2 = st.columns(2)
                                    m1.metric("Citations", len(citations))
                                    m2.metric("References", len(references))
                                    
                                    st.markdown("### Citation Graph")
                                    citation_graph = citation_analyzer.create_citation_graph(
                                        paper.paper_id, citations, references
                                    )
                                    st.plotly_chart(citation_graph, use_container_width=True)
            
            with analysis_tab:
                with st.spinner("Analyzing research landscape..."):
                    moa_papers = convert_exa_to_moa_format(papers)
                    analysis = moa.analyze_research_landscape(moa_papers, query)
                    
                    st.markdown("### Research Gaps Analysis")
                    if analysis.get('gaps'):
                        gaps_table = moa.create_research_gaps_table(analysis['gaps'])
                        st.plotly_chart(gaps_table, use_container_width=True)
                    else:
                        st.info("No research gaps identified.")
                    
                    st.markdown("### Research Hypotheses")
                    if analysis.get('hypotheses'):
                        hypotheses_table = moa.create_hypothesis_table(analysis['hypotheses'])
                        st.plotly_chart(hypotheses_table, use_container_width=True)
                    else:
                        st.info("No hypotheses generated.")
                    
                    if analysis.get('limitations'):
                        st.markdown("### Current Limitations")
                        for limitation in analysis['limitations']:
                            st.write(f"- {limitation}")
            
            with debug_tab:
                st.markdown("### Raw MOA Responses")
                st.json(analysis)

                st.markdown("### Paper Data")
                st.json(moa_papers)
                
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            st.error(f"Analysis error: {str(e)}")

if __name__ == "__main__":
    main()