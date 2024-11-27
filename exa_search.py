from typing import List, Dict
import re
from exa_py import Exa
from dataclasses import dataclass

@dataclass
class ResearchPaper:
    title: str
    url: str
    author: str
    published_date: str
    highlights: List[str]
    highlight_scores: List[float]
    score: float
    summary: str
    paper_id: str = None

class ExaSearch:
    def __init__(self, api_key: str):
        self.exa = Exa(api_key)
    
    def format_date(self, date_str: str) -> str:
        return date_str if date_str else "Unknown"
    
    def search_papers(self, query: str) -> List[ResearchPaper]:
        results = self.exa.search_and_contents(
            query,
            type="keyword",
            category="research paper",
            include_domains=[
                "nature.com", "science.org", "cell.com",
                "sciencedirect.com", "arxiv.org", "pnas.org"
            ],
            num_results=10,
            highlights=True,
            summary=True
        )
        
        papers = []
        for r in results.results:
            paper = ResearchPaper(
                title=getattr(r, 'title', ''),
                url=getattr(r, 'url', ''),
                author=getattr(r, 'author', 'Unknown'),
                published_date=self.format_date(getattr(r, 'published_date', None)),
                highlights=getattr(r, 'highlights', []),
                highlight_scores=getattr(r, 'highlight_scores', []),
                score=getattr(r, 'score', None),
                summary=getattr(r, 'summary', ''),
                paper_id=self.extract_paper_id(r)
            )
            if paper.paper_id:
                papers.append(paper)
        return papers[:5]
    
    def extract_paper_id(self, result) -> str:
        url = getattr(result, 'url', '') or ''
        text = getattr(result, 'text', '') or ''
        
        if not url and not text:
            return None
            
        doi_pattern = r'(?:DOI:|doi:)?\s*(10\.\d{4,}/[-._;()/:\w]+)'
        arxiv_pattern = r'(?:arXiv:|arxiv:)?\s*(\d{4}\.\d{4,})'
        
        doi_match = re.search(doi_pattern, url) or re.search(doi_pattern, text)
        arxiv_match = re.search(arxiv_pattern, url) or re.search(arxiv_pattern, text)
        
        if doi_match:
            return f"DOI:{doi_match.group(1)}"
        elif arxiv_match:
            return f"arXiv:{arxiv_match.group(1)}"
        return None