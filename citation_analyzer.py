import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict, Any, Callable
import requests
from datetime import datetime
import time
import logging
from functools import wraps
import random

class RateLimitExceeded(Exception):
    pass

class CitationAnalyzer:
    def __init__(self):
        self.colors = {
            'node': '#2B3467',
            'highlight': '#EB455F',
            'edge': '#BAD7E9',
            'citation': '#7aa2f7',
            'reference': '#9ece6a'
        }
        self.cache = {}
        self.logger = logging.getLogger('CitationAnalyzer')
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum time between requests in seconds

    def rate_limit_with_retry(max_retries: int = 3, base_delay: float = 1.0) -> Callable:
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                retries = 0
                while retries <= max_retries:
                    try:
                        # Implement request throttling
                        current_time = time.time()
                        time_since_last_request = current_time - self.last_request_time
                        if time_since_last_request < self.min_request_interval:
                            sleep_time = self.min_request_interval - time_since_last_request
                            time.sleep(sleep_time)
                        
                        result = func(self, *args, **kwargs)
                        self.last_request_time = time.time()
                        return result
                    
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 429:  # Rate limit exceeded
                            if retries == max_retries:
                                raise RateLimitExceeded("Rate limit exceeded. Please try again later.")
                            
                            # Calculate delay with exponential backoff and jitter
                            delay = base_delay * (2 ** retries) + random.uniform(0, 0.1)
                            self.logger.warning(f"Rate limit hit, retrying in {delay:.2f} seconds...")
                            time.sleep(delay)
                            retries += 1
                            continue
                        raise
                    except Exception as e:
                        self.logger.error(f"Error in {func.__name__}: {str(e)}")
                        raise
            return wrapper
        return decorator

    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate a cache key from the endpoint and parameters"""
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{endpoint}?{param_str}"

    def _get_cached_response(self, cache_key: str) -> Any:
        """Get response from cache if it exists and is fresh (less than 1 hour old)"""
        if cache_key in self.cache:
            timestamp, data = self.cache[cache_key]
            if time.time() - timestamp < 3600:  # Cache valid for 1 hour
                return data
        return None

    def _cache_response(self, cache_key: str, response: Any) -> None:
        """Cache the response with current timestamp"""
        self.cache[cache_key] = (time.time(), response)

    @rate_limit_with_retry(max_retries=3, base_delay=1.0)
    def get_paper(self, paper_id: str) -> Dict:
        """Get paper details with enhanced error handling and validation"""
        if not paper_id:
            raise ValueError("Paper ID cannot be empty")
            
        if not isinstance(paper_id, str):
            paper_id = str(paper_id)
            
        params = {'fields': 'title,authors,year,citationCount,referenceCount'}
        cache_key = self._get_cache_key(f'paper/{paper_id}', params)
        
        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        try:
            response = requests.get(
                f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}',
                params=params,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            if not result:
                raise ValueError(f"No data found for paper ID: {paper_id}")
                
            # Validate and provide defaults for essential fields
            result = {
                'title': result.get('title', 'Unknown Title'),
                'authors': result.get('authors', []),
                'year': result.get('year', datetime.now().year),
                'citationCount': result.get('citationCount', 0),
                'referenceCount': result.get('referenceCount', 0)
            }
            
            # Cache the response
            self._cache_response(cache_key, result)
            return result
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout while fetching paper {paper_id}")
            raise TimeoutError(f"Request timeout for paper {paper_id}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching paper {paper_id}: {str(e)}")
            raise

    @rate_limit_with_retry(max_retries=3, base_delay=1.0)
    def get_citations(self, paper_id: str) -> List[Dict]:
        params = {'fields': 'title,authors,year,citationCount', 'limit': 50}
        cache_key = self._get_cache_key(f'paper/{paper_id}/citations', params)
        
        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        try:
            response = requests.get(
                f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations',
                params=params,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()['data']
            
            # Cache the response
            self._cache_response(cache_key, result)
            return result
        except Exception as e:
            self.logger.error(f"Error fetching citations for paper {paper_id}: {str(e)}")
            return []

    @rate_limit_with_retry(max_retries=3, base_delay=1.0)
    def get_references(self, paper_id: str) -> List[Dict]:
        params = {'fields': 'title,authors,year,citationCount', 'limit': 50}
        cache_key = self._get_cache_key(f'paper/{paper_id}/references', params)
        
        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        try:
            response = requests.get(
                f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references',
                params=params,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()['data']
            
            # Cache the response
            self._cache_response(cache_key, result)
            return result
        except Exception as e:
            self.logger.error(f"Error fetching references for paper {paper_id}: {str(e)}")
            return []

    def validate_paper_data(self, paper_data: Dict) -> Dict:
        """Validate and clean paper data, providing fallback values for missing fields"""
        if not isinstance(paper_data, dict):
            return {
                'title': str(paper_data) if paper_data else 'Unknown Title',
                'year': datetime.now().year,
                'citationCount': 0
            }
        
        return {
            'title': paper_data.get('title', 'Unknown Title'),
            'year': paper_data.get('year', datetime.now().year),
            'citationCount': paper_data.get('citationCount', 0)
        }

    def create_citation_graph(self, paper_id: str, citations: List[Dict], references: List[Dict]) -> go.Figure:
        """Create citation network visualization with enhanced data validation"""
        try:
            self.logger.debug(f"Creating citation graph for paper_id: {paper_id}")
            self.logger.debug(f"Input data types - citations: {type(citations)}, references: {type(references)}")
            self.logger.debug(f"Number of citations: {len(citations) if isinstance(citations, list) else 'N/A'}")
            self.logger.debug(f"Number of references: {len(references) if isinstance(references, list) else 'N/A'}")
            
            # Validate input types
            if not isinstance(citations, list) or not isinstance(references, list):
                self.logger.error(f"Invalid input types - citations: {type(citations)}, references: {type(references)}")
                raise ValueError("Invalid input: citations and references must be lists")
            
            # Validate input data structure
            for i, citation in enumerate(citations):
                if not isinstance(citation, dict):
                    self.logger.warning(f"Invalid citation data at index {i}: {type(citation)}")
                    
            for i, reference in enumerate(references):
                if not isinstance(reference, dict):
                    self.logger.warning(f"Invalid reference data at index {i}: {type(reference)}")
            
            G = nx.Graph()
            current_year = datetime.now().year
            self.logger.debug(f"Initialized empty graph, current_year: {current_year}")
            
            # Add main paper node with validation
            try:
                main_node_attrs = {
                    "size": 50,
                    "color": self.colors['highlight'],
                    "year": current_year,
                    "citations": len(citations) if isinstance(citations, list) else 0,
                    "type": 'main'
                }
                G.add_node("Main Paper", **main_node_attrs)
                self.logger.debug(f"Added main paper node with attributes: {main_node_attrs}")
            except Exception as e:
                self.logger.error(f"Error adding main paper node: {str(e)}")
                raise

            # Add citations
            for cite in citations:
                if isinstance(cite, dict) and 'citingPaper' in cite:
                    paper_data = cite['citingPaper']
                else:
                    paper_data = cite

                year = paper_data.get('year', current_year) if isinstance(paper_data, dict) else current_year
                title = paper_data.get('title', 'Unknown Title') if isinstance(paper_data, dict) else str(paper_data)
                title = title[:40] + "..." if len(title) > 40 else title
                
                G.add_node(title, 
                          size=30,
                          color=self.colors['citation'],
                          year=year,
                          citations=paper_data.get('citationCount', 0) if isinstance(paper_data, dict) else 0,
                          type='citation')
                G.add_edge("Main Paper", title)

            # Add references
            for ref in references:
                if isinstance(ref, dict) and 'citedPaper' in ref:
                    paper_data = ref['citedPaper']
                else:
                    paper_data = ref

                year = paper_data.get('year', current_year) if isinstance(paper_data, dict) else current_year
                title = paper_data.get('title', 'Unknown Title') if isinstance(paper_data, dict) else str(paper_data)
                title = title[:40] + "..." if len(title) > 40 else title
                
                G.add_node(title, 
                          size=30,
                          color=self.colors['reference'],
                          year=year,
                          citations=paper_data.get('citationCount', 0) if isinstance(paper_data, dict) else 0,
                          type='reference')
                G.add_edge("Main Paper", title)

            pos = nx.spring_layout(G, k=1, iterations=50)

            # Create edges
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color=self.colors['edge']),
                hoverinfo='none',
                mode='lines'
            )

            # Create nodes
            node_x = []
            node_y = []
            node_sizes = []
            node_colors = []
            node_texts = []
            node_years = []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_data = G.nodes[node]
                node_sizes.append(node_data['size'])
                node_colors.append(node_data['color'])
                node_years.append(node_data['year'])
                
                text = f"Title: {node}<br>"
                text += f"Year: {node_data['year']}<br>"
                text += f"Citations: {node_data['citations']}<br>"
                text += f"Type: {node_data['type']}"
                node_texts.append(text)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_years,
                textposition="top center",
                hovertext=node_texts,
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(color='#ffffff', width=1)
                )
            )

            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text='Citation Network<br><sup>Size indicates citation count • Color indicates paper type • Numbers show publication year</sup>',
                        x=0.5,
                        y=0.95
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=80),
                    plot_bgcolor='#1a1b26',
                    paper_bgcolor='#1a1b26',
                    font=dict(color='#ffffff'),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    dragmode='pan',
                    modebar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        color='#ffffff',
                        activecolor='#EB455F'
                    )
                )
            )

            # Add buttons for interactivity
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        x=0.1,
                        y=1.1,
                        showactive=True,
                        buttons=[
                            dict(
                                label="Reset",
                                method="relayout",
                                args=[{"xaxis.range": None, "yaxis.range": None}]
                            ),
                            dict(
                                label="Zoom In",
                                method="relayout",
                                args=[{"xaxis.range": [-1, 1], "yaxis.range": [-1, 1]}]
                            ),
                            dict(
                                label="Zoom Out",
                                method="relayout",
                                args=[{"xaxis.range": [-2, 2], "yaxis.range": [-2, 2]}]
                            )
                        ],
                        font=dict(color='#ffffff'),
                        bgcolor='#2B3467',
                        bordercolor='#EB455F'
                    )
                ]
            )
            
            return fig
        except Exception as e:
            self.logger.error(f"Error creating citation graph: {str(e)}")
            raise
