import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict
import requests
from datetime import datetime

class CitationAnalyzer:
    def __init__(self):
        self.colors = {
            'node': '#2B3467',
            'highlight': '#EB455F',
            'edge': '#BAD7E9',
            'citation': '#7aa2f7',
            'reference': '#9ece6a'
        }
    
    def get_paper(self, paper_id: str) -> Dict:
        response = requests.get(
            f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}',
            params={'fields': 'title,authors,year,citationCount,referenceCount'}
        )
        response.raise_for_status()
        return response.json()
        
    def get_citations(self, paper_id: str) -> List[Dict]:
        response = requests.get(
            f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations',
            params={'fields': 'title,authors,year,citationCount', 'limit': 50}
        )
        response.raise_for_status()
        return response.json()['data']
        
    def get_references(self, paper_id: str) -> List[Dict]:
        response = requests.get(
            f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references',
            params={'fields': 'title,authors,year,citationCount', 'limit': 50}
        )
        response.raise_for_status()
        return response.json()['data']

    def create_citation_graph(self, paper_id: str, citations: List[Dict], references: List[Dict]) -> go.Figure:
        G = nx.Graph()
        current_year = datetime.now().year
        
        # Add main paper node
        G.add_node("Main Paper", 
                  size=50,
                  color=self.colors['highlight'],
                  year=current_year,
                  citations=len(citations),
                  type='main')

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