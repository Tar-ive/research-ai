import os
from dataclasses import dataclass
from typing import List, Dict, Any
import logging
import json
import plotly.graph_objects as go
from groq import Groq

@dataclass
class ResearchGap:
    topic: str
    evidence: List[str]
    confidence: float
    potential_impact: str
    suggested_approach: str

@dataclass
class Hypothesis:
    statement: str
    evidence: List[str]
    test_methodology: str
    expected_outcome: str
    requirements: List[str]
    confidence: float

class GroqClient:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        
    def generate_response(self, prompt: str, model: str, temperature: float, system_prompt: str = "You are a helpful assistant.") -> Dict[str, Any]:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=temperature,
                max_tokens=2048,
                top_p=1,
                stream=False
            )
            
            if not chat_completion.choices or not chat_completion.choices[0].message.content:
                raise ValueError("Empty response from Groq API")
            
            content = chat_completion.choices[0].message.content
            logging.info(f"API Response: {content}")
            return {"choices": [{"text": content}]}
        except Exception as e:
            logging.error(f"Groq API error: {str(e)}")
            return {"choices": [{"text": "{}"}]}

class MOASystem:
    def __init__(self, groq_api_key: str):
        self.logger = logging.getLogger('MOASystem')
        self.groq_client = GroqClient(groq_api_key)
        self.agents = self._setup_agents()
        
    def _setup_agents(self) -> Dict[str, Any]:
        return {
            "analysis_agent": {
                "system_prompt": """You are an expert research analyst specialized in identifying research gaps and generating hypotheses. 
                Your goal is to analyze research papers and identify:
                1. Novel research opportunities
                2. Methodological limitations
                3. Unexplored areas
                4. Potential theoretical connections""",
                "model_name": "llama3-8b-8192",
                "temperature": 0.3
            },
            "hypothesis_agent": {
                "system_prompt": """You are a scientific hypothesis generator.
                Your task is to generate specific, testable hypotheses that:
                1. Address identified research gaps
                2. Build on existing methodologies
                3. Can be empirically validated
                4. Have clear success criteria""",
                "model_name": "llama3-8b-8192",
                "temperature": 0.4
            }
        }

    def _create_analysis_prompt(self, papers: List[Dict], query: str) -> str:
        paper_analysis = []
        
        for paper in papers:
            analysis = [
                f"\nTitle: {paper['title']}",
                f"Summary: {paper.get('summary', '')}",
                f"Key Findings: {'; '.join(paper.get('highlights', []))}",
                f"Citation Count: {paper.get('citations', 0)}",
                "References: " + (str(len(paper.get('references', []))) if paper.get('references') else "Unknown")
            ]
            paper_analysis.append("\n".join(analysis))

        prompt = f"""Analyze these papers on '{query}':

{"\n---\n".join(paper_analysis)}

Identify:

1. RESEARCH GAPS:
- What important questions remain unanswered?
- What methodological limitations exist?
- What theoretical connections are unexplored?
For each gap, provide:
- Specific evidence from papers
- Confidence level (0-1)
- Potential impact
- Suggested research approach

2. CURRENT LIMITATIONS:
- Technical constraints
- Methodological weaknesses
- Data limitations
- Theoretical boundaries

3. CITATION PATTERNS:
- Highly cited works
- Emerging trends
- Research clusters
- Methodology preferences

Format the response as a valid JSON object with the following structure:
{{
    "gaps": [
        {{
            "topic": "Describe the specific research gap",
            "evidence": ["Evidence 1", "Evidence 2"],
            "confidence": 0.8,
            "potential_impact": "Description of potential impact",
            "suggested_approach": "Detailed approach suggestion"
        }}
    ],
    "limitations": [
        "Limitation 1",
        "Limitation 2"
    ],
    "citation_patterns": {{
        "pattern_name": {{
            "citation_count": 5,
            "papers": ["Paper 1", "Paper 2"]
        }}
    }}
}}"""

        self.logger.info(f"Analysis Prompt:\n{prompt}")
        return prompt

    def _create_hypothesis_prompt(self, analysis: Dict, papers: List[Dict], query: str) -> str:
        prompt = f"""Based on the research analysis for '{query}':

ANALYSIS RESULTS:
{json.dumps(analysis, indent=2)}

PAPER SUMMARIES:
{json.dumps([{"title": p.get('title', ''), "summary": p.get('summary', '')} for p in papers], indent=2)}

Generate testable hypotheses that:
1. Address identified research gaps
2. Build on existing methodologies
3. Can be empirically validated
4. Have clear success criteria

For each hypothesis, provide:
1. Clear statement of the hypothesis
2. Supporting evidence from papers
3. Detailed test methodology
4. Expected outcomes
5. Required resources/conditions
6. Confidence score (0-1)

Format response as a JSON object with this structure:
{{
    "hypotheses": [
        {{
            "statement": "Specific testable hypothesis",
            "evidence": ["Supporting evidence 1", "Supporting evidence 2"],
            "test_methodology": "Detailed test methodology",
            "expected_outcome": "Expected results",
            "requirements": ["Requirement 1", "Requirement 2"],
            "confidence": 0.8
        }}
    ]
}}"""

        self.logger.info(f"Hypothesis Prompt:\n{prompt}")
        return prompt

    def analyze_research_landscape(self, papers: List[Dict], query: str) -> Dict:
        try:
            # Initial Analysis
            analysis_prompt = self._create_analysis_prompt(papers, query)
            analysis_response = self.groq_client.generate_response(
                analysis_prompt,
                self.agents["analysis_agent"]["model_name"],
                self.agents["analysis_agent"]["temperature"],
                self.agents["analysis_agent"]["system_prompt"]
            )
            analysis_results = self._parse_analysis_response(analysis_response)
            
            # Generate Hypotheses
            hypothesis_prompt = self._create_hypothesis_prompt(analysis_results, papers, query)
            hypothesis_response = self.groq_client.generate_response(
                hypothesis_prompt,
                self.agents["hypothesis_agent"]["model_name"],
                self.agents["hypothesis_agent"]["temperature"],
                self.agents["hypothesis_agent"]["system_prompt"]
            )
            hypotheses = self._parse_hypothesis_response(hypothesis_response)
            
            return {
                "gaps": analysis_results.get("gaps", []),
                "limitations": analysis_results.get("limitations", []),
                "hypotheses": hypotheses.get("hypotheses", []),
                "citation_patterns": analysis_results.get("citation_patterns", {})
            }
            
        except Exception as e:
            self.logger.error(f"Analysis error: {str(e)}")
            return {"gaps": [], "limitations": [], "hypotheses": [], "citation_patterns": {}}

    def _parse_analysis_response(self, response: Dict) -> Dict:
        try:
            content = response['choices'][0]['text']
            json_start = content.find('{')
            if json_start != -1:
                json_str = content[json_start:]
                return json.loads(json_str)
            return {"hypotheses": []}
        except Exception as e:
            self.logger.error(f"Error parsing hypothesis response: {str(e)}")
            return {"hypotheses": []}

    def _parse_hypothesis_response(self, response: Dict) -> Dict:
        try:
            content = response['choices'][0]['text']
            json_start = content.find('{')
            if json_start != -1:
                json_str = content[json_start:]
                return json.loads(json_str)
            return {"hypotheses": []}
        except Exception as e:
            self.logger.error(f"Error parsing hypothesis response: {str(e)}")
            return {"hypotheses": []}

    def create_research_gaps_table(self, gaps: List[Dict]) -> go.Figure:
        headers = ["Gap", "Evidence", "Confidence", "Impact", "Approach"]
        rows = []
        
        for gap in gaps:
            rows.append([
                gap["topic"],
                "<br>".join(gap["evidence"]),
                f"{gap.get('confidence', 0):.2f}",
                gap.get("potential_impact", ""),
                gap.get("suggested_approach", "")
            ])
            
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color='#1a1b26',
                font=dict(color='white'),
                align='left'
            ),
            cells=dict(
                values=list(zip(*rows)),
                fill_color='#24283b',
                font=dict(color='white'),
                align='left'
            )
        )])
        
        fig.update_layout(
            title='Research Gaps Analysis',
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='#1a1b26'
        )
        
        return fig

    def create_hypothesis_table(self, hypotheses: List[Dict]) -> go.Figure:
        headers = ["Hypothesis", "Evidence", "Test Method", "Expected Outcome", "Requirements", "Confidence"]
        rows = []
        
        for hyp in hypotheses:
            rows.append([
                hyp["statement"],
                "<br>".join(hyp["evidence"]),
                hyp["test_methodology"],
                hyp["expected_outcome"],
                "<br>".join(hyp["requirements"]),
                f"{hyp.get('confidence', 0):.2f}"
            ])
            
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color='#1a1b26',
                font=dict(color='white'),
                align='left'
            ),
            cells=dict(
                values=list(zip(*rows)),
                fill_color='#24283b',
                font=dict(color='white'),
                align='left'
            )
        )])
        
        fig.update_layout(
            title='Research Hypotheses Analysis',
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='#1a1b26'
        )
        
        return fig