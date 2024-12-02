import logging
from typing import Dict, List
import json
from dataclasses import dataclass
import plotly.graph_objects as go

class MOASystem:
    def __init__(self, api_key: str):
        self.logger = logging.getLogger('MOASystem')
        self.api_key = api_key
        self.agents = {
            "analysis_agent": {
                "model_name": "mixtral-8x7b-32768",
                "temperature": 0.7,
                "system_prompt": "You are a research analysis assistant. Analyze papers and identify gaps, hypotheses, and patterns."
            }
        }

    def groq_client_generate_response(self, prompt: str, model: str, temperature: float, system_prompt: str) -> str:
        # Placeholder for actual Groq API implementation
        # This would typically make an API call to Groq
        return json.dumps({
            "gaps": [
                {
                    "topic": "Data collection methodology",
                    "description": "Current research lacks standardized data collection methods",
                    "impact": "High"
                }
            ],
            "hypotheses": [
                {
                    "statement": "Improved data collection methods lead to more accurate results",
                    "confidence": 0.85,
                    "supporting_evidence": ["Previous studies show correlation between methodology and accuracy"]
                }
            ]
        })

    def _create_analysis_prompt(self, papers: List[Dict], query: str) -> str:
        paper_summaries = "\n".join([
            f"Title: {p['title']}\nSummary: {p['summary']}\n"
            for p in papers
        ])
        return f"""
        Research Query: {query}
        
        Papers to analyze:
        {paper_summaries}
        
        Please analyze these papers and identify:
        1. Research gaps
        2. Potential hypotheses
        3. Citation patterns
        4. Limitations in current research
        """

    def validate_paper_data(self, paper: Dict) -> Dict:
        """Validate and clean paper data"""
        if not isinstance(paper, dict):
            self.logger.warning(f"Invalid paper data format: {type(paper)}")
            return {}
            
        required_fields = ['title', 'summary']
        for field in required_fields:
            if field not in paper:
                self.logger.warning(f"Missing required field: {field}")
                return {}
                
        return paper

    def _parse_analysis_response(self, response: str) -> Dict:
        """Parse analysis response with enhanced error handling and validation"""
        self.logger.debug(f"Raw API response received: {response[:1000]}...")  # Log first 1000 chars
        
        try:
            # Validate input type
            if not isinstance(response, str):
                self.logger.error(f"Invalid response type: {type(response)}")
                raise ValueError(f"Expected string response, got {type(response)}")

            # Parse JSON with detailed error handling
            try:
                data = json.loads(response)
                self.logger.debug(f"Parsed JSON data structure: {json.dumps(data, indent=2)}")
            except json.JSONDecodeError as je:
                self.logger.error(f"JSON parsing error at position {je.pos}: {je.msg}")
                self.logger.debug(f"JSON context: {response[max(0, je.pos-50):je.pos+50]}")
                raise
            
            # Validate expected data structure
            if not isinstance(data, dict):
                self.logger.error(f"Invalid data structure type: {type(data)}")
                raise ValueError(f"Invalid response format: expected dictionary, got {type(data)}")
                
            # Log data structure before processing
            self.logger.debug("Data structure validation:")
            self.logger.debug(f"Keys present: {list(data.keys())}")
            self.logger.debug(f"Value types: {[(k, type(v)) for k, v in data.items()]}")
                
            # Provide default values for missing sections
            result = {
                "gaps": [],
                "hypotheses": [],
                "citation_patterns": {},
                "limitations": [],
                "statistical_requirements": {}
            }
            
            # Validate and merge each section with detailed logging
            if "gaps" in data:
                if not isinstance(data["gaps"], list):
                    self.logger.error(f"Invalid gaps type: {type(data['gaps'])}")
                    raise ValueError(f"Expected list for gaps, got {type(data['gaps'])}")
                
                self.logger.debug(f"Processing {len(data['gaps'])} gaps")
                result["gaps"] = []
                for i, gap in enumerate(data["gaps"]):
                    try:
                        validated_gap = {
                            "topic": gap.get("topic", "Unknown Topic"),
                            "description": gap.get("description", "No description available"),
                            "impact": gap.get("impact", "Unknown")
                        }
                        result["gaps"].append(validated_gap)
                        self.logger.debug(f"Processed gap {i+1}: {validated_gap}")
                    except Exception as e:
                        self.logger.error(f"Error processing gap {i+1}: {str(e)}")
                        self.logger.debug(f"Problematic gap data: {gap}")
                
            if "hypotheses" in data:
                if not isinstance(data["hypotheses"], list):
                    self.logger.error(f"Invalid hypotheses type: {type(data['hypotheses'])}")
                    raise ValueError(f"Expected list for hypotheses, got {type(data['hypotheses'])}")
                
                self.logger.debug(f"Processing {len(data['hypotheses'])} hypotheses")
                result["hypotheses"] = []
                for i, hyp in enumerate(data["hypotheses"]):
                    try:
                        confidence = hyp.get("confidence", 0.0)
                        if not isinstance(confidence, (int, float)):
                            self.logger.warning(f"Invalid confidence value: {confidence}, using 0.0")
                            confidence = 0.0
                            
                        validated_hyp = {
                            "statement": hyp.get("statement", "Unknown hypothesis"),
                            "confidence": float(confidence),
                            "supporting_evidence": hyp.get("supporting_evidence", [])
                        }
                        result["hypotheses"].append(validated_hyp)
                        self.logger.debug(f"Processed hypothesis {i+1}: {validated_hyp}")
                    except Exception as e:
                        self.logger.error(f"Error processing hypothesis {i+1}: {str(e)}")
                        self.logger.debug(f"Problematic hypothesis data: {hyp}")
                
            self.logger.debug("Final processed result structure:")
            self.logger.debug(f"Result keys: {list(result.keys())}")
            self.logger.debug(f"Number of gaps: {len(result['gaps'])}")
            self.logger.debug(f"Number of hypotheses: {len(result['hypotheses'])}")
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse analysis response: {str(e)}")
            return {
                "gaps": [],
                "hypotheses": [],
                "citation_patterns": {},
                "limitations": [],
                "error": "Failed to parse analysis response"
            }
        except Exception as e:
            self.logger.error(f"Unexpected error parsing response: {str(e)}")
            return {
                "gaps": [],
                "hypotheses": [],
                "citation_patterns": {},
                "limitations": [],
                "error": f"Unexpected error: {str(e)}"
            }

    def analyze_research_landscape(self, papers: List[Dict], query: str) -> Dict:
        try:
            analysis_prompt = self._create_analysis_prompt(papers, query)
            analysis_response = self.groq_client_generate_response(
                analysis_prompt,
                self.agents["analysis_agent"]["model_name"],
                self.agents["analysis_agent"]["temperature"],
                self.agents["analysis_agent"]["system_prompt"]
            )
            analysis_results = self._parse_analysis_response(analysis_response)
            
            analysis_results['statistical_requirements'] = self._generate_statistical_requirements(
                analysis_results.get('gaps', []),
                analysis_results.get('hypotheses', [])
            )
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Analysis error: {str(e)}")
            return {
                "gaps": [],
                "limitations": [],
                "hypotheses": [],
                "citation_patterns": {},
                "statistical_requirements": {}
            }

    def _generate_statistical_requirements(self, gaps: List[Dict], hypotheses: List[Dict]) -> Dict:
        requirements = {
            'gaps': {},
            'hypotheses': {}
        }
        
        for gap in gaps:
            requirements['gaps'][gap['topic']] = {
                'suggested_tests': self._determine_required_tests(gap),
                'data_requirements': self._determine_data_requirements(gap)
            }
        
        for hypothesis in hypotheses:
            requirements['hypotheses'][hypothesis['statement']] = {
                'suggested_tests': self._determine_required_tests(hypothesis),
                'data_requirements': self._determine_data_requirements(hypothesis)
            }
        
        return requirements

    def _determine_required_tests(self, item: Dict) -> List[str]:
        tests = []
        description = item.get('topic', '') or item.get('statement', '')
        
        if 'compare' in description.lower() or 'difference' in description.lower():
            tests.append('t_test')
        if 'relationship' in description.lower() or 'correlation' in description.lower():
            tests.append('correlation')
        if 'predict' in description.lower() or 'impact' in description.lower():
            tests.append('regression')
        if 'groups' in description.lower() or 'multiple' in description.lower():
            tests.append('anova')
        
        return tests or ['t_test', 'correlation']

    def _determine_data_requirements(self, item: Dict) -> Dict:
        return {
            'sample_size': self._estimate_sample_size(item),
            'variables': self._identify_variables(item),
            'data_type': self._determine_data_type(item)
        }

    def _estimate_sample_size(self, item: Dict) -> int:
        return 100

    def _identify_variables(self, item: Dict) -> Dict:
        return {
            'dependent': [],
            'independent': [],
            'covariates': []
        }

    def _determine_data_type(self, item: Dict) -> str:
        return 'continuous'

    def create_research_gaps_table(self, gaps: List[Dict]) -> go.Figure:
        headers = ['Topic', 'Description', 'Impact']
        rows = [[
            gap.get('topic', ''),
            gap.get('description', ''),
            gap.get('impact', '')
        ] for gap in gaps]
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color='#1a1b26',
                align='left',
                font=dict(color='white')
            ),
            cells=dict(
                values=list(zip(*rows)),
                fill_color='#24283b',
                align='left',
                font=dict(color='white')
            )
        )])
        
        fig.update_layout(
            title='Research Gaps Analysis',
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='#1a1b26'
        )
        
        return fig

    def create_hypothesis_table(self, hypotheses: List[Dict]) -> go.Figure:
        headers = ['Hypothesis', 'Confidence', 'Supporting Evidence']
        rows = [[
            h.get('statement', ''),
            f"{h.get('confidence', 0):.2%}",
            "\n".join(h.get('supporting_evidence', []))
        ] for h in hypotheses]
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color='#1a1b26',
                align='left',
                font=dict(color='white')
            ),
            cells=dict(
                values=list(zip(*rows)),
                fill_color='#24283b',
                align='left',
                font=dict(color='white')
            )
        )])
        
        fig.update_layout(
            title='Research Hypotheses',
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='#1a1b26'
        )
        
        return fig
