from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
import plotly.graph_objects as go
import logging
from data_analyzer import DataAnalyzer, StatTestResult

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HypothesisValidator')

@dataclass
class ValidationResult:
    """Class to hold the results of hypothesis validation"""
    hypothesis: str
    test_results: List[StatTestResult]
    validation_status: str
    confidence: float
    recommendations: List[str]
    visualization: Optional[go.Figure] = None
    
    @property
    def statistic(self) -> float:
        """Return the test statistic from the first test result"""
        if self.test_results:
            return self.test_results[0].statistic
        return None
    
    @property
    def p_value(self) -> float:
        """Return the p-value from the first test result"""
        if self.test_results:
            return self.test_results[0].p_value
        return None
    
    @property
    def effect_size(self) -> float:
        """Return the effect size from the first test result"""
        if self.test_results:
            return self.test_results[0].effect_size
        return None

class DataDebugger:
    """Class for debugging data flow through the validation process"""
    
    @staticmethod
    def print_data_state(location: str, data: Any):
        """Print the current state of data at any location in the code"""
        logger.debug(f"\n{'='*50}")
        logger.debug(f"Location: {location}")
        
        try:
            if isinstance(data, dict):
                logger.debug("Data type: Dictionary")
                logger.debug("Contents:")
                for key, value in data.items():
                    logger.debug(f"  {key}: {type(value)} = {value}")
            elif isinstance(data, list):
                logger.debug("Data type: List")
                logger.debug(f"Length: {len(data)}")
                logger.debug("Contents:")
                for i, item in enumerate(data):
                    logger.debug(f"  [{i}]: {type(item)} = {item}")
            else:
                logger.debug(f"Data type: {type(data)}")
                logger.debug(f"Value: {data}")
        except Exception as e:
            logger.error(f"Error printing data: {str(e)}")
        
        logger.debug(f"{'='*50}\n")

class HypothesisValidator:
    """Class for validating hypotheses using statistical tests"""
    
    def __init__(self):
        """Initialize the validator with a data analyzer and debugger"""
        self.data_analyzer = DataAnalyzer()
        self.debugger = DataDebugger()
        
    def generate_synthetic_dataset(self, hypothesis_params: Dict) -> Dict[str, np.ndarray]:
        """Generate synthetic data based on hypothesis parameters"""
        self.debugger.print_data_state("generate_synthetic_dataset input", hypothesis_params)
        
        datasets = {}
        key_mappings = {
            'x': 'var1',
            'y': 'var2',
            'var1': 'var1',
            'var2': 'var2',
            'group1': 'group1',
            'group2': 'group2'
        }
        
        try:
            for group_name, params in hypothesis_params.items():
                data = self.data_analyzer.generate_synthetic_data(
                    mean=params.get('mean', 0),
                    std=params.get('std', 1),
                    size=params.get('size', 100),
                    distribution=params.get('distribution', 'normal')
                )
                # Use mapped key if it exists, otherwise use original
                mapped_name = key_mappings.get(group_name, group_name)
                datasets[mapped_name] = data
                
            # If old keys exist, duplicate them to new keys
            if 'x' in datasets and 'var1' not in datasets:
                datasets['var1'] = datasets['x']
            if 'y' in datasets and 'var2' not in datasets:
                datasets['var2'] = datasets['y']
                
            self.debugger.print_data_state("generated_datasets", datasets)
            return datasets
            
        except Exception as e:
            logger.error(f"Error generating synthetic dataset: {str(e)}")
            raise

    def validate_hypothesis(self, 
                          hypothesis: Dict,
                          confidence_threshold: float = 0.95) -> ValidationResult:
        """Validate a hypothesis using synthetic data and statistical tests"""
        self.debugger.print_data_state("validate_hypothesis input", {
            'hypothesis': hypothesis,
            'confidence_threshold': confidence_threshold
        })
        
        try:
            # Generate synthetic data based on hypothesis parameters
            datasets = self.generate_synthetic_dataset(hypothesis.get('data_params', {}))
            
            # Perform relevant statistical tests
            test_results = []
            for test_name in hypothesis.get('required_tests', []):
                if test_name in self.data_analyzer.available_tests:
                    try:
                        if test_name == 't_test':
                            if 'group1' not in datasets or 'group2' not in datasets:
                                raise KeyError(f"Missing required groups for t-test")
                            result = self.data_analyzer.perform_t_test(
                                datasets['group1'],
                                datasets['group2']
                            )
                        elif test_name == 'correlation':
                            if ('var1' in datasets and 'var2' in datasets):
                                result = self.data_analyzer.perform_correlation(
                                    datasets['var1'],
                                    datasets['var2']
                                )
                            elif ('x' in datasets and 'y' in datasets):
                                result = self.data_analyzer.perform_correlation(
                                    datasets['x'],
                                    datasets['y']
                                )
                            else:
                                raise KeyError("Missing correlation variables (need either var1/var2 or x/y)")
                        elif test_name == 'anova':
                            groups = [datasets[g] for g in datasets.keys()]
                            if len(groups) < 2:
                                raise ValueError("ANOVA requires at least 2 groups")
                            result = self.data_analyzer.perform_anova(groups)
                        elif test_name == 'regression':
                            if 'X' not in datasets or 'y' not in datasets:
                                raise KeyError("Missing X or y for regression analysis")
                            result = self.data_analyzer.perform_regression(
                                datasets['X'],
                                datasets['y']
                            )
                        
                        test_results.append(result)
                        self.debugger.print_data_state(f"{test_name} result", result)
                        
                    except Exception as e:
                        logger.error(f"Error performing {test_name}: {str(e)}")
                        continue
            
            if not test_results:
                raise ValueError("No statistical tests were successfully completed")
            
            # Calculate overall confidence
            confidence = np.mean([1 - result.p_value for result in test_results])
            
            # Determine validation status
            validation_status = (
                "Validated" if confidence >= confidence_threshold
                else "Partially Validated" if confidence >= 0.5
                else "Not Validated"
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                test_results,
                confidence,
                validation_status
            )
            
            # Create summary visualization
            summary_viz = self.data_analyzer.create_summary_table(test_results)
            
            result = ValidationResult(
                hypothesis=hypothesis.get('statement', ''),
                test_results=test_results,
                validation_status=validation_status,
                confidence=confidence,
                recommendations=recommendations,
                visualization=summary_viz
            )
            
            self.debugger.print_data_state("validation_result", result)
            return result
            
        except Exception as e:
            logger.error(f"Error in validate_hypothesis: {str(e)}")
            raise

    def _generate_recommendations(self,
                                test_results: List[StatTestResult],
                                confidence: float,
                                validation_status: str) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if validation_status == "Not Validated":
            recommendations.extend([
                "Consider revising hypothesis statement",
                "Increase sample size for more statistical power",
                "Check for potential confounding variables"
            ])
        elif validation_status == "Partially Validated":
            recommendations.extend([
                "Additional tests may be needed",
                "Consider collecting more data",
                "Examine effect sizes for practical significance"
            ])
        else:
            recommendations.extend([
                "Proceed with formal experimental validation",
                "Document methodology for replication",
                "Consider publishing findings"
            ])
        
        # Add test-specific recommendations
        for result in test_results:
            if result.p_value > 0.05:
                recommendations.append(
                    f"Review {result.test_name} assumptions and data quality"
                )
            if result.effect_size < 0.3:
                recommendations.append(
                    f"Effect size for {result.test_name} is small, "
                    "consider practical significance"
                )
                
        return list(set(recommendations))  # Remove any duplicates

    def create_validation_summary(self, 
                                results: List[ValidationResult]) -> go.Figure:
        """Create a summary visualization of multiple hypothesis validations"""
        headers = [
            'Hypothesis',
            'Validation Status',
            'Confidence',
            'Key Recommendations'
        ]
        
        rows = [[
            r.hypothesis,
            r.validation_status,
            f"{r.confidence:.2%}",
            "<br>".join(r.recommendations[:2])  # Show top 2 recommendations
        ] for r in results]
        
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
            title='Hypothesis Validation Summary',
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='#1a1b26'
        )
        
        return fig

# Example usage
if __name__ == "__main__":
    # Example hypothesis dictionary
    example_hypothesis = {
        'statement': 'Group 2 shows significantly higher values than Group 1',
        'data_params': {
            'group1': {'mean': 0, 'std': 1, 'size': 100},
            'group2': {'mean': 2, 'std': 1, 'size': 100}
        },
        'required_tests': ['t_test']
    }
    
    # Create validator and run validation
    validator = HypothesisValidator()
    result = validator.validate_hypothesis(example_hypothesis)
    
    # Print results
    print(f"Hypothesis: {result.hypothesis}")
    print(f"Status: {result.validation_status}")
    print(f"Confidence: {result.confidence:.2%}")
    print("\nRecommendations:")
    for rec in result.recommendations:
        print(f"- {rec}")