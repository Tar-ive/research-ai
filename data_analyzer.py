import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional, Union, Any
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
import logging
from typing import TypeVar, Generic

# Type variable for generic type hints
T = TypeVar('T', np.ndarray, pd.Series, pd.DataFrame)


@dataclass
class StatTestResult:
    """
    Dataclass to store statistical test results

    Attributes:
        test_name (str): Name of the statistical test
        statistic (float): Test statistic value
        p_value (float): P-value from the test
        effect_size (float): Effect size measure
        interpretation (str): Human-readable interpretation of results
        visualization (Optional[go.Figure]): Plotly figure for visualization
        error (Optional[str]): Error message if analysis failed
    """
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    interpretation: str
    visualization: Optional[go.Figure] = None
    error: Optional[str] = None


class DataAnalyzer:
    """
    Class for performing statistical analysis and generating visualizations
    
    Methods:
        generate_synthetic_data: Generate synthetic data for testing
        perform_t_test: Perform independent t-test
        perform_anova: Perform one-way ANOVA
        perform_regression: Perform linear regression
        perform_correlation: Perform correlation analysis
    """

    def __init__(self):
        """Initialize DataAnalyzer with available tests and logging setup"""
        self.available_tests = {
            't_test': self.perform_t_test,
            'anova': self.perform_anova,
            'regression': self.perform_regression,
            'correlation': self.perform_correlation
        }
        self.logger = logging.getLogger('DataAnalyzer')
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def generate_synthetic_data(self,
                                mean: float,
                                std: float,
                                size: int,
                                distribution: str = 'normal') -> np.ndarray:
        """
        Generate synthetic data for testing purposes
        
        Args:
            mean: Central tendency parameter
            std: Spread parameter (standard deviation)
            size: Number of samples to generate
            distribution: Type of distribution ('normal' or 'uniform')
            
        Returns:
            np.ndarray: Generated synthetic data
            
        Raises:
            ValueError: If input parameters are invalid
        """
        try:
            # Input validation with detailed error messages
            if not isinstance(mean, (int, float)):
                raise ValueError(f"Mean must be numeric, got {type(mean)}")
            if not isinstance(std, (int, float)):
                raise ValueError(
                    f"Standard deviation must be numeric, got {type(std)}")
            if std <= 0:
                raise ValueError(
                    f"Standard deviation must be positive, got {std}")
            if not isinstance(size, int):
                raise ValueError(f"Size must be integer, got {type(size)}")
            if size <= 0:
                raise ValueError(f"Size must be positive, got {size}")
            if distribution not in ['normal', 'uniform']:
                raise ValueError(f"Unsupported distribution: {distribution}")

            # Log input parameters
            self.logger.debug(
                f"Generating synthetic data - Parameters: mean={mean}, std={std}, size={size}, distribution={distribution}"
            )

            # Generate data based on distribution type
            if distribution == 'normal':
                data = np.random.normal(mean, std, size)
            else:  # uniform
                data = np.random.uniform(mean - std, mean + std, size)

            # Log generated data characteristics
            self.logger.debug(
                f"Generated data summary: shape={data.shape}, mean={np.mean(data):.2f}, std={np.std(data):.2f}"
            )
            self.logger.debug(
                f"Data range: min={np.min(data):.2f}, max={np.max(data):.2f}")

            return data

        except Exception as e:
            self.logger.error(f"Error generating synthetic data: {str(e)}")
            raise ValueError(f"Failed to generate synthetic data: {str(e)}")

    def perform_t_test(self, group1: np.ndarray, group2: np.ndarray) -> StatTestResult:
        """
        Perform independent t-test between two groups with enhanced validation
        
        Args:
            group1: First group data
            group2: Second group data
            
        Returns:
            StatTestResult: Results of the t-test
            
        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Input validation
            if not isinstance(group1, np.ndarray) or not isinstance(group2, np.ndarray):
                raise ValueError(f"Invalid input types: group1={type(group1)}, group2={type(group2)}")
            
            if len(group1) < 2 or len(group2) < 2:
                raise ValueError(f"Groups must have at least 2 samples: group1={len(group1)}, group2={len(group2)}")
                
            if not np.isfinite(group1).all() or not np.isfinite(group2).all():
                raise ValueError("Groups contain non-finite values")

            # Log input characteristics
            self.logger.debug(f"T-test input - Group 1: n={len(group1)}, mean={np.mean(group1):.2f}, std={np.std(group1):.2f}")
            self.logger.debug(f"T-test input - Group 2: n={len(group2)}, mean={np.mean(group2):.2f}, std={np.std(group2):.2f}")

            # Perform t-test
            statistic, p_value = stats.ttest_ind(group1, group2)
            
            if not np.isfinite(statistic) or not np.isfinite(p_value):
                raise ValueError("Test produced non-finite results")

            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
            effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std != 0 else 0

            # Create enhanced visualization
            fig = go.Figure()

            # Add box plots with outlier points
            fig.add_trace(go.Box(
                y=group1,
                name='Group 1',
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8
            ))
            fig.add_trace(go.Box(
                y=group2,
                name='Group 2',
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8
            ))

            # Enhanced layout
            fig.update_layout(
                title='Group Comparison (T-Test)',
                yaxis_title='Values',
                showlegend=True,
                template='plotly_white',
                boxmode='group',
                annotations=[
                    dict(
                        x=0.5,
                        y=1.05,
                        showarrow=False,
                        text=f'p-value: {p_value:.4f}',
                        xref='paper',
                        yref='paper'
                    )
                ]
            )

            # Create detailed interpretation
            if p_value < 0.05:
                effect_magnitude = "large" if abs(effect_size) > 0.8 else "medium" if abs(effect_size) > 0.5 else "small"
                interpretation = (
                    f"Significant difference found (p < 0.05)\n"
                    f"Effect size (Cohen's d) = {effect_size:.2f} ({effect_magnitude} effect)"
                )
            else:
                interpretation = (
                    f"No significant difference found (p = {p_value:.4f})\n"
                    f"Effect size (Cohen's d) = {effect_size:.2f}"
                )

            # Log results
            self.logger.debug(f"T-test results: statistic={statistic:.3f}, p={p_value:.3f}, effect_size={effect_size:.3f}")

            return StatTestResult(
                test_name='Independent T-Test',
                statistic=statistic,
                p_value=p_value,
                effect_size=effect_size,
                interpretation=interpretation,
                visualization=fig
            )

        except ValueError as ve:
            self.logger.error(f"T-test validation error: {str(ve)}")
            return StatTestResult(
                test_name='T-Test',
                statistic=float('nan'),
                p_value=float('nan'),
                effect_size=float('nan'),
                interpretation=f"Test failed: {str(ve)}",
                visualization=None,
                error=str(ve)
            )
        except Exception as e:
            self.logger.error(f"T-test unexpected error: {str(e)}")
            return StatTestResult(
                test_name='T-Test',
                statistic=float('nan'),
                p_value=float('nan'),
                effect_size=float('nan'),
                interpretation=f"Test failed: Unexpected error occurred",
                visualization=None,
                error=str(e)
            )

    def perform_anova(self, groups: List[np.ndarray]) -> StatTestResult:
        """
        Perform one-way ANOVA on multiple groups
        
        Args:
            groups: List of arrays containing group data
            
        Returns:
            StatTestResult: Test results including visualization
            
        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Input validation
            if not isinstance(groups, list):
                raise ValueError(f"Groups must be a list, got {type(groups)}")
            if len(groups) < 2:
                raise ValueError("At least two groups required for ANOVA")

            for i, group in enumerate(groups):
                if not isinstance(group, np.ndarray):
                    raise ValueError(
                        f"Group {i+1} must be numpy array, got {type(group)}")
                if group.size == 0:
                    raise ValueError(f"Group {i+1} is empty")
                if not np.isfinite(group).all():
                    raise ValueError(f"Group {i+1} contains non-finite values")

            # Log input characteristics
            self.logger.debug(f"ANOVA input - Number of groups: {len(groups)}")
            for i, group in enumerate(groups):
                self.logger.debug(
                    f"Group {i+1} stats: n={len(group)}, mean={np.mean(group):.2f}, std={np.std(group):.2f}"
                )

            # Perform ANOVA
            f_statistic, p_value = stats.f_oneway(*groups)

            # Calculate effect size (eta-squared)
            total_mean = np.mean([np.mean(g) for g in groups])
            between_ss = sum(
                len(g) * (np.mean(g) - total_mean)**2 for g in groups)
            total_ss = sum(sum((x - total_mean)**2) for g in groups for x in g)
            effect_size = between_ss / total_ss

            # Log test results
            self.logger.debug(
                f"ANOVA results: F={f_statistic:.3f}, p={p_value:.3f}, eta_squared={effect_size:.3f}"
            )

            # Create visualization
            fig = go.Figure()
            for i, group in enumerate(groups):
                fig.add_trace(
                    go.Box(y=group, name=f'Group {i+1}', boxpoints='outliers'))

            fig.update_layout(title='One-way ANOVA Group Comparison',
                              yaxis_title='Values',
                              template='plotly_dark',
                              showlegend=True)

            # Interpret results
            interpretation = (
                f"Significant differences between groups (p < 0.05)\n"
                f"Effect size (η²={effect_size:.2f}) is " +
                ("large" if effect_size > 0.14 else
                 "medium" if effect_size > 0.06 else "small")
            ) if p_value < 0.05 else "No significant differences between groups"

            return StatTestResult(test_name='One-way ANOVA',
                                  statistic=f_statistic,
                                  p_value=p_value,
                                  effect_size=effect_size,
                                  interpretation=interpretation,
                                  visualization=fig)

        except Exception as e:
            self.logger.error(f"Error in ANOVA: {str(e)}")
            raise ValueError(f"ANOVA failed: {str(e)}")

    def perform_regression(self, X: Union[np.ndarray, pd.Series],
                           y: Union[np.ndarray, pd.Series]) -> StatTestResult:
        """
        Perform linear regression analysis
        
        Args:
            X: Independent variable(s)
            y: Dependent variable
            
        Returns:
            StatTestResult: Test results including visualization
            
        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Input validation
            if not isinstance(X, (np.ndarray, pd.Series)):
                raise ValueError(
                    f"X must be numpy array or pandas Series, got {type(X)}")
            if not isinstance(y, (np.ndarray, pd.Series)):
                raise ValueError(
                    f"y must be numpy array or pandas Series, got {type(y)}")

            # Ensure proper shapes
            X = np.array(X).reshape(-1, 1) if len(
                X.shape) == 1 else np.array(X)
            y = np.array(y).reshape(-1, 1) if len(
                y.shape) == 1 else np.array(y)

            if len(X) != len(y):
                raise ValueError(
                    f"X and y must have same length, got {len(X)} and {len(y)}"
                )

            # Log input characteristics
            self.logger.debug(
                f"Regression input - X shape: {X.shape}, y shape: {y.shape}")

            # Add constant term and fit model
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit()

            # Log model summary
            self.logger.debug(
                f"Regression results - R²={model.rsquared:.3f}, F={model.fvalue:.3f}, p={model.f_pvalue:.3f}"
            )

            # Create visualization
            fig = go.Figure()

            # Scatter plot of data points
            fig.add_trace(
                go.Scatter(x=X.flatten(),
                           y=y.flatten(),
                           mode='markers',
                           name='Data Points',
                           marker=dict(size=8, opacity=0.6)))

            # Regression line
            x_range = np.linspace(X.min(), X.max(), 100)
            X_pred = sm.add_constant(x_range.reshape(-1, 1))
            y_pred = model.predict(X_pred)

            fig.add_trace(
                go.Scatter(x=x_range,
                           y=y_pred,
                           mode='lines',
                           name='Regression Line',
                           line=dict(color='red', width=2)))

            fig.update_layout(title='Linear Regression Analysis',
                              xaxis_title='Independent Variable',
                              yaxis_title='Dependent Variable',
                              template='plotly_dark',
                              showlegend=True)

            # Interpret results
            interpretation = (
                f"Model is significant (p < 0.05)\n"
                f"R² = {model.rsquared:.3f} ({model.rsquared*100:.1f}% variance explained)\n"
                f"Model fit is " +
                ("strong" if model.rsquared > 0.7 else
                 "moderate" if model.rsquared > 0.3 else "weak")
            ) if model.f_pvalue < 0.05 else "Model is not significant"

            return StatTestResult(test_name='Linear Regression',
                                  statistic=model.fvalue,
                                  p_value=model.f_pvalue,
                                  effect_size=model.rsquared,
                                  interpretation=interpretation,
                                  visualization=fig)

        except Exception as e:
            self.logger.error(f"Error in regression: {str(e)}")
            raise ValueError(f"Regression failed: {str(e)}")

    def perform_correlation(self, x: Union[np.ndarray, pd.Series],
                            y: Union[np.ndarray, pd.Series]) -> StatTestResult:
        """
        Perform correlation analysis between two variables with enhanced validation
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            StatTestResult: Test results including visualization
            
        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Input validation with detailed error messages
            if not isinstance(x, (np.ndarray, pd.Series)):
                raise ValueError(
                    f"x must be numpy array or pandas Series, got {type(x)}")
            if not isinstance(y, (np.ndarray, pd.Series)):
                raise ValueError(
                    f"y must be numpy array or pandas Series, got {type(y)}")

            # Convert to numpy arrays and flatten
            try:
                x = np.array(x, dtype=float).flatten()
                y = np.array(y, dtype=float).flatten()
            except (ValueError, TypeError) as e:
                raise ValueError(f"Error converting data to numeric arrays: {str(e)}")

            # Length validation
            if len(x) != len(y):
                raise ValueError(
                    f"x and y must have same length, got {len(x)} and {len(y)}")
            if len(x) < 2:
                raise ValueError(
                    "At least two data points required for correlation")

            # Check for non-finite values
            if not np.isfinite(x).all():
                raise ValueError("x contains non-finite values (inf or nan)")
            if not np.isfinite(y).all():
                raise ValueError("y contains non-finite values (inf or nan)")

            # Check for zero variance
            if np.var(x) == 0:
                raise ValueError("x has zero variance (all values are identical)")
            if np.var(y) == 0:
                raise ValueError("y has zero variance (all values are identical)")

            # Log input characteristics
            self.logger.debug(f"Correlation input - n={len(x)}")
            self.logger.debug(
                f"X stats: mean={np.mean(x):.2f}, std={np.std(x):.2f}, range=[{np.min(x):.2f}, {np.max(x):.2f}]")
            self.logger.debug(
                f"Y stats: mean={np.mean(y):.2f}, std={np.std(y):.2f}, range=[{np.min(y):.2f}, {np.max(y):.2f}]")

            # Calculate correlation with error handling
            try:
                correlation, p_value = stats.pearsonr(x, y)
                if not np.isfinite(correlation) or not np.isfinite(p_value):
                    raise ValueError("Correlation calculation produced non-finite results")
            except Exception as e:
                raise ValueError(f"Error calculating correlation: {str(e)}")

            # Log results
            self.logger.debug(
                f"Correlation results: r={correlation:.3f}, p={p_value:.3f}")

            # Create enhanced visualization
            fig = go.Figure()

            # Scatter plot
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',
                name='Data Points',
                marker=dict(
                    size=8,
                    color='#1f77b4',
                    opacity=0.6,
                    line=dict(width=1, color='DarkSlateGrey')
                )
            ))

            # Add trend line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_range = np.linspace(np.min(x), np.max(x), 100)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                name='Trend Line',
                line=dict(color='red', dash='dash')
            ))

            # Enhanced layout
            fig.update_layout(
                title=f'Correlation Analysis (r={correlation:.3f}, p={p_value:.4f})',
                xaxis_title='X Variable',
                yaxis_title='Y Variable',
                template='plotly_white',
                showlegend=True,
                annotations=[
                    dict(
                        x=0.02,
                        y=0.98,
                        text=f'n = {len(x)}',
                        showarrow=False,
                        xref='paper',
                        yref='paper'
                    )
                ]
            )

            # Create detailed interpretation
            if p_value < 0.05:
                strength = ("strong" if abs(correlation) > 0.7 else
                          "moderate" if abs(correlation) > 0.3 else "weak")
                interpretation = (
                    f"Significant {'positive' if correlation > 0 else 'negative'} "
                    f"correlation (p < 0.05)\n"
                    f"Correlation strength (r={abs(correlation):.2f}) is {strength}\n"
                    f"R² = {correlation**2:.3f} ({correlation**2*100:.1f}% of variance explained)"
                )
            else:
                interpretation = (
                    f"No significant correlation (p = {p_value:.4f})\n"
                    f"Correlation coefficient: r = {correlation:.3f}"
                )

            return StatTestResult(
                test_name='Pearson Correlation',
                statistic=correlation,
                p_value=p_value,
                effect_size=correlation**2,  # r-squared
                interpretation=interpretation,
                visualization=fig)

        except ValueError as ve:
            self.logger.error(f"Correlation validation error: {str(ve)}")
            return StatTestResult(
                test_name='Pearson Correlation',
                statistic=float('nan'),
                p_value=float('nan'),
                effect_size=float('nan'),
                interpretation=f"Analysis failed: {str(ve)}",
                visualization=None,
                error=str(ve)
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in correlation: {str(e)}")
            return StatTestResult(
                test_name='Pearson Correlation',
                statistic=float('nan'),
                p_value=float('nan'),
                effect_size=float('nan'),
                interpretation="Analysis failed: Unexpected error occurred",
                visualization=None,
                error=str(e)
            )

    def create_summary_table(self, results: List[StatTestResult]) -> go.Figure:
        """
        Create a summary table of multiple statistical test results
        
        Args:
            results: List of StatTestResult objects
            
        Returns:
            go.Figure: Plotly figure containing summary table
            
        Raises:
            ValueError: If input is invalid
        """
        try:
            # Input validation
            if not isinstance(results, list):
                raise ValueError(
                    f"Results must be a list, got {type(results)}")
            if not results:
                raise ValueError("Results list is empty")
            if not all(isinstance(r, StatTestResult) for r in results):
                raise ValueError("All results must be StatTestResult objects")

            # Create table data
            headers = [
                'Test', 'Statistic', 'P-Value', 'Effect Size', 'Interpretation'
            ]
            rows = [[
                r.test_name, f"{r.statistic:.3f}", f"{r.p_value:.3f}",
                f"{r.effect_size:.3f}", r.interpretation
            ] for r in results]

            # Create figure
            fig = go.Figure(data=[
                go.Table(header=dict(values=headers,
                                     fill_color='#1a1b26',
                                     align='left',
                                     font=dict(color='white')),
                         cells=dict(values=list(zip(*rows)),
                                    fill_color='#24283b',
                                    align='left',
                                    font=dict(color='white')))
            ])

            fig.update_layout(title='Statistical Analysis Summary',
                              margin=dict(l=0, r=0, t=30, b=0),
                              paper_bgcolor='#1a1b26')

            return fig

        except Exception as e:
            self.logger.error(f"Error creating summary table: {str(e)}")
            raise ValueError(f"Failed to create summary table: {str(e)}")
