"""
Statistical Analysis Module

This module implements statistical tests for analyzing GradCAM++ results
and cut-and-paste validation outcomes.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Performs statistical analyses for model evaluation results.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Args:
            significance_level: Significance level for hypothesis tests
        """
        self.alpha = significance_level
        
    def kruskal_wallis_test(self, groups: List[np.ndarray], 
                           group_names: Optional[List[str]] = None) -> Dict:
        """
        Perform Kruskal-Wallis H test for comparing multiple groups.
        
        Args:
            groups: List of arrays, each representing a group
            group_names: Optional names for groups
            
        Returns:
            Test results dictionary
        """
        # Filter out empty groups
        valid_groups = [g for g in groups if len(g) > 0]
        
        if len(valid_groups) < 2:
            logger.warning("Less than 2 non-empty groups for Kruskal-Wallis test")
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'df': 0,
                'n_groups': len(valid_groups),
                'significant': False
            }
        
        # Perform test
        h_stat, p_value = stats.kruskal(*valid_groups)
        
        results = {
            'test': 'Kruskal-Wallis',
            'statistic': float(h_stat),
            'p_value': float(p_value),
            'df': len(valid_groups) - 1,
            'n_groups': len(valid_groups),
            'significant': p_value < self.alpha
        }
        
        # Add group statistics
        if group_names:
            group_stats = {}
            for i, (group, name) in enumerate(zip(groups, group_names)):
                if len(group) > 0:
                    group_stats[name] = {
                        'n': len(group),
                        'mean': float(np.mean(group)),
                        'median': float(np.median(group)),
                        'std': float(np.std(group)),
                        'iqr': float(np.percentile(group, 75) - np.percentile(group, 25))
                    }
            results['group_statistics'] = group_stats
            
        return results
    
    def dunn_test(self, groups: List[np.ndarray],
                  group_names: Optional[List[str]] = None,
                  correction: str = 'bonferroni') -> Dict:
        """
        Perform Dunn's test for post-hoc pairwise comparisons.
        
        Args:
            groups: List of arrays for each group
            group_names: Optional names for groups
            correction: Multiple comparison correction method
            
        Returns:
            Test results dictionary
        """
        # Prepare data
        data = []
        labels = []
        
        for i, group in enumerate(groups):
            if len(group) > 0:
                data.extend(group)
                label = group_names[i] if group_names else f"Group_{i}"
                labels.extend([label] * len(group))
                
        if len(set(labels)) < 2:
            logger.warning("Less than 2 groups for Dunn test")
            return {'error': 'Insufficient groups'}
            
        # Convert to DataFrame
        df = pd.DataFrame({'value': data, 'group': labels})
        
        # Rank the data
        df['rank'] = df['value'].rank()
        
        # Calculate mean ranks
        mean_ranks = df.groupby('group')['rank'].agg(['mean', 'count'])
        
        # Pairwise comparisons
        unique_groups = df['group'].unique()
        n_comparisons = len(unique_groups) * (len(unique_groups) - 1) // 2
        comparisons = []
        
        n_total = len(data)
        
        for i in range(len(unique_groups)):
            for j in range(i + 1, len(unique_groups)):
                group1 = unique_groups[i]
                group2 = unique_groups[j]
                
                # Get mean ranks and sample sizes
                r1 = mean_ranks.loc[group1, 'mean']
                r2 = mean_ranks.loc[group2, 'mean']
                n1 = mean_ranks.loc[group1, 'count']
                n2 = mean_ranks.loc[group2, 'count']
                
                # Calculate z-statistic
                se = np.sqrt((n_total * (n_total + 1) / 12) * (1/n1 + 1/n2))
                z = abs(r1 - r2) / se
                
                # Two-tailed p-value
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
                
                comparisons.append({
                    'group1': group1,
                    'group2': group2,
                    'mean_rank_diff': float(abs(r1 - r2)),
                    'z_statistic': float(z),
                    'p_value': float(p_value),
                    'n1': int(n1),
                    'n2': int(n2)
                })
        
        # Apply multiple comparison correction
        p_values = [comp['p_value'] for comp in comparisons]
        
        if correction == 'bonferroni':
            corrected_p = [min(p * n_comparisons, 1.0) for p in p_values]
        else:
            _, corrected_p, _, _ = multipletests(p_values, method=correction)
            
        # Update comparisons with corrected p-values
        for comp, p_corr in zip(comparisons, corrected_p):
            comp['p_value_corrected'] = float(p_corr)
            comp['significant'] = p_corr < self.alpha
            
        results = {
            'test': 'Dunn',
            'correction': correction,
            'n_comparisons': n_comparisons,
            'comparisons': comparisons,
            'mean_ranks': mean_ranks.to_dict()
        }
        
        return results
    
    def mann_whitney_u_test(self, group1: np.ndarray, 
                           group2: np.ndarray,
                           alternative: str = 'two-sided') -> Dict:
        """
        Perform Mann-Whitney U test for comparing two groups.
        
        Args:
            group1: First group data
            group2: Second group data
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Test results dictionary
        """
        if len(group1) == 0 or len(group2) == 0:
            logger.warning("Empty group(s) for Mann-Whitney U test")
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False
            }
            
        # Perform test
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        r = 1 - (2 * u_stat) / (n1 * n2)
        
        results = {
            'test': 'Mann-Whitney U',
            'statistic': float(u_stat),
            'p_value': float(p_value),
            'alternative': alternative,
            'effect_size': float(r),
            'n1': n1,
            'n2': n2,
            'significant': p_value < self.alpha,
            'group1_median': float(np.median(group1)),
            'group2_median': float(np.median(group2))
        }
        
        return results
    
    def binomial_test(self, successes: int, n_trials: int,
                     expected_prob: float = 0.5,
                     alternative: str = 'two-sided') -> Dict:
        """
        Perform binomial test.
        
        Args:
            successes: Number of successes
            n_trials: Total number of trials
            expected_prob: Expected probability of success
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Test results dictionary
        """
        # Use scipy's binomial test
        p_value = stats.binomtest(successes, n_trials, expected_prob, 
                                 alternative=alternative).pvalue
        
        observed_prob = successes / n_trials if n_trials > 0 else 0
        
        results = {
            'test': 'Binomial',
            'successes': successes,
            'n_trials': n_trials,
            'observed_probability': float(observed_prob),
            'expected_probability': float(expected_prob),
            'p_value': float(p_value),
            'alternative': alternative,
            'significant': p_value < self.alpha
        }
        
        return results
    
    def mcnemar_test(self, contingency_table: np.ndarray,
                    correction: bool = True) -> Dict:
        """
        Perform McNemar's test for paired nominal data.
        
        Args:
            contingency_table: 2x2 contingency table
            correction: Whether to apply continuity correction
            
        Returns:
            Test results dictionary
        """
        if contingency_table.shape != (2, 2):
            raise ValueError("Contingency table must be 2x2")
            
        # Extract values
        n01 = contingency_table[0, 1]  # Changed from 0 to 1
        n10 = contingency_table[1, 0]  # Changed from 1 to 0
        
        # Calculate test statistic
        if correction and (n01 + n10) > 0:
            chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        elif (n01 + n10) > 0:
            chi2 = (n01 - n10) ** 2 / (n01 + n10)
        else:
            chi2 = 0
            
        # Calculate p-value
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        
        results = {
            'test': 'McNemar',
            'statistic': float(chi2),
            'p_value': float(p_value),
            'correction': correction,
            'contingency_table': contingency_table.tolist(),
            'significant': p_value < self.alpha
        }
        
        return results
    
    def analyze_aoi_by_layer(self, aoi_data: pd.DataFrame) -> Dict:
        """
        Analyze AOI values across different layers.
        
        Args:
            aoi_data: DataFrame with columns 'layer', 'aoi_value', 'class'
            
        Returns:
            Analysis results
        """
        results = {}
        
        # Overall comparison across layers
        layers = aoi_data['layer'].unique()
        layer_groups = [aoi_data[aoi_data['layer'] == layer]['aoi_value'].values 
                       for layer in layers]
        
        results['overall_layer_comparison'] = self.kruskal_wallis_test(
            layer_groups, group_names=list(layers)
        )
        
        # Post-hoc if significant
        if results['overall_layer_comparison']['significant']:
            results['layer_pairwise'] = self.dunn_test(
                layer_groups, group_names=list(layers)
            )
            
        # Per-class analysis
        results['per_class'] = {}
        for class_name in aoi_data['class'].unique():
            class_data = aoi_data[aoi_data['class'] == class_name]
            
            class_layers = class_data['layer'].unique()
            class_groups = [class_data[class_data['layer'] == layer]['aoi_value'].values 
                          for layer in class_layers]
            
            if len(class_groups) >= 2:
                results['per_class'][class_name] = self.kruskal_wallis_test(
                    class_groups, group_names=list(class_layers)
                )
                
        return results
    
    def analyze_cut_paste_results(self, accuracy_matrix: np.ndarray,
                                 class_names: List[str]) -> Dict:
        """
        Analyze cut-and-paste validation results.
        
        Args:
            accuracy_matrix: Accuracy matrix (source x background)
            class_names: List of class names
            
        Returns:
            Analysis results
        """
        results = {}
        n_classes = len(class_names)
        
        # Test if diagonal (same background) is better than off-diagonal
        diagonal = np.diag(accuracy_matrix)
        off_diagonal = accuracy_matrix[~np.eye(n_classes, dtype=bool)]
        
        results['diagonal_vs_offdiagonal'] = self.mann_whitney_u_test(
            diagonal, off_diagonal, alternative='greater'
        )
        
        # Per-class context dependency
        results['per_class_dependency'] = {}
        
        for i, class_name in enumerate(class_names):
            if i < accuracy_matrix.shape[0]:
                same_bg = accuracy_matrix[i, i]
                diff_bg = np.concatenate([accuracy_matrix[i, :i], 
                                        accuracy_matrix[i, i+1:]])
                
                if len(diff_bg) > 0:
                    # Test if same background accuracy is higher
                    n_same = 1
                    n_diff = len(diff_bg)
                    
                    # Use binomial test
                    successes = np.sum(same_bg > diff_bg)
                    results['per_class_dependency'][class_name] = self.binomial_test(
                        successes, n_diff, expected_prob=0.5, alternative='greater'
                    )
                    
        return results
    
    def analyze_correctness_impact(self, metrics_df: pd.DataFrame,
                                  metric_columns: List[str]) -> Dict:
        """
        Analyze impact of prediction correctness on metrics.
        
        Args:
            metrics_df: DataFrame with 'correct' column and metric columns
            metric_columns: List of metric column names to analyze
            
        Returns:
            Analysis results
        """
        results = {}
        
        correct_data = metrics_df[metrics_df['correct'] == True]
        incorrect_data = metrics_df[metrics_df['correct'] == False]
        
        for metric in metric_columns:
            if metric in metrics_df.columns:
                correct_values = correct_data[metric].dropna().values
                incorrect_values = incorrect_data[metric].dropna().values
                
                if len(correct_values) > 0 and len(incorrect_values) > 0:
                    results[metric] = self.mann_whitney_u_test(
                        correct_values, incorrect_values
                    )
                    
        return results
    
    def generate_summary_statistics(self, data: pd.DataFrame,
                                   group_col: str,
                                   value_col: str) -> pd.DataFrame:
        """
        Generate summary statistics for grouped data.
        
        Args:
            data: Input DataFrame
            group_col: Column name for grouping
            value_col: Column name for values
            
        Returns:
            Summary statistics DataFrame
        """
        summary = data.groupby(group_col)[value_col].agg([
            'count', 'mean', 'std', 'min',
            ('q25', lambda x: x.quantile(0.25)),
            'median',
            ('q75', lambda x: x.quantile(0.75)),
            'max'
        ]).round(4)
        
        # Add IQR
        summary['iqr'] = summary['q75'] - summary['q25']
        
        return summary