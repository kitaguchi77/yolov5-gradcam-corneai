"""
Visualization Module

This module provides functions for visualizing GradCAM++ results,
statistical analyses, and cut-and-paste validation outcomes.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import cv2
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """
    Handles visualization of analysis results.
    """
    
    def __init__(self, output_dir: str = 'results/visualizations',
                dpi: int = 300):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            dpi: DPI for saved figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        # Custom colormap for heatmaps (blue to red)
        colors = ['blue', 'cyan', 'yellow', 'red']
        n_bins = 256
        self.custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
    def plot_cam_overlay(self, image: np.ndarray,
                        cam: np.ndarray,
                        title: str = "Grad-CAM++ Visualization",
                        alpha: float = 0.5,
                        colorbar: bool = True,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot CAM heatmap overlaid on image.
        
        Args:
            image: Original image
            cam: CAM heatmap
            title: Plot title
            alpha: Overlay transparency
            colorbar: Whether to show colorbar
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Display image
        ax.imshow(image)
        
        # Overlay CAM
        im = ax.imshow(cam, cmap=self.custom_cmap, alpha=alpha, 
                      interpolation='bilinear')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        if colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Activation', fontsize=12)
            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_layer_comparison(self, cams_dict: Dict[str, np.ndarray],
                            original_image: np.ndarray,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot CAMs from different layers for comparison.
        
        Args:
            cams_dict: Dictionary of layer_name -> CAM
            original_image: Original image
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_layers = len(cams_dict)
        n_cols = 3
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, (layer_name, cam) in enumerate(cams_dict.items()):
            if idx < len(axes):
                ax = axes[idx]
                ax.imshow(original_image)
                im = ax.imshow(cam, cmap=self.custom_cmap, alpha=0.5)
                ax.set_title(f'Layer {layer_name}', fontsize=14)
                ax.axis('off')
                
        # Remove empty subplots
        for idx in range(len(cams_dict), len(axes)):
            axes[idx].axis('off')
            
        plt.suptitle('Grad-CAM++ Across Layers', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_aoi_boxplot(self, aoi_data: pd.DataFrame,
                        x_col: str = 'layer',
                        y_col: str = 'aoi_value',
                        hue_col: Optional[str] = 'class',
                        title: str = "AOI Distribution by Layer",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create boxplot for AOI values.
        
        Args:
            aoi_data: DataFrame with AOI values
            x_col: Column for x-axis
            y_col: Column for y-axis (AOI values)
            hue_col: Column for grouping
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if hue_col:
            sns.boxplot(data=aoi_data, x=x_col, y=y_col, hue=hue_col, ax=ax)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            sns.boxplot(data=aoi_data, x=x_col, y=y_col, ax=ax)
            
        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('AOI_50 Value', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Add significance markers if available
        if 'significant' in aoi_data.columns:
            # Add markers for significant differences
            pass
            
        plt.xticks(rotation=45 if len(aoi_data[x_col].unique()) > 5 else 0)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_cut_paste_heatmap(self, accuracy_matrix: np.ndarray,
                              class_names: List[str],
                              title: str = "Cut-and-Paste Accuracy Matrix",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot heatmap of cut-and-paste accuracy matrix.
        
        Args:
            accuracy_matrix: Accuracy matrix (source x background)
            class_names: List of class names
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(accuracy_matrix, 
                   xticklabels=class_names,
                   yticklabels=class_names,
                   annot=True,
                   fmt='.2f',
                   cmap='RdYlBu_r',
                   center=0.5,
                   vmin=0,
                   vmax=1,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={'label': 'Accuracy'},
                   ax=ax)
        
        ax.set_xlabel('Background Class', fontsize=14, fontweight='bold')
        ax.set_ylabel('Source Class', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Rotate labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Highlight diagonal
        for i in range(len(class_names)):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=False, 
                                 edgecolor='black', lw=3))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_statistical_comparison(self, stats_results: Dict,
                                   metric_name: str = "AOI_50",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize statistical test results.
        
        Args:
            stats_results: Dictionary of statistical test results
            metric_name: Name of the metric being compared
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if 'comparisons' not in stats_results:
            logger.warning("No pairwise comparisons found in results")
            return None
            
        comparisons = stats_results['comparisons']
        
        # Create comparison matrix
        groups = list(set([c['group1'] for c in comparisons] + 
                         [c['group2'] for c in comparisons]))
        n_groups = len(groups)
        
        p_matrix = np.ones((n_groups, n_groups))
        
        for comp in comparisons:
            i = groups.index(comp['group1'])
            j = groups.index(comp['group2'])
            p_val = comp['p_value_corrected']
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val
            
        # Create significance matrix
        sig_matrix = p_matrix < 0.05
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # P-value heatmap
        sns.heatmap(p_matrix, 
                   xticklabels=groups,
                   yticklabels=groups,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlGn_r',
                   center=0.05,
                   vmin=0,
                   vmax=0.1,
                   square=True,
                   ax=ax1)
        ax1.set_title('Corrected P-values', fontsize=14)
        
        # Significance heatmap
        sns.heatmap(sig_matrix,
                   xticklabels=groups,
                   yticklabels=groups,
                   cmap=['white', 'darkgreen'],
                   cbar_kws={'label': 'Significant (p < 0.05)'},
                   square=True,
                   ax=ax2)
        ax2.set_title('Significant Differences', fontsize=14)
        
        plt.suptitle(f'{metric_name} Statistical Comparisons', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def plot_iou_distribution(self, iou_data: pd.DataFrame,
                            group_col: str = 'class',
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot IoU distribution by class or layer.
        
        Args:
            iou_data: DataFrame with IoU values
            group_col: Column to group by
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Violin plot
        sns.violinplot(data=iou_data, x=group_col, y='iou', ax=ax1)
        ax1.set_xlabel('')
        ax1.set_ylabel('IoU', fontsize=12)
        ax1.set_title('IoU Distribution', fontsize=14)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        
        # Strip plot with correct/incorrect coloring
        if 'correct' in iou_data.columns:
            sns.stripplot(data=iou_data, x=group_col, y='iou', 
                         hue='correct', ax=ax2, alpha=0.7)
            ax2.legend(title='Prediction', loc='lower right')
        else:
            sns.stripplot(data=iou_data, x=group_col, y='iou', ax=ax2, alpha=0.7)
            
        ax2.set_xlabel(group_col.replace('_', ' ').title(), fontsize=12)
        ax2.set_ylabel('IoU', fontsize=12)
        ax2.set_title('Individual IoU Values', fontsize=14)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        
        plt.xticks(rotation=45 if len(iou_data[group_col].unique()) > 5 else 0)
        plt.suptitle('IoU Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def create_interactive_cam_plot(self, image: np.ndarray,
                                  cams_dict: Dict[str, np.ndarray],
                                  save_path: Optional[str] = None):
        """
        Create interactive plot for exploring CAMs across layers.
        
        Args:
            image: Original image
            cams_dict: Dictionary of layer_name -> CAM
            save_path: Path to save HTML
        """
        fig = go.Figure()
        
        # Add original image
        fig.add_trace(go.Image(z=image, name='Original', visible=True))
        
        # Add CAM overlays for each layer
        for layer_name, cam in cams_dict.items():
            # Create overlay
            overlay = image.copy().astype(float)
            cam_colored = plt.cm.jet(cam)[:, :, :3] * 255
            overlay = overlay * 0.5 + cam_colored * 0.5
            
            fig.add_trace(go.Image(z=overlay.astype(np.uint8), 
                                 name=f'Layer {layer_name}',
                                 visible=False))
        
        # Create buttons for layer selection
        buttons = []
        for i, layer_name in enumerate(['Original'] + list(cams_dict.keys())):
            visibility = [False] * (len(cams_dict) + 1)
            visibility[i] = True
            
            buttons.append(dict(
                label=layer_name,
                method='update',
                args=[{'visible': visibility}]
            ))
        
        fig.update_layout(
            updatemenus=[dict(
                type='buttons',
                direction='left',
                buttons=buttons,
                pad={'r': 10, 't': 10},
                showactive=True,
                x=0.1,
                xanchor='left',
                y=1.1,
                yanchor='top'
            )],
            title='Interactive Grad-CAM++ Visualization',
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_context_dependency(self, dependency_scores: Dict[str, float],
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot context dependency scores for each class.
        
        Args:
            dependency_scores: Dictionary of class -> dependency score
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = list(dependency_scores.keys())
        scores = list(dependency_scores.values())
        
        # Create bar plot
        bars = ax.bar(classes, scores, color=['red' if s > 0.1 else 'blue' 
                                             for s in scores])
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom' if height > 0 else 'top')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Disease Class', fontsize=12)
        ax.set_ylabel('Context Dependency Score', fontsize=12)
        ax.set_title('Context Dependency by Disease Class', fontsize=16, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def generate_report_figures(self, results: Dict,
                              output_prefix: str = 'report'):
        """
        Generate all figures for the analysis report.
        
        Args:
            results: Dictionary containing all analysis results
            output_prefix: Prefix for output filenames
        """
        figures = {}
        
        # Generate each type of figure
        if 'aoi_data' in results:
            fig = self.plot_aoi_boxplot(
                results['aoi_data'],
                save_path=self.output_dir / f'{output_prefix}_aoi_boxplot.png'
            )
            figures['aoi_boxplot'] = fig
            
        if 'cut_paste_matrix' in results:
            fig = self.plot_cut_paste_heatmap(
                results['cut_paste_matrix'],
                results['class_names'],
                save_path=self.output_dir / f'{output_prefix}_cutpaste_matrix.png'
            )
            figures['cutpaste_matrix'] = fig
            
        if 'statistical_results' in results:
            fig = self.plot_statistical_comparison(
                results['statistical_results'],
                save_path=self.output_dir / f'{output_prefix}_statistics.png'
            )
            figures['statistics'] = fig
            
        logger.info(f"Generated {len(figures)} figures for report")
        
        return figures