#!/usr/bin/env python3
"""
YOLOv5 GradCAM++ and Cut-and-Paste Validation Pipeline

Main script for running the complete analysis pipeline for anterior segment
disease classification explainability.
"""

import argparse
import logging
import yaml
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from models.yolov5_model import YOLOv5Model
from models.yolov5_gradcam import YOLOv5GradCAMPlusPlus
from validation.cut_and_paste import CutAndPasteValidator
from validation.expert_comparison import ExpertComparisonValidator
from utils.data_loader import DataLoader
from utils.metrics import MetricsCalculator
from analysis.statistical_analysis import StatisticalAnalyzer
from analysis.visualization import Visualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='YOLOv5 GradCAM++ and Cut-and-Paste Analysis'
    )
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to YOLOv5 weights')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data list (CSV or JSON)')
    
    # Optional arguments
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for inference')
    
    # Analysis options
    parser.add_argument('--gradcam', action='store_true',
                       help='Run GradCAM++ analysis')
    parser.add_argument('--cut-paste', action='store_true',
                       help='Run cut-and-paste validation')
    parser.add_argument('--expert-comparison', action='store_true',
                       help='Compare with expert annotations')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')
    
    # Additional options
    parser.add_argument('--target-layers', nargs='+',
                       default=['17', '20', '23', '24_m_0', '24_m_1', '24_m_2'],
                       help='Target layers for GradCAM++')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--save-visualizations', action='store_true',
                       help='Save all visualizations')
    
    args = parser.parse_args()
    
    # If --all is specified, enable all analyses
    if args.all:
        args.gradcam = True
        args.cut_paste = True
        args.expert_comparison = True
        
    return args


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_gradcam_analysis(model: YOLOv5Model,
                        data_loader: DataLoader,
                        target_layers: list,
                        output_dir: Path,
                        visualizer: Visualizer) -> dict:
    """Run GradCAM++ analysis on test data."""
    logger.info("Starting GradCAM++ analysis...")
    
    # Initialize GradCAM++
    gradcam = YOLOv5GradCAMPlusPlus(model, target_layers, use_cuda=False)
    
    # Results storage
    all_results = []
    aoi_data = []
    
    # Process each image
    for img_data in tqdm(data_loader.images, desc="Processing images"):
        # Run inference
        results = model.predict(img_data.image_path)
        
        if len(results['classes']) > 0:
            # Get predicted class
            pred_class = results['classes'][results['scores'].argmax()]
            
            # Generate CAMs for all layers
            cams = gradcam.generate_all_cams(img_data.image_path, pred_class)
            
            # Collect AOI data
            for layer, (cam, metadata) in cams.items():
                aoi_data.append({
                    'image_path': img_data.image_path,
                    'true_class': img_data.label,
                    'pred_class': pred_class,
                    'layer': layer,
                    'aoi_value': metadata['aoi_50'],
                    'class': img_data.class_name,
                    'correct': pred_class == img_data.label
                })
                
            # Save visualizations if requested
            if visualizer:
                img = data_loader.load_image(img_data.image_path)
                fig = visualizer.plot_layer_comparison(
                    {k: v[0] for k, v in cams.items()},
                    img,
                    save_path=output_dir / 'gradcam' / f'{Path(img_data.image_path).stem}_layers.png'
                )
                plt.close(fig)
                
    # Create DataFrame
    df_aoi = pd.DataFrame(aoi_data)
    df_aoi.to_csv(output_dir / 'gradcam_aoi_results.csv', index=False)
    
    logger.info(f"GradCAM++ analysis complete. Processed {len(data_loader.images)} images")
    
    return {
        'aoi_data': df_aoi,
        'summary': df_aoi.groupby(['layer', 'class'])['aoi_value'].describe()
    }


def run_cut_paste_validation(model: YOLOv5Model,
                           data_loader: DataLoader,
                           config: dict,
                           output_dir: Path) -> dict:
    """Run cut-and-paste validation."""
    logger.info("Starting cut-and-paste validation...")
    
    # Initialize validator
    validator = CutAndPasteValidator(model, config.get('cut_paste', {}))
    
    # Load annotations
    if 'cornea_annotations' in config['paths']:
        validator.load_annotations(config['paths']['cornea_annotations'])
    else:
        logger.error("No cornea annotations path in config")
        return {}
        
    # Select backgrounds
    validator.backgrounds = data_loader.create_background_pool(
        min_confidence=config['cut_paste'].get('min_confidence', 0.9),
        model=model
    )
    
    # Run validation
    test_images = [img.image_path for img in data_loader.images]
    test_labels = [img.label for img in data_loader.images]
    
    results_df = validator.run_validation(
        test_images, test_labels,
        output_path=output_dir / 'cut_paste_results.csv'
    )
    
    # Compute accuracy matrix
    accuracy_matrix = validator.compute_accuracy_matrix(results_df)
    np.save(output_dir / 'cut_paste_accuracy_matrix.npy', accuracy_matrix)
    
    # Analyze context dependency
    dependency_analysis = validator.analyze_context_dependency(results_df)
    
    with open(output_dir / 'context_dependency_analysis.json', 'w') as f:
        json.dump(dependency_analysis, f, indent=2)
        
    logger.info("Cut-and-paste validation complete")
    
    return {
        'results_df': results_df,
        'accuracy_matrix': accuracy_matrix,
        'dependency_analysis': dependency_analysis
    }


def run_expert_comparison(model: YOLOv5Model,
                         gradcam: YOLOv5GradCAMPlusPlus,
                         data_loader: DataLoader,
                         config: dict,
                         output_dir: Path) -> dict:
    """Compare model attention with expert annotations."""
    logger.info("Starting expert annotation comparison...")
    
    # Initialize validator
    validator = ExpertComparisonValidator(gradcam)
    
    # Load expert annotations
    if 'expert_annotations' in config['paths']:
        validator.load_expert_annotations(config['paths']['expert_annotations'])
    else:
        logger.error("No expert annotations path in config")
        return {}
        
    # Get predictions
    predictions = []
    image_paths = []
    
    for img_data in data_loader.images:
        results = model.predict(img_data.image_path)
        if len(results['classes']) > 0:
            pred_class = results['classes'][results['scores'].argmax()]
            predictions.append(pred_class)
            image_paths.append(img_data.image_path)
            
    # Validate against expert annotations
    validation_df = validator.validate_dataset(
        image_paths, predictions,
        target_layers=['23'],  # Focus on layer 23 as per paper
        output_path=output_dir / 'expert_comparison_results.csv'
    )
    
    # Create ground truth mapping
    ground_truth = {img.image_path: img.label for img in data_loader.images}
    
    # Analyze by correctness
    correctness_analysis = validator.analyze_by_correctness(validation_df, ground_truth)
    
    # Compute clinical relevance scores
    relevance_scores = validator.compute_clinical_relevance_score(validation_df)
    
    # Save results
    with open(output_dir / 'clinical_relevance_analysis.json', 'w') as f:
        json.dump({
            'correctness_analysis': correctness_analysis,
            'relevance_scores': relevance_scores
        }, f, indent=2)
        
    logger.info("Expert comparison complete")
    
    return {
        'validation_df': validation_df,
        'correctness_analysis': correctness_analysis,
        'relevance_scores': relevance_scores
    }


def run_statistical_analysis(results: dict, 
                           config: dict,
                           output_dir: Path) -> dict:
    """Run statistical analyses on results."""
    logger.info("Running statistical analyses...")
    
    analyzer = StatisticalAnalyzer(
        significance_level=config['analysis'].get('significance_level', 0.05)
    )
    
    stats_results = {}
    
    # Analyze GradCAM++ results
    if 'gradcam_results' in results:
        aoi_data = results['gradcam_results']['aoi_data']
        
        # Analyze AOI by layer
        layer_analysis = analyzer.analyze_aoi_by_layer(aoi_data)
        stats_results['layer_analysis'] = layer_analysis
        
        # Analyze correctness impact
        correctness_impact = analyzer.analyze_correctness_impact(
            aoi_data, ['aoi_value']
        )
        stats_results['correctness_impact'] = correctness_impact
        
    # Analyze cut-and-paste results
    if 'cut_paste_results' in results:
        accuracy_matrix = results['cut_paste_results']['accuracy_matrix']
        class_names = list(model.class_names.values())
        
        cut_paste_stats = analyzer.analyze_cut_paste_results(
            accuracy_matrix, class_names
        )
        stats_results['cut_paste_analysis'] = cut_paste_stats
        
    # Save statistical results
    with open(output_dir / 'statistical_analysis_results.json', 'w') as f:
        json.dump(stats_results, f, indent=2, default=str)
        
    logger.info("Statistical analyses complete")
    
    return stats_results


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'gradcam').mkdir(exist_ok=True)
    (output_dir / 'cut_paste').mkdir(exist_ok=True)
    (output_dir / 'statistics').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command line arguments
    config['model']['weights_path'] = args.weights
    config['model']['device'] = args.device
    
    # Initialize model
    logger.info("Loading YOLOv5 model...")
    model = YOLOv5Model(
        weights_path=args.weights,
        device=args.device,
        config_path=args.config
    )
    
    # Initialize data loader
    logger.info("Loading test data...")
    data_loader = DataLoader(args.config)
    data_loader.load_image_list(args.test_data)
    
    # Initialize visualizer
    visualizer = Visualizer(
        output_dir=output_dir / 'visualizations',
        dpi=config['output'].get('dpi', 300)
    ) if args.save_visualizations else None
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator(model.class_names)
    
    # Results storage
    all_results = {}
    
    # Run GradCAM++ analysis
    if args.gradcam:
        gradcam_results = run_gradcam_analysis(
            model, data_loader, args.target_layers, 
            output_dir, visualizer
        )
        all_results['gradcam_results'] = gradcam_results
        
    # Run cut-and-paste validation  
    if args.cut_paste:
        cut_paste_results = run_cut_paste_validation(
            model, data_loader, config, output_dir
        )
        all_results['cut_paste_results'] = cut_paste_results
        
    # Run expert comparison
    if args.expert_comparison and args.gradcam:
        gradcam = YOLOv5GradCAMPlusPlus(model, args.target_layers)
        expert_results = run_expert_comparison(
            model, gradcam, data_loader, config, output_dir
        )
        all_results['expert_results'] = expert_results
        
    # Run statistical analyses
    if len(all_results) > 0:
        stats_results = run_statistical_analysis(
            all_results, config, output_dir / 'statistics'
        )
        all_results['statistical_results'] = stats_results
        
    # Generate visualizations
    if visualizer and len(all_results) > 0:
        logger.info("Generating visualizations...")
        
        # Prepare data for visualization
        viz_data = {
            'class_names': list(model.class_names.values())
        }
        
        if 'gradcam_results' in all_results:
            viz_data['aoi_data'] = all_results['gradcam_results']['aoi_data']
            
        if 'cut_paste_results' in all_results:
            viz_data['cut_paste_matrix'] = all_results['cut_paste_results']['accuracy_matrix']
            
        if 'statistical_results' in all_results:
            viz_data['statistical_results'] = all_results['statistical_results']
            
        # Generate report figures
        figures = visualizer.generate_report_figures(viz_data)
        
    # Generate summary report
    logger.info("Generating summary report...")
    summary = {
        'n_images_processed': len(data_loader.images),
        'analyses_performed': list(all_results.keys()),
        'output_directory': str(output_dir)
    }
    
    # Add key findings
    if 'gradcam_results' in all_results:
        aoi_summary = all_results['gradcam_results']['aoi_data'].groupby('layer')['aoi_value'].describe()
        summary['mean_aoi_by_layer'] = aoi_summary['mean'].to_dict()
        
    if 'cut_paste_results' in all_results:
        dep_scores = all_results['cut_paste_results']['dependency_analysis']
        summary['context_dependency_scores'] = {
            k: v['context_dependency_score'] 
            for k, v in dep_scores.items()
        }
        
    # Save summary
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Analysis complete! Results saved to {output_dir}")
    

if __name__ == '__main__':
    main()