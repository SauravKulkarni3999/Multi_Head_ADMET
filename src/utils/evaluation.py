"""
Comprehensive Evaluation Script for Multi-Task ADMET Model

This script provides robust evaluation metrics for classification and regression tasks
with proper error handling and detailed reporting.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import argparse
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ADMETEvaluator:
    """
    Comprehensive evaluator for multi-task ADMET models
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
    def evaluate_classification_task(self, predictions: np.ndarray, targets: np.ndarray, 
                                   task_name: str) -> Dict:
        """Evaluate classification task with comprehensive metrics"""
        try:
            # Basic metrics
            auc = roc_auc_score(targets, predictions)
            ap = average_precision_score(targets, predictions)
            
            # Convert to binary predictions
            binary_preds = (predictions > 0.5).astype(int)
            
            # Confusion matrix
            cm = confusion_matrix(targets, binary_preds)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # Additional metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            return {
                'task': task_name,
                'auc': auc,
                'average_precision': ap,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'n_samples': len(targets),
                'n_positive': int(targets.sum()),
                'n_negative': int(len(targets) - targets.sum())
            }
            
        except Exception as e:
            logger.error(f"Error evaluating classification task {task_name}: {e}")
            return {
                'task': task_name,
                'error': str(e),
                'n_samples': len(targets)
            }
    
    def evaluate_regression_task(self, predictions: np.ndarray, targets: np.ndarray, 
                               task_name: str) -> Dict:
        """Evaluate regression task with comprehensive metrics"""
        try:
            # Basic metrics
            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets, predictions)
            r2 = r2_score(targets, predictions)
            
            # Additional metrics
            mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
            
            return {
                'task': task_name,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'n_samples': len(targets),
                'target_mean': float(targets.mean()),
                'target_std': float(targets.std()),
                'pred_mean': float(predictions.mean()),
                'pred_std': float(predictions.std())
            }
            
        except Exception as e:
            logger.error(f"Error evaluating regression task {task_name}: {e}")
            return {
                'task': task_name,
                'error': str(e),
                'n_samples': len(targets)
            }
    
    def evaluate_model(self, data_loader, split_name: str = 'test') -> Dict:
        """Evaluate model on given data loader"""
        self.model.eval()
        
        all_predictions = {task: [] for task in self.model.all_tasks}
        all_targets = {task: [] for task in self.model.all_tasks}
        
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                masks = {k: v.to(self.device) for k, v in batch['masks'].items()}
                
                # Forward pass
                outputs = self.model(features, masks)
                
                # Collect predictions and targets
                for task in self.model.all_tasks:
                    if masks[task].sum() > 0:
                        mask = masks[task].bool()
                        
                        if task in self.model.classification_tasks:
                            pred = outputs[f'{task}_probs'][mask].cpu().numpy()
                        else:
                            pred = outputs[f'{task}_pred'][mask].cpu().numpy()
                        
                        target = targets[task][mask].cpu().numpy()
                        
                        all_predictions[task].append(pred)
                        all_targets[task].append(target)
        
        # Concatenate results
        for task in self.model.all_tasks:
            if all_predictions[task]:
                all_predictions[task] = np.concatenate(all_predictions[task])
                all_targets[task] = np.concatenate(all_targets[task])
        
        # Evaluate each task
        results = {}
        for task in self.model.all_tasks:
            if task in all_predictions and len(all_predictions[task]) > 0:
                if task in self.model.classification_tasks:
                    results[task] = self.evaluate_classification_task(
                        all_predictions[task], all_targets[task], task
                    )
                else:
                    results[task] = self.evaluate_regression_task(
                        all_predictions[task], all_targets[task], task
                    )
        
        return results
    
    def generate_report(self, results: Dict, output_dir: Path):
        """Generate comprehensive evaluation report"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary DataFrame
        summary_data = []
        for task, metrics in results.items():
            if 'error' not in metrics:
                row = {'task': task}
                row.update({k: v for k, v in metrics.items() if k not in ['confusion_matrix']})
                summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(output_dir / 'evaluation_summary.csv', index=False)
            
            # Print summary
            print("\n" + "="*80)
            print("EVALUATION SUMMARY")
            print("="*80)
            print(summary_df.to_string(index=False))
            
            # Generate plots
            self._generate_plots(results, output_dir)
        else:
            print("No valid results to report")
    
    def _generate_plots(self, results: Dict, output_dir: Path):
        """Generate evaluation plots"""
        # Task performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Classification metrics
        class_tasks = [task for task in results.keys() 
                      if task in self.model.classification_tasks and 'error' not in results[task]]
        if class_tasks:
            auc_scores = [results[task]['auc'] for task in class_tasks]
            axes[0, 0].bar(class_tasks, auc_scores)
            axes[0, 0].set_title('Classification AUC Scores')
            axes[0, 0].set_ylabel('AUC')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Regression metrics
        reg_tasks = [task for task in results.keys() 
                    if task in self.model.regression_tasks and 'error' not in results[task]]
        if reg_tasks:
            r2_scores = [results[task]['r2'] for task in reg_tasks]
            axes[0, 1].bar(reg_tasks, r2_scores)
            axes[0, 1].set_title('Regression R² Scores')
            axes[0, 1].set_ylabel('R²')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Loss comparison
        all_tasks = [task for task in results.keys() if 'error' not in results[task]]
        if all_tasks:
            if class_tasks:
                class_metrics = ['precision', 'recall', 'f1_score']
                class_data = [[results[task][metric] for task in class_tasks] 
                             for metric in class_metrics]
                axes[1, 0].boxplot(class_data, labels=class_metrics)
                axes[1, 0].set_title('Classification Metrics Distribution')
                axes[1, 0].set_ylabel('Score')
            
            if reg_tasks:
                reg_metrics = ['mse', 'mae', 'r2']
                reg_data = [[results[task][metric] for task in reg_tasks] 
                           for metric in reg_metrics]
                axes[1, 1].boxplot(reg_data, labels=reg_metrics)
                axes[1, 1].set_title('Regression Metrics Distribution')
                axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate ADMET Multi-Task Model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to processed data')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Output directory for evaluation reports')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Model configuration file')
    
    args = parser.parse_args()
    
    # Load model and data
    try:
        from src.models.admet_model import MultiTaskADMETPredictor
        from src.data.preprocessor import ADMETPreprocessor
        from src.data.data_loader import MultiTaskDataLoader
        
        # Load preprocessor and data
        preprocessor = ADMETPreprocessor()
        preprocessor.load_processed_data()
        splits = preprocessor.create_data_splits()
        
        # Create data loader
        data_loader = MultiTaskDataLoader(preprocessor, batch_size=128)
        loaders = data_loader.create_data_loaders(splits)
        
        # Load model
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model = MultiTaskADMETPredictor(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        evaluator = ADMETEvaluator(model)
        results = evaluator.evaluate_model(loaders['test'], 'test')
        
        # Generate report
        output_dir = Path(args.output_dir)
        evaluator.generate_report(results, output_dir)
        
        print(f"\nEvaluation completed! Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()