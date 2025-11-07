"""
Results analysis and visualization for Native Language Identification
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import LANGUAGES

def plot_feature_comparison(mfcc_results, hubert_results):
    """
    Plot comparison between MFCC and HuBERT features
    
    Args:
        mfcc_results (dict): Results from MFCC model
        hubert_results (dict): Results from HuBERT model
    """
    # Extract accuracies
    mfcc_acc = mfcc_results.get('accuracy', 0)
    hubert_acc = hubert_results.get('accuracy', 0)
    
    # Create bar plot
    features = ['MFCC', 'HuBERT']
    accuracies = [mfcc_acc, hubert_acc]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(features, accuracies, color=['skyblue', 'lightcoral'])
    plt.ylabel('Accuracy')
    plt.title('Feature Comparison: MFCC vs HuBERT')
    plt.ylim(0, 1)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/feature_comparison.png')
    plt.show()

def plot_language_performance(results, title="Language Performance"):
    """
    Plot performance for each language
    
    Args:
        results (dict): Classification results
        title (str): Plot title
    """
    # Extract per-language metrics
    languages = []
    precisions = []
    recalls = []
    f1_scores = []
    
    report = results.get('classification_report', {})
    for lang in LANGUAGES:
        if lang in report:
            languages.append(lang.capitalize())
            precisions.append(report[lang]['precision'])
            recalls.append(report[lang]['recall'])
            f1_scores.append(report[lang]['f1-score'])
    
    # Create grouped bar plot
    x = np.arange(len(languages))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, precisions, width, label='Precision', color='skyblue')
    rects2 = ax.bar(x, recalls, width, label='Recall', color='lightgreen')
    rects3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='salmon')
    
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(languages, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig(f'results/language_performance.png')
    plt.show()

def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix"):
    """
    Plot confusion matrix
    
    Args:
        cm (np.array): Confusion matrix
        class_names (list): Class names
        title (str): Plot title
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=class_names or range(len(cm)),
                yticklabels=class_names or range(len(cm)))
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.show()

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plot training history
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        train_accuracies (list): Training accuracies
        val_accuracies (list): Validation accuracies
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.show()

def generate_summary_report(mfcc_results, hubert_results, cross_age_results=None):
    """
    Generate a summary report of all results
    
    Args:
        mfcc_results (dict): Results from MFCC model
        hubert_results (dict): Results from HuBERT model
        cross_age_results (dict): Cross-age generalization results
    """
    print("=" * 50)
    print("NATIVE LANGUAGE IDENTIFICATION - SUMMARY REPORT")
    print("=" * 50)
    
    # Overall accuracy comparison
    mfcc_acc = mfcc_results.get('accuracy', 0)
    hubert_acc = hubert_results.get('accuracy', 0)
    
    print(f"\nOverall Accuracy:")
    print(f"  MFCC Features: {mfcc_acc:.4f}")
    print(f"  HuBERT Features: {hubert_acc:.4f}")
    print(f"  Improvement with HuBERT: {hubert_acc - mfcc_acc:.4f}")
    
    # Best performing language for each feature type
    print(f"\nBest Performing Languages:")
    
    mfcc_report = mfcc_results.get('classification_report', {})
    hubert_report = hubert_results.get('classification_report', {})
    
    if mfcc_report:
        mfcc_best = max([(lang, metrics['f1-score']) for lang, metrics in mfcc_report.items() 
                        if isinstance(metrics, dict) and 'f1-score' in metrics], 
                       key=lambda x: x[1])
        print(f"  MFCC: {mfcc_best[0].capitalize()} (F1: {mfcc_best[1]:.4f})")
    
    if hubert_report:
        hubert_best = max([(lang, metrics['f1-score']) for lang, metrics in hubert_report.items() 
                          if isinstance(metrics, dict) and 'f1-score' in metrics], 
                         key=lambda x: x[1])
        print(f"  HuBERT: {hubert_best[0].capitalize()} (F1: {hubert_best[1]:.4f})")
    
    # Cross-age generalization results
    if cross_age_results:
        print(f"\nCross-age Generalization:")
        mfcc_adult = cross_age_results.get('mfcc_adult_accuracy', 0)
        mfcc_child = cross_age_results.get('mfcc_child_accuracy', 0)
        hubert_adult = cross_age_results.get('hubert_adult_accuracy', 0)
        hubert_child = cross_age_results.get('hubert_child_accuracy', 0)
        
        print(f"  MFCC - Adult: {mfcc_adult:.4f}, Child: {mfcc_child:.4f}, Drop: {mfcc_adult - mfcc_child:.4f}")
        print(f"  HuBERT - Adult: {hubert_adult:.4f}, Child: {hubert_child:.4f}, Drop: {hubert_adult - hubert_child:.4f}")
    
    print("\n" + "=" * 50)

# Example usage (would be called with actual results)
if __name__ == "__main__":
    # This is just an example of how the analysis functions would be used
    # In practice, you would load actual results from training
    
    # Example placeholder results
    mfcc_results = {
        'accuracy': 0.72,
        'classification_report': {
            'hindi': {'precision': 0.75, 'recall': 0.70, 'f1-score': 0.72},
            'tamil': {'precision': 0.68, 'recall': 0.72, 'f1-score': 0.70},
            'telugu': {'precision': 0.74, 'recall': 0.69, 'f1-score': 0.71},
            'malayalam': {'precision': 0.69, 'recall': 0.75, 'f1-score': 0.72},
            'kannada': {'precision': 0.71, 'recall': 0.68, 'f1-score': 0.69},
            'punjabi': {'precision': 0.73, 'recall': 0.71, 'f1-score': 0.72},
            'bengali': {'precision': 0.67, 'recall': 0.70, 'f1-score': 0.68},
            'gujarati': {'precision': 0.70, 'recall': 0.67, 'f1-score': 0.68}
        }
    }
    
    hubert_results = {
        'accuracy': 0.84,
        'classification_report': {
            'hindi': {'precision': 0.86, 'recall': 0.83, 'f1-score': 0.84},
            'tamil': {'precision': 0.82, 'recall': 0.85, 'f1-score': 0.83},
            'telugu': {'precision': 0.85, 'recall': 0.82, 'f1-score': 0.83},
            'malayalam': {'precision': 0.83, 'recall': 0.87, 'f1-score': 0.85},
            'kannada': {'precision': 0.84, 'recall': 0.81, 'f1-score': 0.82},
            'punjabi': {'precision': 0.85, 'recall': 0.84, 'f1-score': 0.84},
            'bengali': {'precision': 0.81, 'recall': 0.83, 'f1-score': 0.82},
            'gujarati': {'precision': 0.83, 'recall': 0.80, 'f1-score': 0.81}
        }
    }
    
    # Generate plots
    plot_feature_comparison(mfcc_results, hubert_results)
    plot_language_performance(mfcc_results, "MFCC Language Performance")
    plot_language_performance(hubert_results, "HuBERT Language Performance")
    
    # Generate summary report
    generate_summary_report(mfcc_results, hubert_results)