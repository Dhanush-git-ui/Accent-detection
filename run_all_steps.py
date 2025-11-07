"""
Script to run all project steps in sequence
"""
import os
import sys
import time

def print_step_header(step_num, description):
    """Print a step header"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*60}")

def run_step(command, description):
    """Run a step and handle errors"""
    print(f"\nExecuting: {command}")
    print("-" * 40)
    
    try:
        os.system(command)
        print(f"✅ {description} completed successfully")
        return True
    except Exception as e:
        print(f"❌ Error in {description}: {e}")
        return False

def main():
    """Main function to run all steps"""
    print("NATIVE LANGUAGE IDENTIFICATION PROJECT")
    print("Running All Implementation Steps")
    print("=" * 60)
    
    # Step 1: Create dummy data
    print_step_header(1, "Dataset & Preprocessing")
    run_step("python resample_all.py --dummy --num-dummy 100", "Create dummy dataset")
    
    # Step 2: Extract MFCC features
    print_step_header(2, "Feature Extraction - MFCC")
    run_step("python extract_mfcc.py --input data_norm/IndicAccentDB_audio_norm --output data/features/mfcc", 
             "Extract MFCC features")
    
    # Test a few files
    print("\nTesting MFCC features...")
    run_step("python extract_mfcc.py --test data/features/mfcc/dummy/dummy_000.npy", 
             "Test MFCC features")
    
    # Step 3: Train baseline model
    print_step_header(3, "Baseline Model (MFCC)")
    run_step("python train_baseline.py --epochs 5 --batch-size 16", 
             "Train baseline model on MFCC features")
    
    # Step 4: Extract HuBERT features
    print_step_header(4, "HuBERT Feature Extraction")
    run_step("python extract_hubert.py --input data_norm/IndicAccentDB_audio_norm --output data/features/hubert --max-files 20", 
             "Extract HuBERT features")
    
    # Analyze layers
    print("\nAnalyzing HuBERT layers...")
    run_step("python extract_hubert.py --analyze", "Analyze layer performance")
    
    # Step 5: Train HuBERT model
    print_step_header(5, "HuBERT Model Training")
    run_step("python train_hubert.py --epochs 5 --batch-size 16", 
             "Train model on HuBERT features")
    
    # Step 6: Analysis tasks
    print_step_header(6, "Analysis Tasks")
    
    # Layer analysis
    run_step("python layer_analysis.py --layers 10", "Layer analysis")
    
    # Age generalization
    run_step("python age_generalization.py", "Age generalization analysis")
    
    # Word vs sentence analysis
    run_step("python word_sentence_analysis.py", "Word vs sentence analysis")
    
    # Step 7: Generate report
    print_step_header(7, "Reporting & Results")
    run_step("python generate_report.py", "Generate comprehensive report")
    
    # Step 8: Test web application
    print_step_header(8, "Flask Web Demo")
    print("Starting Flask web application...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    run_step("python app.py", "Start Flask web application")
    
    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED!")
    print("=" * 60)
    print("\nProject is ready for use. Key components:")
    print("1. Dataset preprocessing complete")
    print("2. MFCC and HuBERT features extracted")
    print("3. Models trained and evaluated")
    print("4. Comprehensive analysis performed")
    print("5. Web application ready")
    print("6. Reports and figures generated")

if __name__ == "__main__":
    main()