#!/usr/bin/env python3
"""
Quick training script to train the enhanced ML models
Run this to train the invoice extraction models with 20+ iterations
"""

import sys
import os
from ml_trainer import main as train_main

def run_training():
    """Execute the training pipeline"""
    print("Starting Enhanced Invoice Extraction Training Pipeline")
    print("=" * 60)
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    try:
        # Run main training
        summary = train_main()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"✓ Total iterations: {summary['total_iterations']}")
        print(f"✓ Best model: {summary['best_model_name']}")
        print(f"✓ Best F1 score: {summary['best_f1_score']:.4f}")
        print(f"✓ Target achieved (F1 > 0.9): {summary['target_achieved']}")
        print(f"✓ Feature count: {summary['feature_count']}")
        print(f"✓ Training completed at: {summary['training_completed_at']}")
        
        print("\n📁 Training artifacts saved:")
        print("   - models/best_model_*.pkl")
        print("   - models/feature_extractor.pkl") 
        print("   - models/training_history.json")
        print("   - models/training_summary.json")
        
        print("\n🚀 Enhanced extraction is now available!")
        print("   The Flask app will automatically use the trained models.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("The application will continue to work with regex-only extraction.")
        return False

if __name__ == "__main__":
    success = run_training()
    sys.exit(0 if success else 1)