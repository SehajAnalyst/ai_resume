"""
CareerFit AI - Model Training Script
Run once to pre-train and save the ML model.
Usage:  python train_model.py
"""

from core import generate_synthetic_dataset, train_model

if __name__ == "__main__":
    print("Generating synthetic dataset…")
    df = generate_synthetic_dataset(n=2000)
    print(f"Dataset shape: {df.shape}")
    print(df["category"].value_counts())

    print("\nTraining Random Forest Classifier…")
    model, le, accuracy = train_model(df)

    print(f"\n✅ Model trained successfully!")
    print(f"   Test Accuracy : {accuracy}%")
    print(f"   Classes       : {list(le.classes_)}")
    print(f"   Saved to      : rf_model.pkl & label_encoder.pkl")

    # Save dataset for reference
    df.to_csv("synthetic_resume_dataset.csv", index=False)
    print(f"   Dataset saved : synthetic_resume_dataset.csv")
