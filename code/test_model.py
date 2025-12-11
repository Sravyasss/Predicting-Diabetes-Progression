"""
Test Script - Use trained model on new patient data
"""

import joblib
import pandas as pd
import numpy as np

def test_patients_from_csv(csv_file):
    """
    Test the trained model on patients from CSV file
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with patient data
    """
    
    print("="*80)
    print("TESTING MODEL ON NEW PATIENTS")
    print("="*80)
    
    # Load the trained model
    try:
        model = joblib.load('output/best_model_optimized.pkl')
        scaler = joblib.load('output/scaler_optimized.pkl')
        label_encoder = joblib.load('output/label_encoder_optimized.pkl')
        print("✓ Loaded trained model")
    except:
        print("✗ Error: Model files not found!")
        print("Make sure you've run diabetes_optimized.py first")
        return
    
    # Load selected features
    try:
        with open('output/selected_features.txt', 'r') as f:
            selected_features = [line.strip() for line in f]
        print(f"✓ Model uses {len(selected_features)} features")
    except:
        print("✗ Error: selected_features.txt not found!")
        return
    
    # Load patient data
    patients_df = pd.read_csv(csv_file)
    print(f"\n✓ Loaded {len(patients_df)} patients from {csv_file}")
    
    print("\n--- Patient Data Preview ---")
    print(patients_df.head())
    
    # Check which features are missing
    available_features = patients_df.columns.tolist()
    missing_features = set(selected_features) - set(available_features)
    extra_features = set(available_features) - set(selected_features)
    
    if missing_features:
        print(f"\n⚠ Missing features (will be set to 0): {missing_features}")
        for feature in missing_features:
            patients_df[feature] = 0
    
    if extra_features:
        print(f"\n⚠ Extra features (will be ignored): {extra_features}")
    
    # Select only the features the model expects
    patients_df = patients_df[selected_features]
    
    # Scale the data
    patients_scaled = scaler.transform(patients_df)
    
    # Make predictions
    predictions_encoded = model.predict(patients_scaled)
    probabilities = model.predict_proba(patients_scaled)
    
    # Decode predictions
    if label_encoder:
        predictions = label_encoder.inverse_transform(predictions_encoded)
        class_names = label_encoder.classes_
    else:
        predictions = predictions_encoded
        class_names = [str(i) for i in range(len(probabilities[0]))]
    
    # Display results
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    results = []
    for i in range(len(patients_df)):
        max_prob = np.max(probabilities[i]) * 100
        
        result = {
            'Patient_ID': i + 1,
            'Prediction': predictions[i],
            'Confidence': f"{max_prob:.1f}%"
        }
        
        # Add key features for this patient
        if 'age' in patients_df.columns:
            result['Age'] = patients_df.iloc[i]['age']
        if 'bmi' in patients_df.columns:
            result['BMI'] = patients_df.iloc[i]['bmi']
        if 'family_history_diabetes' in patients_df.columns:
            result['Family_History'] = patients_df.iloc[i]['family_history_diabetes']
        
        results.append(result)
        
        # Print detailed info for each patient
        print(f"\n--- Patient {i+1} ---")
        print(f"Prediction: {predictions[i]}")
        print(f"Confidence: {max_prob:.1f}%")
        print(f"Probability breakdown:")
        for cls, prob in zip(class_names, probabilities[i]):
            bar = '█' * int(prob * 30)
            print(f"  {cls:20s}: {prob*100:5.1f}% {bar}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('prediction_results.csv', index=False)
    print("\n✓ Saved results to prediction_results.csv")
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print("\nPrediction Distribution:")
    print(pd.Series(predictions).value_counts())
    
    print("\nAverage Confidence by Prediction:")
    for pred in set(predictions):
        mask = predictions == pred
        avg_conf = np.mean([probabilities[i][predictions_encoded[i]] 
                           for i in range(len(predictions)) if mask[i]])
        print(f"  {pred}: {avg_conf*100:.1f}%")
    
    return results_df


if __name__ == "__main__":
    # Test on the dummy data
    test_patients_from_csv('test_patients.csv')
