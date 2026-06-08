"""
Explainable AI - Feature Importance and Model Interpretation
For SMOTE Balanced Random Forest Model
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class TheftExplainer:
    def __init__(self):
        """Load the SMOTE balanced model and files"""
        print("="*60)
        print("Loading SMOTE Balanced Model for Explainable AI")
        print("="*60)
        
        self.model = joblib.load('models/rf_balanced_smote.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        self.label_encoder = joblib.load('models/label_encoder.pkl')
        self.feature_names = joblib.load('models/feature_names.pkl')
        
        # Get feature importance from model
        self.feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"✅ Model loaded: Random Forest (SMOTE Balanced)")
        print(f"✅ Features: {len(self.feature_names)}")
        print(f"✅ Classes: {list(self.label_encoder.classes_)}")
        print("="*60)
    
    def get_feature_importance(self, top_n=10):
        """
        Get global feature importance
        
        Parameters:
        top_n: Number of top features to return
        
        Returns:
        DataFrame with feature names and importance scores
        """
        return self.feature_importance.head(top_n)
    
    def plot_feature_importance(self, top_n=8, save_path=None):
        """
        Plot feature importance bar chart
        
        Parameters:
        top_n: Number of top features to show
        save_path: Path to save the figure (optional)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        imp_df = self.feature_importance.head(top_n)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(imp_df)))
        
        bars = ax.barh(imp_df['Feature'], imp_df['Importance'] * 100, color=colors)
        ax.set_xlabel('Importance (%)', fontsize=12)
        ax.set_title('Feature Importance for Electricity Theft Detection', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for bar, val in zip(bars, imp_df['Importance'] * 100):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Figure saved: {save_path}")
        
        return fig
    
    def explain_prediction(self, input_values):
        """
        Explain a single prediction
        
        Parameters:
        input_values: List of 10 feature values
        
        Returns:
        Dictionary with prediction, confidence, and explanation
        """
        input_array = np.array([input_values])
        input_scaled = self.scaler.transform(input_array)
        
        # Get prediction
        prediction = self.model.predict(input_scaled)[0]
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        confidence = np.max(probabilities) * 100
        
        # Get top 3 predicted classes
        prob_df = pd.DataFrame({
            'Class': self.label_encoder.classes_,
            'Probability': probabilities * 100
        }).sort_values('Probability', ascending=False)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': prob_df,
            'top_3_classes': prob_df.head(3)['Class'].tolist(),
            'top_3_probabilities': prob_df.head(3)['Probability'].tolist()
        }
    
    def get_class_description(self, class_name):
        """Get description of theft type"""
        descriptions = {
            'Normal': "Normal consumption pattern - No theft detected",
            'Theft1': "Constant reduction (0.1-0.8x of actual consumption)",
            'Theft2': "Zero consumption during random periods",
            'Theft3': "Hourly random reduction (0.1-0.8x)",
            'Theft4': "Random fraction of mean consumption",
            'Theft5': "Reports mean consumption constantly"
        }
        return descriptions.get(class_name, "Unknown class")
    
    def generate_report(self, input_values):
        """
        Generate a complete explainability report for a prediction
        
        Parameters:
        input_values: List of 10 feature values
        
        Returns:
        Dictionary with complete explanation
        """
        explanation = self.explain_prediction(input_values)
        
        report = {
            'input_values': dict(zip(self.feature_names, input_values)),
            'prediction': explanation['predicted_class'],
            'confidence': explanation['confidence'],
            'prediction_description': self.get_class_description(explanation['predicted_class']),
            'top_features': self.get_feature_importance(5).to_dict('records'),
            'all_probabilities': explanation['all_probabilities'].to_dict('records'),
            'recommendation': self._get_recommendation(explanation)
        }
        
        return report
    
    def _get_recommendation(self, explanation):
        """Generate recommendation based on prediction and confidence"""
        if explanation['predicted_class'] == 'Normal':
            return "No action needed. Consumption pattern is normal."
        elif explanation['confidence'] > 80:
            return f"HIGH CONFIDENCE: Strong evidence of {explanation['predicted_class']}. Recommend immediate investigation."
        elif explanation['confidence'] > 50:
            return f"MODERATE CONFIDENCE: Possible {explanation['predicted_class']}. Schedule inspection."
        else:
            return f"LOW CONFIDENCE: Borderline case. Manual review recommended."


# Test the explainer
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING THEFT EXPLAINER")
    print("="*60)
    
    # Initialize explainer
    explainer = TheftExplainer()
    
    # Test cases
    test_cases = {
        "Normal": [22.04, 3.59, 0, 0, 4.59, 8.19, 136.59, 124.00, 3.34, 9.25],
        "Theft": [12.5, 3.2, 5.1, 2.8, 1.5, 6.2, 7.5, 4.2, 2.5, 3.5],
        "Mixed": [45.00, 12.00, 8.00, 5.00, 10.00, 25.00, 35.00, 20.00, 8.00, 10.00]
    }
    
    for name, values in test_cases.items():
        print(f"\n{'='*60}")
        print(f"Test Case: {name}")
        print(f"{'='*60}")
        
        result = explainer.explain_prediction(values)
        
        print(f"\n🔮 Prediction: {result['predicted_class']}")
        print(f"📊 Confidence: {result['confidence']:.1f}%")
        print(f"\n📋 Top 3 Predictions:")
        for cls, prob in zip(result['top_3_classes'], result['top_3_probabilities']):
            print(f"   {cls}: {prob:.1f}%")
    
    # Plot feature importance
    print("\n" + "="*60)
    print("Generating Feature Importance Plot...")
    print("="*60)
    
    fig = explainer.plot_feature_importance(save_path='results/feature_importance_smote.png')
    plt.close(fig)
    
    print("\n✅ TheftExplainer is ready to use!")
    print("\n📌 This explainer uses the SMOTE Balanced Model (94.60% accuracy)")