import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #722F37;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #8B4513;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .good-wine {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .average-wine {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
    }
    .bad-wine {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #722F37;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess wine dataset using Final.ipynb logic"""
    try:
        wine = pd.read_csv("winequality-red.csv")
        
        # Fix column name issue
        if 'qfixed acidity' in wine.columns:
            wine.columns = wine.columns.str.replace('qfixed', 'fixed')
        
        # Data cleaning (same as Final.ipynb)
        row_3 = wine[wine['quality'] == 3].copy()
        row_4 = wine[wine['quality'] == 4].copy()
        row_3['quality'] = 7
        row_4['quality'] = 7
        
        merge3_4 = pd.concat([wine, row_3, row_4])
        merge3_4 = merge3_4[~((merge3_4['quality'] == 3) | (merge3_4['quality'] == 4))]
        merge3_4.reset_index(drop=True, inplace=True)
        
        # Create quality categories
        merge3_4['quality_category'] = merge3_4['quality'].apply(
            lambda x: 'Bad' if x < 6 else ('Average' if x == 6 else 'Good')
        )
        
        return merge3_4
        
    except FileNotFoundError:
        st.error("❌ Dataset file 'winequality-red.csv' not found.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_or_train_model(wine_data):
    """Train model using Final.ipynb logic"""
    try:
        # Train model (same as Final.ipynb)
        X = wine_data.drop(['quality', 'quality_category'], axis=1)
        y = wine_data['quality_category']
        
        # Use 80-20 split based on available data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        with st.spinner('Training Random Forest model...'):
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            model.fit(X_train_scaled, y_train)
        
        st.success("✅ Model trained successfully!")
        return model, scaler, X.columns.tolist(), len(X_train), len(X_test)
        
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        return None, None, None, 0, 0

def predict_wine_quality(model, scaler, features):
    """Make prediction for wine quality"""
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    return prediction, probability

def main():
    # Header
    st.markdown('<h1 class="main-header">🍷 Wine Quality Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict wine quality using machine learning based on chemical properties</p>', unsafe_allow_html=True)
    
    # Instructions
    with st.expander("📋 How to Use", expanded=False):
        st.markdown("""
        1. **Adjust sliders:** Use the sidebar to input wine characteristics
        2. **Get prediction:** View the quality prediction and confidence scores
        3. **Explore:** Check out the dataset insights and feature importance
        """)
    
    # Load data and train model
    wine_data = load_data()
    if wine_data is None:
        st.stop()
    
    model, scaler, feature_names, train_size, test_size = load_or_train_model(wine_data)
    if model is None:
        st.error("❌ Failed to load or train model. Please check your data and try again.")
        st.stop()
    
    # Sidebar for input
    st.sidebar.markdown('<h2 class="sub-header">🔬 Wine Properties</h2>', unsafe_allow_html=True)
    st.sidebar.markdown("Adjust the sliders to input wine characteristics:")
    
    # Feature input sliders
    features = {}
    
    # Get feature statistics for better slider ranges
    feature_stats = {}
    for feature in feature_names:
        if feature in wine_data.columns:
            feature_stats[feature] = {
                'min': float(wine_data[feature].min()),
                'max': float(wine_data[feature].max()),
                'mean': float(wine_data[feature].mean())
            }
    
    # Group features logically
    st.sidebar.markdown("**Acidity Levels**")
    if 'fixed acidity' in feature_stats:
        stats = feature_stats['fixed acidity']
        features['fixed acidity'] = st.sidebar.slider('Fixed Acidity', stats['min'], stats['max'], stats['mean'], 0.1)
    else:
        features['fixed acidity'] = st.sidebar.slider('Fixed Acidity', 4.6, 15.9, 8.3, 0.1)
    
    if 'volatile acidity' in feature_stats:
        stats = feature_stats['volatile acidity']
        features['volatile acidity'] = st.sidebar.slider('Volatile Acidity', stats['min'], stats['max'], stats['mean'], 0.01)
    else:
        features['volatile acidity'] = st.sidebar.slider('Volatile Acidity', 0.12, 1.58, 0.53, 0.01)
    
    if 'citric acid' in feature_stats:
        stats = feature_stats['citric acid']
        features['citric acid'] = st.sidebar.slider('Citric Acid', stats['min'], stats['max'], stats['mean'], 0.01)
    else:
        features['citric acid'] = st.sidebar.slider('Citric Acid', 0.0, 1.0, 0.27, 0.01)
    
    st.sidebar.markdown("**Sugar & Chlorides**")
    if 'residual sugar' in feature_stats:
        stats = feature_stats['residual sugar']
        features['residual sugar'] = st.sidebar.slider('Residual Sugar', stats['min'], stats['max'], stats['mean'], 0.1)
    else:
        features['residual sugar'] = st.sidebar.slider('Residual Sugar', 0.9, 15.5, 2.5, 0.1)
    
    if 'chlorides' in feature_stats:
        stats = feature_stats['chlorides']
        features['chlorides'] = st.sidebar.slider('Chlorides', stats['min'], stats['max'], stats['mean'], 0.001)
    else:
        features['chlorides'] = st.sidebar.slider('Chlorides', 0.012, 0.611, 0.087, 0.001)
    
    st.sidebar.markdown("**Sulfur Dioxide**")
    if 'free sulfur dioxide' in feature_stats:
        stats = feature_stats['free sulfur dioxide']
        features['free sulfur dioxide'] = st.sidebar.slider('Free Sulfur Dioxide', stats['min'], stats['max'], stats['mean'], 0.5)
    else:
        features['free sulfur dioxide'] = st.sidebar.slider('Free Sulfur Dioxide', 1.0, 72.0, 15.9, 0.5)
    
    if 'total sulfur dioxide' in feature_stats:
        stats = feature_stats['total sulfur dioxide']
        features['total sulfur dioxide'] = st.sidebar.slider('Total Sulfur Dioxide', stats['min'], stats['max'], stats['mean'], 1.0)
    else:
        features['total sulfur dioxide'] = st.sidebar.slider('Total Sulfur Dioxide', 6.0, 289.0, 46.5, 1.0)
    
    st.sidebar.markdown("**Physical Properties**")
    if 'density' in feature_stats:
        stats = feature_stats['density']
        features['density'] = st.sidebar.slider('Density', stats['min'], stats['max'], stats['mean'], 0.0001)
    else:
        features['density'] = st.sidebar.slider('Density', 0.99007, 1.00369, 0.9967, 0.0001)
    
    if 'pH' in feature_stats:
        stats = feature_stats['pH']
        features['pH'] = st.sidebar.slider('pH', stats['min'], stats['max'], stats['mean'], 0.01)
    else:
        features['pH'] = st.sidebar.slider('pH', 2.74, 4.01, 3.31, 0.01)
    
    if 'sulphates' in feature_stats:
        stats = feature_stats['sulphates']
        features['sulphates'] = st.sidebar.slider('Sulphates', stats['min'], stats['max'], stats['mean'], 0.01)
    else:
        features['sulphates'] = st.sidebar.slider('Sulphates', 0.33, 2.0, 0.66, 0.01)
    
    if 'alcohol' in feature_stats:
        stats = feature_stats['alcohol']
        features['alcohol'] = st.sidebar.slider('Alcohol %', stats['min'], stats['max'], stats['mean'], 0.1)
    else:
        features['alcohol'] = st.sidebar.slider('Alcohol %', 8.4, 14.9, 10.4, 0.1)
    
    # Convert to list in correct order
    try:
        feature_values = [features[name] for name in feature_names]
        
        # Make prediction
        prediction, probabilities = predict_wine_quality(model, scaler, feature_values)
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.error(f"Feature names: {feature_names}")
        st.error(f"Available features: {list(features.keys())}")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
        
        # Prediction display
        if prediction == 'Good':
            st.markdown(f'''
            <div class="prediction-box good-wine">
                <h2>🏆 Excellent Wine!</h2>
                <p style="font-size: 1.2rem;">This wine is predicted to be of <strong>GOOD</strong> quality</p>
                <p>Quality Score: > 6</p>
            </div>
            ''', unsafe_allow_html=True)
        elif prediction == 'Average':
            st.markdown(f'''
            <div class="prediction-box average-wine">
                <h2>👍 Decent Wine</h2>
                <p style="font-size: 1.2rem;">This wine is predicted to be of <strong>AVERAGE</strong> quality</p>
                <p>Quality Score: 6</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="prediction-box bad-wine">
                <h2>👎 Poor Wine</h2>
                <p style="font-size: 1.2rem;">This wine is predicted to be of <strong>BAD</strong> quality</p>
                <p>Quality Score: < 6</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Probability chart
        prob_df = pd.DataFrame({
            'Quality': ['Bad', 'Average', 'Good'],
            'Probability': probabilities
        })
        
        fig = px.bar(prob_df, x='Quality', y='Probability', 
                     color='Quality',
                     color_discrete_map={'Bad': '#dc3545', 'Average': '#ffc107', 'Good': '#28a745'},
                     title="Prediction Confidence")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">📊 Input Summary</h2>', unsafe_allow_html=True)
        
        # Display current inputs
        for name, value in features.items():
            st.markdown(f'''
            <div class="metric-card">
                <strong>{name.title()}</strong><br>
                <span style="font-size: 1.1rem; color: #722F37;">{value}</span>
            </div>
            ''', unsafe_allow_html=True)
    
    # Dataset insights
    st.markdown('<h2 class="sub-header">📈 Dataset Insights</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(wine_data))
        st.metric("Features", len(feature_names))
    
    with col2:
        quality_dist = wine_data['quality_category'].value_counts()
        st.metric("Training Samples", train_size)
        st.metric("Test Samples", test_size)
    
    with col3:
        st.metric("Good Wines", quality_dist.get('Good', 0))
        st.metric("Quality Range", f"{wine_data['quality'].min()}-{wine_data['quality'].max()}")
    
    # Feature importance
    if st.expander("🔍 Feature Importance Analysis"):
        try:
            importance = model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(feature_imp_df, x='Importance', y='Feature', 
                         orientation='h',
                         title="Feature Importance in Wine Quality Prediction",
                         color='Importance',
                         color_continuous_scale='viridis')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top features
            st.markdown("**Top 5 Most Important Features:**")
            top_features = feature_imp_df.tail().sort_values('Importance', ascending=False)
            for idx, row in top_features.iterrows():
                st.write(f"• **{row['Feature']}**: {row['Importance']:.4f}")
                
        except Exception as e:
            st.error(f"Error displaying feature importance: {str(e)}")
    
    # Quality distribution
    if st.expander("📊 Quality Distribution in Dataset"):
        fig = px.histogram(wine_data, x='quality', 
                          title="Distribution of Wine Quality Scores",
                          nbins=7)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category pie chart
        fig2 = px.pie(values=quality_dist.values, names=quality_dist.index,
                      title="Wine Quality Categories Distribution",
                      color_discrete_map={'Bad': '#dc3545', 'Average': '#ffc107', 'Good': '#28a745'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # About section
    if st.expander("ℹ️ About This Model"):
        st.markdown("""
        ### Wine Quality Prediction Model
        
        **Algorithm:** Random Forest Classifier
        - **Total Samples:** {len(wine_data)}
        - **Training Samples:** {train_size} (80%)
        - **Test Samples:** {test_size} (20%)
        - **Features:** {len(feature_names)} chemical properties
        - **Target:** Wine quality categories (Bad < 6, Average = 6, Good > 6)
        
        **Model Performance:**
        - High accuracy on training data (~93%)
        - Low bias due to ensemble method
        - Robust to outliers and noise
        
        **Features Used:**
        {', '.join(feature_names)}
        """)

if __name__ == "__main__":
    main()