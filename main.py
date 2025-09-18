import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import re
import datetime
import time
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .emotion-result {
        font-size: 2rem;
        text-align: center;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .sidebar-content {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced emotion emoji dictionary with more emotions
emotion_emoji_dict = {
    "anger": "😡", "disgust": "🤮", "fear": "😱", "joy": "😄", 
    "neutral": "😐", "sad": "😢", "sadness": "😔", "shame": "😳", 
    "surprise": "😲", "love": "🥰", "excitement": "🤩", "confusion": "😕",
    "anticipation": "🤔", "trust": "😊", "optimism": "🌟", "pessimism": "😞"
}

# Color mapping for emotions
emotion_colors = {
    "anger": "#ff4444", "disgust": "#8e44ad", "fear": "#34495e", "joy": "#f1c40f",
    "neutral": "#95a5a6", "sad": "#3498db", "sadness": "#2980b9", "shame": "#e74c3c",
    "surprise": "#ff9f43", "love": "#e91e63", "excitement": "#ff6b6b", "confusion": "#f39c12"
}

@st.cache_resource
def load_model():
    """Load the emotion detection model with error handling"""
    try:
        model_path = r"C:/Users/omrew/OneDrive/Documents/project/Text emotion detection Using/model/text_emotion.pkl"
        pipe_lr = joblib.load(model_path)
        return pipe_lr, True
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please check the file path.")
        return None, False
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, False

def predict_emotion(docx, model):
    """Predict emotion from text"""
    if model is None:
        return "unknown"
    try:
        results = model.predict([docx])
        return results[0]
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "unknown"

def get_prediction_proba(docx, model):
    """Get prediction probabilities"""
    if model is None:
        return np.array([[0.1] * 8])  # Default probabilities
    try:
        results = model.predict_proba([docx])
        return results
    except Exception as e:
        st.error(f"Error getting probabilities: {str(e)}")
        return np.array([[0.1] * len(model.classes_)])

def analyze_text_statistics(text):
    """Analyze various text statistics"""
    blob = TextBlob(text)
    
    stats = {
        'word_count': len(text.split()),
        'char_count': len(text),
        'sentence_count': len(blob.sentences),
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0
    }
    return stats

def create_emotion_gauge(emotion, confidence, color):
    """Create a gauge chart for emotion confidence"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{emotion.title()} Confidence"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    fig.update_layout(height=300)
    return fig

def create_word_cloud(text):
    """Create word cloud from text"""
    try:
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        return fig
    except Exception as e:
        st.error(f"Error creating word cloud: {str(e)}")
        return None

def sentiment_over_time():
    """Show sentiment analysis over time for batch processing"""
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    
    if len(st.session_state.emotion_history) > 1:
        df_history = pd.DataFrame(st.session_state.emotion_history)
        
        fig = px.line(df_history, x='timestamp', y='confidence', 
                     color='emotion', title='Emotion Detection Over Time')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def batch_emotion_analysis(texts):
    """Analyze emotions for multiple texts"""
    model, model_loaded = load_model()
    if not model_loaded:
        return None
    
    results = []
    for i, text in enumerate(texts):
        if text.strip():
            emotion = predict_emotion(text, model)
            probabilities = get_prediction_proba(text, model)[0]
            confidence = np.max(probabilities)
            
            results.append({
                'text_id': i + 1,
                'text': text[:50] + "..." if len(text) > 50 else text,
                'emotion': emotion,
                'confidence': confidence,
                'emoji': emotion_emoji_dict.get(emotion, "❓")
            })
    
    return pd.DataFrame(results)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Advanced Text Emotion Detection</h1>', 
                unsafe_allow_html=True)
    
    # Load model
    model, model_loaded = load_model()
    
    if not model_loaded:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings & Information")
        
        # Mode selection
        analysis_mode = st.selectbox(
            "Choose Analysis Mode",
            ["Single Text Analysis", "Batch Text Analysis", "Live Text Analysis", "Text Comparison"]
        )
        
        # Model information
        if model:
            st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
            st.subheader("📊 Model Info")
            st.write(f"**Classes:** {len(model.classes_)}")
            st.write(f"**Available Emotions:**")
            for emotion in model.classes_:
                emoji = emotion_emoji_dict.get(emotion, "❓")
                st.write(f"• {emoji} {emotion.title()}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Statistics
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.subheader("📈 Session Stats")
        if 'total_analyses' not in st.session_state:
            st.session_state.total_analyses = 0
        st.metric("Total Analyses", st.session_state.total_analyses)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content based on mode
    if analysis_mode == "Single Text Analysis":
        single_text_analysis(model)
    elif analysis_mode == "Batch Text Analysis":
        batch_text_analysis(model)
    elif analysis_mode == "Live Text Analysis":
        live_text_analysis(model)
    elif analysis_mode == "Text Comparison":
        text_comparison_analysis(model)

def single_text_analysis(model):
    """Single text analysis with enhanced features"""
    st.subheader("📝 Single Text Analysis")
    
    # Text input options
    input_method = st.radio("Choose input method:", ["Text Area", "File Upload"])
    
    raw_text = ""
    if input_method == "Text Area":
        raw_text = st.text_area("Enter your text here:", height=150, 
                               placeholder="Type or paste your text here...")
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
        if uploaded_file is not None:
            raw_text = str(uploaded_file.read(), "utf-8")
            st.text_area("Uploaded text:", value=raw_text, height=150, disabled=True)
    
    if st.button("🔍 Analyze Emotion", type="primary") and raw_text.strip():
        st.session_state.total_analyses += 1
        
        with st.spinner("Analyzing emotion..."):
            # Get predictions
            prediction = predict_emotion(raw_text, model)
            probability = get_prediction_proba(raw_text, model)
            confidence = np.max(probability)
            
            # Text statistics
            text_stats = analyze_text_statistics(raw_text)
            
            # Store in history
            if 'emotion_history' not in st.session_state:
                st.session_state.emotion_history = []
            
            st.session_state.emotion_history.append({
                'timestamp': datetime.datetime.now(),
                'emotion': prediction,
                'confidence': confidence,
                'text_length': len(raw_text)
            })
        
        # Results layout
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.subheader("📊 Analysis Results")
            
            # Main emotion result
            emoji_icon = emotion_emoji_dict.get(prediction, "❓")
            color = emotion_colors.get(prediction, "#95a5a6")
            
            st.markdown(f"""
            <div class="emotion-result" style="background-color: {color}20; border-left: 5px solid {color};">
                <h2>{emoji_icon} {prediction.title()}</h2>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Text statistics
            st.subheader("📈 Text Statistics")
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.metric("Word Count", text_stats['word_count'])
                st.metric("Sentences", text_stats['sentence_count'])
                st.metric("Avg Word Length", f"{text_stats['avg_word_length']:.1f}")
            
            with stats_col2:
                st.metric("Characters", text_stats['char_count'])
                st.metric("Polarity", f"{text_stats['polarity']:.2f}")
                st.metric("Subjectivity", f"{text_stats['subjectivity']:.2f}")
        
        with col2:
            # Probability visualization
            st.subheader("🎯 Confidence Gauge")
            gauge_fig = create_emotion_gauge(prediction, confidence, color)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Probability bar chart
            st.subheader("📊 All Emotions")
            prob_df = pd.DataFrame(probability, columns=model.classes_)
            prob_df_clean = prob_df.T.reset_index()
            prob_df_clean.columns = ["emotion", "probability"]
            prob_df_clean = prob_df_clean.sort_values('probability', ascending=True)
            
            fig = px.bar(prob_df_clean, x='probability', y='emotion', 
                        orientation='h', color='emotion',
                        title="Probability Distribution")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.subheader("💡 Insights")
            
            # Emotion insights
            if confidence > 0.8:
                st.success("🎯 High confidence prediction!")
            elif confidence > 0.6:
                st.warning("⚠️ Moderate confidence")
            else:
                st.error("❓ Low confidence")
            
            # Sentiment insights
            if text_stats['polarity'] > 0.1:
                st.info("😊 Positive sentiment")
            elif text_stats['polarity'] < -0.1:
                st.info("😔 Negative sentiment")
            else:
                st.info("😐 Neutral sentiment")
            
            # Subjectivity insights
            if text_stats['subjectivity'] > 0.5:
                st.info("🗣️ Subjective text")
            else:
                st.info("📰 Objective text")
        
        # Word cloud
        st.subheader("☁️ Word Cloud")
        if len(raw_text.split()) > 3:
            wordcloud_fig = create_word_cloud(raw_text)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
        else:
            st.info("Need more words to generate word cloud")
        
        # Show emotion history
        if len(st.session_state.emotion_history) > 1:
            st.subheader("📈 Emotion Analysis History")
            sentiment_over_time()

def batch_text_analysis(model):
    """Batch analysis for multiple texts"""
    st.subheader("📚 Batch Text Analysis")
    
    batch_input = st.text_area(
        "Enter multiple texts (one per line):",
        height=200,
        placeholder="Text 1\nText 2\nText 3\n..."
    )
    
    if st.button("🔍 Analyze All Texts", type="primary") and batch_input.strip():
        texts = [text.strip() for text in batch_input.split('\n') if text.strip()]
        
        if texts:
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                results_df = batch_emotion_analysis(texts)
            
            if results_df is not None:
                st.session_state.total_analyses += len(texts)
                
                # Results display
                st.subheader("📊 Batch Analysis Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Texts", len(results_df))
                with col2:
                    st.metric("Avg Confidence", f"{results_df['confidence'].mean():.2%}")
                with col3:
                    most_common = results_df['emotion'].mode().iloc[0]
                    st.metric("Most Common Emotion", most_common.title())
                with col4:
                    high_conf = len(results_df[results_df['confidence'] > 0.8])
                    st.metric("High Confidence", f"{high_conf}/{len(results_df)}")
                
                # Results table
                st.subheader("📋 Detailed Results")
                display_df = results_df.copy()
                display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
                st.dataframe(display_df, use_container_width=True)
                
                # Emotion distribution
                st.subheader("📊 Emotion Distribution")
                emotion_counts = results_df['emotion'].value_counts()
                fig = px.pie(values=emotion_counts.values, names=emotion_counts.index,
                           title="Emotion Distribution in Batch")
                st.plotly_chart(fig, use_container_width=True)

def live_text_analysis(model):
    """Live text analysis with real-time updates"""
    st.subheader("🔴 Live Text Analysis")
    
    st.info("Type in the text area below and see real-time emotion analysis!")
    
    # Live text input
    live_text = st.text_area("Type here for live analysis:", 
                            key="live_text", height=150)
    
    if live_text and len(live_text.strip()) > 10:
        # Real-time prediction
        prediction = predict_emotion(live_text, model)
        probability = get_prediction_proba(live_text, model)
        confidence = np.max(probability)
        
        # Live results
        emoji_icon = emotion_emoji_dict.get(prediction, "❓")
        color = emotion_colors.get(prediction, "#95a5a6")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; border-radius: 15px; 
                       background: linear-gradient(135deg, {color}40, {color}20);">
                <h1>{emoji_icon}</h1>
                <h3>{prediction.title()}</h3>
                <p>Confidence: {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Live probability chart
            prob_df = pd.DataFrame(probability, columns=model.classes_)
            prob_df_clean = prob_df.T.reset_index()
            prob_df_clean.columns = ["emotion", "probability"]
            
            fig = px.bar(prob_df_clean, x='emotion', y='probability',
                        color='emotion', title="Live Emotion Probabilities")
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="live_chart")

def text_comparison_analysis(model):
    """Compare emotions between two texts"""
    st.subheader("⚖️ Text Comparison Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Text A:**")
        text_a = st.text_area("Enter first text:", key="text_a", height=150)
    
    with col2:
        st.write("**Text B:**")
        text_b = st.text_area("Enter second text:", key="text_b", height=150)
    
    if st.button("🔍 Compare Texts", type="primary") and text_a.strip() and text_b.strip():
        st.session_state.total_analyses += 2
        
        # Analyze both texts
        pred_a = predict_emotion(text_a, model)
        prob_a = get_prediction_proba(text_a, model)
        conf_a = np.max(prob_a)
        
        pred_b = predict_emotion(text_b, model)
        prob_b = get_prediction_proba(text_b, model)
        conf_b = np.max(prob_b)
        
        # Comparison results
        st.subheader("📊 Comparison Results")
        
        comp_col1, comp_col2, comp_col3 = st.columns([1, 1, 1])
        
        with comp_col1:
            emoji_a = emotion_emoji_dict.get(pred_a, "❓")
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; border-radius: 10px; 
                       background-color: #e3f2fd;">
                <h3>Text A</h3>
                <h1>{emoji_a}</h1>
                <h4>{pred_a.title()}</h4>
                <p>Confidence: {conf_a:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with comp_col2:
            if pred_a == pred_b:
                st.markdown("""
                <div style="text-align: center; padding: 1rem;">
                    <h3>🎯 Same Emotion!</h3>
                    <p>Both texts express the same emotion</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center; padding: 1rem;">
                    <h3>⚡ Different Emotions!</h3>
                    <p>Texts express different emotions</p>
                </div>
                """, unsafe_allow_html=True)
        
        with comp_col2:
            emoji_b = emotion_emoji_dict.get(pred_b, "❓")
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; border-radius: 10px; 
                       background-color: #fff3e0;">
                <h3>Text B</h3>
                <h1>{emoji_b}</h1>
                <h4>{pred_b.title()}</h4>
                <p>Confidence: {conf_b:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Side-by-side probability comparison
        st.subheader("📊 Probability Comparison")
        
        prob_df_a = pd.DataFrame(prob_a, columns=model.classes_).T
        prob_df_b = pd.DataFrame(prob_b, columns=model.classes_).T
        
        comparison_df = pd.DataFrame({
            'emotion': model.classes_,
            'text_a': prob_df_a.iloc[:, 0],
            'text_b': prob_df_b.iloc[:, 0]
        })
        
        fig = px.bar(comparison_df, x='emotion', y=['text_a', 'text_b'],
                    title="Emotion Probability Comparison", barmode='group')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()