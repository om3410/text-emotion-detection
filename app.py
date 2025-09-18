import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Fixed: load instead of loa, proper file path format, proper parentheses
pipe_lr = joblib.load(r"C:/Users/omrew/OneDrive/Documents/project/Text emotion detection Using/model/text_emotion.pkl")

# Fixed: dictionary name typo
emotion_emoji_dict = {"anger": "😡", "disgust": "🤮", "fear": "😱", "joy": "😂", "neutral": "😐", "sad":"😢",
                      "sadness": "😔", "shame": "🫣", "surprise": "😲"}

def predict_emotion(docx):
    results = pipe_lr.predict([docx])
    return results[0]  # Fixed: square brackets instead of parentheses

def get_prediction_proba(docx):  # Fixed: typo in function name
    results = pipe_lr.predict_proba([docx])  # Fixed: missing _proba and added input parameter
    return results

def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect emotion in text")
    
    with st.form("my_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')  # Fixed: typo in variable name
        
    if submit_text:  # Fixed: removed parentheses - it's a boolean, not a function
        col1, col2 = st.columns(2)
        
        prediction = predict_emotion(raw_text)
        probability = get_prediction_proba(raw_text)  # Fixed: function name typo
        
        with col1:
            st.success("Original Text")
            st.write(raw_text)
            
            st.success("Prediction")  # Fixed: typo
            emoji_icon = emotion_emoji_dict[prediction]  # Fixed: square brackets for dictionary access
            
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))  # Fixed: proper format syntax
            
        with col2:
            st.success("Prediction Probability")
            prob_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            prob_df_clean = prob_df.T.reset_index()
            prob_df_clean.columns = ["emotion", "probability"]  # Fixed: proper column assignment
            
            fig = alt.Chart(prob_df_clean).mark_bar().encode(x='emotion', y='probability', color='emotion')  # Fixed: column names and color parameter
            st.altair_chart(fig, use_container_width=True)
            
if __name__ == '__main__':
    main()