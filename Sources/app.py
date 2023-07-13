import streamlit as st
# import pytorch
import random as random
from transformers import pipeline
from preprocessing_text import cleanData

button_style = '''
    <style>
        .stButton button {
            
            background-color: #0072B1;
            color: white;
            border-radius: 5px;
            font-weight: bold;
            padding: 8px 16px;
            box-shadow: none;
            
        }
        .stButton button:hover {
            color: white;
            background-color: #0072B1;
            box-shadow: none;
            border: none;
        }
    </style>'''

st.set_page_config(page_title="Sentiment Analysis", page_icon= "😊")

st.image('images/sentiment.png', width=300)

st.title("Sentiment Analysis")
st.markdown(button_style, unsafe_allow_html=True)


# Sidebar
st.sidebar.title("Team Information")
st.sidebar.write("Members:")
st.sidebar.write("- 19120501 - Nguyễn Nhật Hảo")
st.sidebar.write("- 20120033 - Võ Hoài An")
st.sidebar.write("- 20120041 - Trần Kim Bảo")
st.sidebar.write("- 20120084 - Nguyễn Văn Hiếu")
st.sidebar.write("- 20120116 - Phạm Lê Quốc Khánh")

# Tiêu đề nhập văn bản
st.header("Input Tweet")

# Ô input để nhập văn bản
context = st.text_area("Please type the tweet in the blank field")
context_processing=cleanData(context)

def to_sentiment(Sentiment):
    if Sentiment == 'LABEL_1':
        return 'Neutral'
    elif Sentiment == 'LABEL_2':
        return 'Positive'
    else:
        return 'Negative'
    

def emoji_path(status):
    if status == "Negative":
        return "images/negative.png"
    if status == "Positive":
        return "images/positive.png"
    else:
        return "images/neutral.png"

# Ô kết quả trả về đúng/sai
if st.button("Predict"):
    model_ckpt='../Model_saved'
    pipe=pipeline('sentiment-analysis',model=model_ckpt)

    result = to_sentiment(pipe(context_processing)[0]['label'])
    score = pipe(context_processing)[0]['score']

    col1, col2 = st.columns(2)
    with col1:
        col1.write('Result: ' +  result)
        col1.write('Score: ' + str(score))
    
    with col2:
        path = emoji_path(result)
        col2.image(path, width=150)

        

