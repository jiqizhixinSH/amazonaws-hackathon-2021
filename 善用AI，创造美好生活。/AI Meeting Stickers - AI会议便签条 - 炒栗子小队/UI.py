import streamlit as st
import numpy as np
import pandas as pd

import pathlib
import matplotlib.pyplot as plt
import pickle
import os
import altair as alt
import boto3
import numpy as np
import en_core_web_sm
import spacy

from typing import List, Sequence, Tuple, Optional, Dict, Union, Callable
from FileControl import *
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from urllib.error import URLError
from urllib.request import urlopen
from datetime import datetime
from engine import engine,loadEngine,saveEngine
from utilities import *
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spacy import displacy
#from .util import load_model, process_text, get_svg, get_html, get_color_styles, LOGO


stop_words = ENGLISH_STOP_WORDS.union(word for word in ['City','San','Jose','Council','Report','motion','Item','Council','Councilmember','Title','Service','DISTRICT','Page','Action','Section','Project','File','appointment','approval','Manager','PUBLIC','Minutes','fee','funding','Amend','provided','Agreement','staff','S','services','changes','the City','Amendment to','of San','City of'])

NER_ATTRS = ["text", "label_", "start", "end", "start_char", "end_char"]
TOKEN_ATTRS = ["idx", "text", "lemma_", "pos_", "tag_", "dep_", "head", "morph",
               "ent_type_", "ent_iob_", "shape_", "is_alpha", "is_ascii",
               "is_digit", "is_punct", "like_num", "is_sent_start"]
# fmt: on
FOOTER = """<span style="font-size: 0.75em">&hearts; Built with [`spacy-streamlit`](https://github.com/explosion/spacy-streamlit)</span>"""


# preload content before the main 
def init():
    # Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)
    if os.path.exists('nlp_engine.pkl'):
        real_engine = loadEngine('nlp_engine.pkl')
    # Load style
    #st.set_page_config(layout="wide")
    st.markdown('<style>' + open('UI-styles.css').read() + '</style>', unsafe_allow_html=True)
    
    return real_engine
# main 
def main(obj_engine):
    # Add selector for the App states
    st.sidebar.title("Control Panel")
    app_mode = st.sidebar.selectbox("Choose your action",
        ["Home","View Meeting Stickers","View Original Minutes","Upload Original Minutes"])
    st.sidebar.success('default core loaded')
    
    # App states control
    # Home and landing page
    if app_mode == "Home":
        landing(obj_engine)
        
    # For checking the source code of the main application
    elif app_mode == "View Original Minutes":
        # readme_text.empty()
        view_original(obj_engine)
    
    # For viewing meeting content
    elif app_mode == "View Meeting Stickers":
        # readme_text.empty()
        view_meeting(obj_engine)
        
    # For loading meeting content
    elif app_mode == "Upload Original Minutes":
        # readme_text.empty()
        content_control(obj_engine)

def get_file_content(filename):
    with open(filename) as infile:
        return infile.read()

def landing(obj_engine):
    
    col1, col2 = st.beta_columns(2)
    # Render the main screen with read me instruction in markdown format
    readme_text = get_file_content("README.md")
    #specialtext = 'this changes all the time'
    col1.markdown(readme_text, unsafe_allow_html=True)
    
    # right side wordcloud
    loadWordCloud(obj_engine.getText(),stop_words,obj = col2,width = 800,height = 1600,max_words=200)
    
    # overall content status update
    statusDict = statSum(obj_engine)
    statusKeys = statusDict.keys()
    status_text = ''
    for k,c in statusDict.items():
        status_text += f"{k} - {c} <br>"
    summary_text = f'''
    ## Record Timeframe:
        {min(obj_engine.chronicle).astype('datetime64[D]')} to {max(obj_engine.chronicle).astype('datetime64[D]')}
    ## Content Summary:
    {status_text}
    '''
    st.sidebar.markdown(summary_text,unsafe_allow_html = True)

npdate2date = lambda x: datetime.utcfromtimestamp(x.astype(datetime)*1e-9)

def view_meeting(obj_engine):
    # get start and end date for timeframe filter
    startdate = npdate2date(min(obj_engine.chronicle))
    enddate = npdate2date(max(obj_engine.chronicle))
    
    # setup sidebar
    keywords = st.sidebar.text_input('enter your keywords, seperate with comma')
    (date_start,date_end) = st.sidebar.slider('what time frame?:', startdate, enddate, (startdate,enddate),key = ('date_start','date_end'))
    hasDollar = st.sidebar.checkbox('contains dollar values')
    result = obj_engine.searchKeywords(keywords,date_start,date_end,hasDollar)
    
    # the main body header
    st.markdown('<style>' + open('UI-styles.css').read() + '</style>', unsafe_allow_html=True)
    
    nlp = en_core_web_sm.load()
    
    col1,col2 = st.beta_columns(2)
    
    if result is not None:
        
        (notes,topics,keywords_LS) = result
        Body_html = f'''
        **keywords**: {keywords}<BR>
        **time frame**:{date_start.date()} to {date_end.date()}<BR>
        **dollar**: {hasDollar}<BR>
        **No of results**:{notes.shape[0]}<BR>
        '''
        st.sidebar.markdown(Body_html, unsafe_allow_html=True)
        selector_dict={}

        #with st.empty() as holder:
        #    holder.write(notes[['index','content','date','hasDollar']])

        for i,n in notes.iterrows():
            
            stickerText = cleanSticker('\n\n',n['content']).replace('\n',' ')
            contentSize = len(stickerText.split(" "))
            stopint = stickerText.find('. ')+1
            
            summary = f'<b>{n["filename"]} - {n["date"].date()}</b><BR> \
            <b>section:</b> {n["mainID"]}.{n["subID"]}<BR> \
            <b>time to read:</b>{round(contentSize/250)} - {round(contentSize/200)} minutes<BR> \
            <b>summary:</b>  {stickerText[:stopint]}'
            
            doc = nlp(stickerText)
            col1.markdown(f'<details><summary>{summary}</summary></details><BR>',unsafe_allow_html=True)
            selector_dict[i] = col1.button('read more >>',key=str(i))
        
        selected_id = [k for k,c in selector_dict.items() if c]
        
        if selected_id != []:
            stickerText = cleanSticker('\n\n',notes[notes['index']==selected_id[0]]['content'].iloc[0])
            doc = nlp(stickerText)
            visualize_ner(doc,labels=nlp.get_pipe("ner").labels,obj = col2)

    else:
        col1.markdown("Please enter keywords to start<BR>View searched result here.", unsafe_allow_html=True)
        col2.markdown("View detailed content highlight here.",unsafe_allow_html=True)

def cleanSticker(cleanText,text):
    text = text.replace(cleanText,'')
    if cleanText in text:
        cleanSticker(cleanText,text)
    else:
        return text

# ready
def view_original(obj_engine):
    
    col1, col2 = st.beta_columns(2)
    fileLS = obj_engine.df['filename'].unique().tolist()
    filename = st.sidebar.selectbox("Choose your file",fileLS)
    
    stop_words = ENGLISH_STOP_WORDS.union(word for word in ['City','San','Jose','Council','Report','motion','Item','Council','Councilmember','Title','Service','DISTRICT','Page','Action','Section','Project','File','appointment','approval','Manager','PUBLIC','Minutes','fee','funding','Amend','provided','Agreement','staff','S','a','b','c','d','services','changes','the City','Amendment to','of San','City of'])
    fileText = obj_engine.getOriginal(filename)
    col2.title('WordCloud')
    loadWordCloud(fileText,stop_words,width = 800,height = 1600,obj = col2,max_words=200)
    main_text = f'''
    # Show file: **{filename}**
    
    {fileText}
    '''
    col1.markdown(main_text,unsafe_allow_html=True)
    
# Issue, the control for upload was gone?
def content_control(obj_engine):
    uploaded_file = st.sidebar.file_uploader("Upload your meeting minutes file", type=["PDF"])
    
    if uploaded_file is not None:
        engineName = 'nlp_engine.pkl'
        
        try:
            obj_engine.addContent(uploaded_file)
        except:
            st.markdown('Server is busy, please try again later', unsafe_allow_html=True)
        '''
        saveEngine(engineName,obj_engine)
        s3 = boto3.client('s3')
        s3.upload_file(
            engineName, 'stickers-lambda',engineName,
            ExtraArgs={'ACL': 'public-read'}
        )
        '''
        EXTERNAL_DEPENDENCIES[engineName]['date']=datetime.today()
        
        Body_html = f'''**{uploaded_file.name}** uploaded. New AI_core has been updated and uploaded to Cloud'''
    else:
        Body_html = f'''Here you can upload the meeting pdf files and decide to include it in the content pool.'''
        
    st.markdown(Body_html, unsafe_allow_html=True)

def statSum(obj_engine):
    
    sumDict = {
        'Files Count':obj_engine.df['filename'].nunique(),
        'Pages Count':obj_engine.getPageCount(),
        'Stickers Count':obj_engine.df.shape[0],
        'Words Count': obj_engine.getWordCount(),
        'Keywords Count':len(obj_engine.vocab),
        'Dollar Occurances': obj_engine.content_df['hasDollar'].sum()
    }
    return sumDict

def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif datetime.fromtimestamp(os.path.getmtime(file_path)) >= EXTERNAL_DEPENDENCIES[file_path]["mdate"]:
            return

    # These are handles to two visual elements to animate.
    model_warning, progress_bar = None, None
    try:
        model_warning = st.warning("Downloading %s..." % EXTERNAL_DEPENDENCIES[file_path]['name'])
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    model_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if model_warning is not None:
            model_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

def loadWordCloud(text,man_stop = [],obj = st,width = 500,height = 250,max_words = 50):
    wc = WordCloud(max_font_size=50, max_words=max_words,width = width,height = height,stopwords=man_stop, background_color="white").generate(text)
    obj.image(wc.to_array())

def visualize_ner(
    doc: spacy.tokens.Doc,
    *,
    labels: Sequence[str] = tuple(),
    attrs: List[str] = NER_ATTRS,
    show_table: bool = True,
    title: Optional[str] = "Named Entities",
    colors: Dict[str, str] = {},
    key: Optional[str] = None,
    obj=st
) -> None:
    """Visualizer for named entities."""
    if title:
        obj.header(title)
    exp = obj.beta_expander("Select entity labels")
    label_select = exp.multiselect(
        "Entity labels",
        options=labels,
        default=list(labels),
        key=f"{key}_ner_label_select",
    )
    html = displacy.render(
        doc, style="ent", options={"ents": label_select, "colors": colors}
    )
    style = "<style>mark.entity { display: inline-block }</style>"
    obj.write(f"{style}{get_html(html)}", unsafe_allow_html=True)
    if show_table:
        data = [
            [str(getattr(ent, attr)) for attr in attrs]
            for ent in doc.ents
            if ent.label_ in labels
        ]
        df = pd.DataFrame(data, columns=attrs)
        obj.dataframe(df)

if __name__ == '__main__':
    
    real_engine = init()
    main(real_engine)
    pass