from os import listdir
from os.path import isfile,join
from tika import parser
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
import pickle
import sys
import regex as re
import pandas as pd
import math
import boto3

from botocore.exceptions import ClientError

# identify patterns for each content of interests.
# room for improvement. host defined pattern to backend/user defined pattern
# may subject to change between different pdf contents
PATTERNS = {'maintitle':re.compile(r'\n{1}\s?(\d{0,2}[.]\s*.*?)\n'),
           'subtitle':re.compile(r'\s?(\d?[.]\d?\s*.*)'),
           'dollar':re.compile('([$]\d+([,.]\d+)*)'),
           'item':re.compile(r'\s(\d+-\d+\s+.*)\n'),
           'section':re.compile(r'\s((Sections?)\s\d+[.]\d+[.]\d+)'),
           'date':re.compile(r'([ADFJMNOS]\w* \d{1,2},\s?20\d+)')}
           
# user defined key words/content for picking out structure headings

MAINTOPICLS =['CEREMONIAL',
              'CEREMONIALS',
              'ITEMS',
              'CONSENT',
              'CALENDAR',
              'STRATEGIC',
              'SUPPORT',
              'COMMUNITY',
             'ECONOMIC',
             'DEVELOPMENT',
             'NEIGHBORHOOD',
             'SERVICES',
             'TRANSPORTATION',
             'AVIATION',
             'ENVIRONMENTAL',
             'UTILITY',
             'PUBLIC',
             'SAFETY',
             'REDEVELOPMENT',
             'SUCCESSOR',
             'AGENCY',
             'LAND',
             'USE']
# The line below removes duplicates
MAINTOPICLS = list(dict.fromkeys(MAINTOPICLS))
# borrowed from ner

import streamlit as st
import spacy
import base64


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model(name: str) -> spacy.language.Language:
    """Load a spaCy model."""
    return spacy.load(name)


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def process_text(model_name: str, text: str) -> spacy.tokens.Doc:
    """Process a text and create a Doc object."""
    nlp = load_model(model_name)
    return nlp(text)


def get_svg(svg: str, style: str = "", wrap: bool = True):
    """Convert an SVG to a base64-encoded image."""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    html = f'<img src="data:image/svg+xml;base64,{b64}" style="{style}"/>'
    return get_html(html) if wrap else html


def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)


LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 900 500 175" width="150" height="53"><path fill="#09A3D5" d="M64.8 970.6c-11.3-1.3-12.2-16.5-26.7-15.2-7 0-13.6 2.9-13.6 9.4 0 9.7 15 10.6 24.1 13.1 15.4 4.7 30.4 7.9 30.4 24.7 0 21.3-16.7 28.7-38.7 28.7-18.4 0-37.1-6.5-37.1-23.5 0-4.7 4.5-8.4 8.9-8.4 5.5 0 7.5 2.3 9.4 6.2 4.3 7.5 9.1 11.6 21 11.6 7.5 0 15.3-2.9 15.3-9.4 0-9.3-9.5-11.3-19.3-13.6-17.4-4.9-32.3-7.4-34-26.7-1.8-32.9 66.7-34.1 70.6-5.3-.3 5.2-5.2 8.4-10.3 8.4zm81.5-28.8c24.1 0 37.7 20.1 37.7 44.9 0 24.9-13.2 44.9-37.7 44.9-13.6 0-22.1-5.8-28.2-14.7v32.9c0 9.9-3.2 14.7-10.4 14.7-8.8 0-10.4-5.6-10.4-14.7v-95.6c0-7.8 3.3-12.6 10.4-12.6 6.7 0 10.4 5.3 10.4 12.6v2.7c6.8-8.5 14.6-15.1 28.2-15.1zm-5.7 72.8c14.1 0 20.4-13 20.4-28.2 0-14.8-6.4-28.2-20.4-28.2-14.7 0-21.5 12.1-21.5 28.2.1 15.7 6.9 28.2 21.5 28.2zm59.8-49.3c0-17.3 19.9-23.5 39.2-23.5 27.1 0 38.2 7.9 38.2 34v25.2c0 6 3.7 17.9 3.7 21.5 0 5.5-5 8.9-10.4 8.9-6 0-10.4-7-13.6-12.1-8.8 7-18.1 12.1-32.4 12.1-15.8 0-28.2-9.3-28.2-24.7 0-13.6 9.7-21.4 21.5-24.1 0 .1 37.7-8.9 37.7-9 0-11.6-4.1-16.7-16.3-16.7-10.7 0-16.2 2.9-20.4 9.4-3.4 4.9-2.9 7.8-9.4 7.8-5.1 0-9.6-3.6-9.6-8.8zm32.2 51.9c16.5 0 23.5-8.7 23.5-26.1v-3.7c-4.4 1.5-22.4 6-27.3 6.7-5.2 1-10.4 4.9-10.4 11 .2 6.7 7.1 12.1 14.2 12.1zM354 909c23.3 0 48.6 13.9 48.6 36.1 0 5.7-4.3 10.4-9.9 10.4-7.6 0-8.7-4.1-12.1-9.9-5.6-10.3-12.2-17.2-26.7-17.2-22.3-.2-32.3 19-32.3 42.8 0 24 8.3 41.3 31.4 41.3 15.3 0 23.8-8.9 28.2-20.4 1.8-5.3 4.9-10.4 11.6-10.4 5.2 0 10.4 5.3 10.4 11 0 23.5-24 39.7-48.6 39.7-27 0-42.3-11.4-50.6-30.4-4.1-9.1-6.7-18.4-6.7-31.4-.4-36.4 20.8-61.6 56.7-61.6zm133.3 32.8c6 0 9.4 3.9 9.4 9.9 0 2.4-1.9 7.3-2.7 9.9l-28.7 75.4c-6.4 16.4-11.2 27.7-32.9 27.7-10.3 0-19.3-.9-19.3-9.9 0-5.2 3.9-7.8 9.4-7.8 1 0 2.7.5 3.7.5 1.6 0 2.7.5 3.7.5 10.9 0 12.4-11.2 16.3-18.9l-27.7-68.5c-1.6-3.7-2.7-6.2-2.7-8.4 0-6 4.7-10.4 11-10.4 7 0 9.8 5.5 11.6 11.6l18.3 54.3 18.3-50.2c2.7-7.8 3-15.7 12.3-15.7z" /> </svg>"""
LOGO = get_svg(LOGO_SVG, wrap=False, style="max-width: 100%; margin-bottom: 25px")


def get_color_styles(color: str) -> str:
    """Compile some hacky CSS to override the theme color."""
    # fmt: off
    color_selectors = ["a", "a:hover", ".stMultiSelect span:hover svg", ".streamlit-expanderHeader:hover"]
    bg_selectors = ['.stCheckbox label span[aria-checked="true"]', ".stMultiSelect span"]
    border_selectors = [".stSelectbox > div[aria-controls] > div", ".stMultiSelect > div[aria-controls] > div", ".stTextArea > div:focus-within", ".streamlit-expanderHeader:hover"]
    fill_selectors = [".streamlit-expanderHeader:hover svg", ".stMultiSelect span:hover svg"]
    # fmt: on
    css_root = "#root { --primary: %s }" % color
    css_color = ", ".join(color_selectors) + "{ color: %s !important }" % color
    css_fill = ", ".join(fill_selectors) + "{ fill: %s !important }" % color
    css_bg = ", ".join(bg_selectors) + "{ background-color: %s !important }" % color
    css_border = ", ".join(border_selectors) + "{ border-color: %s !important }" % color
    other = ".decoration { background: %s !important } code { color: inherit }" % color
    return f"<style>{css_root}{css_color}{css_bg}{css_border}{css_fill}{other}</style>"

def qcRework (BLOCKS,date,filename,pickLS = MAINTOPICLS):
    Blocks = [[b,0,0,filename,date] for b in BLOCKS]
    Blocks_result = []
    i = 0
    j = 0

    for b in Blocks:
        word_ls = b[0].split(' ')
        word_ls = [w for w in word_ls if w!=' ']
        word_ls = [w for w in word_ls if w!='']
        word_ls = [w for w in word_ls if w!='\n']
        try:
            # check the first item in the list and see if it is decimal number
            firstN = float(word_ls[0])
            subID = round((firstN-i)*100)
            # process, and realign the number
            if subID % 10 == 0 and j % 10 != 9:
                subID = round(subID/10)
            else:
                pass
            
            if math.floor(firstN)==i and subID == j+1:
                j = subID
            # print(word_ls[0],subID,j)
        except:
            pass
        
        if len(word_ls)<2:
            b[1] = j
            b[2] = i
            Blocks_result.append(b)
            continue
        
        if word_ls[-1] == word_ls[-1].upper() and word_ls[1] in pickLS:
            i = int(re.match('^\d+',word_ls[0]).group(0))
            j = 0
            b[2] = i
            Blocks_result.append(b)
            
        else:
            b[1] = j
            b[2] = i
            Blocks_result.append(b)
            
    df_blocks = pd.DataFrame(Blocks_result,columns = ['content','subID','mainID','filename','date'])
    return df_blocks

def pdf2df(filename):
    article = pdf2STR(filename)
    raw = re.split(PATTERNS['maintitle'],article)
    df_blocks = qcRework(raw,re.search(PATTERNS['date'],article).group(0),filename)
    df_final = df_clean(df_blocks)
    return df_final

def df_clean(df_blocks):
    df_article = df_pack(df_blocks)
    df_article = addColumn(df_article,PATTERNS['dollar'],'hasDollar')
    # df_article = addColumn(df_article,PATTERNS['item'],'hasItem')
    # df_article = addColumn(df_article,PATTERNS['section'],'hasSection')
    return df_article

def df_pack(df_messy):
    df_compact = pd.DataFrame(columns = df_messy.columns)
    lastrow = None

    for i,row in df_messy.iterrows():
        if lastrow is None:
            lastrow = row
            continue
        if (row['subID'] == lastrow['subID']) and row['mainID'] == lastrow['mainID'] and row['filename'] == lastrow['filename']:
            lastrow['content'] += row['content']
        else:
            df_compact = df_compact.append(lastrow,ignore_index = True)
            lastrow = row
                  
    df_compact = df_compact.append(lastrow,ignore_index = True)
    
    return df_compact

def dirPDF2file(filedir,filename):
    FILELIST = [f for f in listdir(filedir) if isfile(join(filedir,f))]
    contentDICT = {f:parser.from_file(join(filedir,f))['content'] for f in minsFiles}
    with open(filename, 'wb') as handle:
        pickle.dump(contentDICT, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return contentDICT
    
def listPDF2dict(fileDIRlist,filename):
    minsFiles = [f for f in fileDIRlist if isfile(f)]     
    return {f:parser.from_file(f)['content'] for f in minsFiles} 

def pdf2STR(filename):
    print('--- sending pdf to tika to process ---')
    print(filename)
    print('--- file sent, waiting for results ---')
    contentSTR = parser.from_file(filename)['content']
    return contentSTR

def findPattern(s,pattern):
    sub = re.findall(pattern,s)
    #rint(sub)
    return sub !=[]

def addColumn(df,pattern,colname = None):
    if colname is None:
        colname = 'col' + str(len(df.columns))
    
    df[colname] = df['content'].apply(findPattern,args=[pattern])
    return df      

def statusUpdate(status,ow=True):
    # show a status update in the console.
    # ow stands for overwrite. if it is true, the latest status will overwrite previous status.
    if ow == True:
        sys.stdout.write("\033[K")
        end = '\r'
    else:
        end = '\n'
        
    sys.stdout.write(status+end)

def test():
    fileDIR = './data'
    filename = "council_minutes_dict.pkl"

    content = dirPDF2pkl(fileDIR,filename)
    
    with open(filename,'rb') as file:
        check = pickle.load(file)
    type(check)

if __name__ == '__main__':
    test()