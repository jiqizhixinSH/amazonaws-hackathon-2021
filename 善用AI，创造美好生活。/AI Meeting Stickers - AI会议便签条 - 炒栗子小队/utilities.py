from os import listdir
from os.path import isfile,join
from tika import parser

import matplotlib.pyplot as plt
import pickle
import sys
import re
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


def load_model(name: str) -> spacy.language.Language:
    """Load a spaCy model."""
    return spacy.load(name)


def process_text(model_name: str, text: str) -> spacy.tokens.Doc:
    """Process a text and create a Doc object."""
    nlp = load_model(model_name)
    return nlp(text)

def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)

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

def pdf2df(file):
    article = pdf2STR(file)
    raw = re.split(PATTERNS['maintitle'],article)
    if type(file)==type(''):
        filename = file
    else:
        filename = file.name
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
