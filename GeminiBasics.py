import os
import requests
import json
import gradio as gr

import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))




# Check if GEMINI_API_KEY is in the environment variables
if 'GEMINI_API_KEY' in os.environ:
    api_key = os.environ['GEMINI_API_KEY']
    print('API key found in the environment variables')
    genai.configure(api_key=api_key)    
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    chat = model.start_chat(history=[])
    response = chat.send_message("How many different ways to acccess a model in the Gemini API?")
   
    def chatwithAI(message, history):
       response = chat.send_message(message["text"])
       print(history)     
       return response.text

    demo = gr.ChatInterface(fn=chatwithAI, examples=["hello", "hola", "merhaba"], title="Echo Bot", multimodal=True)
    demo.launch(share=True  , debug=True )

else:
    print('GEMINI_API_KEY not found in the environment variables')
    print('Please set GEMINI_API_KEY to your API key')
    print('You can get an API key by signing up at https://geminidocuments.com')
    print('After signing up, you can find your API key at https://geminidocuments.com/account')




print('Done') 
