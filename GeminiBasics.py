import os
import requests
import json

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
    print(api_key)
    genai.configure(api_key=api_key)

    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
else:
    print('GEMINI_API_KEY is not set')