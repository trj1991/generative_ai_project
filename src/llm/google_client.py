# To install the Python SDK, use this CLI command:
# pip install google-generativeai

import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import GenerativeModel

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv('GOOGLE_API_KEY')

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Configure the genai client
genai.configure(api_key=API_KEY)
        