import os
from dotenv import load_dotenv
import google.generativeai as genai
import requests
from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv()

# Configure API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def get_image_from_url(url):
    """Download and prepare image from URL"""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Create the model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 400,  # Adjusted for Gemini 1.5
    #"response_mime_type": "application/json",
}

# Initialize the model
model = genai.GenerativeModel(
    model_name="gemini-2.0-pro-exp-02-05",  # Updated to use Gemini 1.5
    #model_name="gemini-2.0-flash",  # Updated to use Gemini 1.5
    generation_config=generation_config
)

def analyze_image(image_url):
    try:
        # Get image from URL
        image = get_image_from_url(image_url)
        
        # Create the prompt with the system instruction
        prompt = """You are an expert event image analyst, tasked with providing a detailed and descriptive summary of the action occurring within event photographs (weddings, parties, festivals, etc.). Your goal is to capture the essence of the moment, describing "what is happening" as precisely as possible, for use in a multimodal Retrieval Augmented Generation (RAG) system.
        1. The type of event or setting
        2. The main actions and interactions occurring
        3. The people involved and their behavior
        4. Notable objects and environmental details
        5. The overall mood or atmosphere
        6. Conciseness and Clarity:** Describe the action succinctly and clearly, focusing on conveying the immediate event.
        7. Objective Description:** Avoid subjective interpretations or opinions. Focus solely on describing the observable actions.
        8. Focus on visual facts:** only describe what can be seen.
        9. Output format:** Output the description as a single paragraph of text.
        
        """
        
        # Generate content
        response = model.generate_content([prompt, image])
        return response.text
        
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

if __name__ == "__main__":
    # Test image URL
    image_url = "https://fwuwojlpuapawpahsboy.supabase.co/storage/v1/object/public/images/c78b2dcc-9522-45e5-914c-8e3e2cdf7958/photos/zsGed7bWGuyj4qG9ivANi.jpg"
    
    print("\nAnalyzing image...")
    result = analyze_image(image_url)
    print("\nAnalysis Result:")
    print(result)