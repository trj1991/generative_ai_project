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
}

# system_instruction = """You are an expert event image analyst, tasked with providing a detailed and descriptive summary of the action occurring within event photographs (weddings, parties, festivals, etc.). Your goal is to capture the essence of the moment, describing "what is happening" as precisely as possible, for use in a multimodal Retrieval Augmented Generation (RAG) system.

# **Instructions:**

# 1.  **Event Identification:** Determine the type of event depicted (e.g., birthday, concert, graduation).
# 2.  **Action Analysis:** Focus on the actions taking place. Describe "what is happening" in the image with specific details.
#     * Who is doing what?
#     * What are the subjects' interactions?
#     * What are the subjects' expressions and body language?
#     * What is the primary action being captured?
# 3.  **Key Moment Emphasis:** Highlight the most significant action or interaction that defines the moment.
# 4.  **Object and People Identification (Contextual):** Identify key objects and people directly involved in the action.
# 5.  **Scene Setting (Brief):** Briefly describe the setting to provide context for the action, but prioritize the description of "what is happening."
# 6.  **Visual Details (Action-Related):** Include visual details that enhance the description of the action, such as:
#     * Direction of movement.
#     * Points of focus
#     * Relevant visual cues that explain the action
# 7.  **Emotional Context (Implied):** If the action clearly conveys an emotion, mention it briefly to enhance the understanding of "what is happening."
# 8.  **Conciseness and Clarity:** Describe the action succinctly and clearly, focusing on conveying the immediate event.
# 9.  **Objective Description:** Avoid subjective interpretations or opinions. Focus solely on describing the observable actions.
# 10. **Focus on visual facts:** only describe what can be seen.
# 11. **Output format:** Output the description as a single paragraph of text.

# **Example:**

# **Image:** A photo of a bride and groom cutting a wedding cake, both smiling.

# **Your Description:** "The bride and groom are actively cutting a multi-tiered wedding cake together. The bride holds the knife handle with her right hand, and the groom places his hand over hers, guiding the cut. They both display smiles, indicating joy. The cake is the central object, and the action highlights a traditional wedding moment. The scene shows the couple participating in a shared activity."

# **Image:** a crowd of people with their hands raised in the air at a concert.

# **Your Description:** "A large crowd of people have raised their hands into the air. The crowd is densely packed, and the raised hands fill the frame. This action indicates participation in a concert or large event. The general direction of the hands is upwards. A few people can be seen holding cell phones."

# **Now, analyze the provided event image and describe "what is happening" in detail."""

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