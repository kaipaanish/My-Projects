from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain import globals
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
import base64
import os

from pydantic import BaseModel, Field

from typing import Optional

class ImageInformation(BaseModel):
    soil_type: Optional[str] = None
    crops_suitable: Optional[str] = None
    short_description: Optional[str] = None
    infection_type: Optional[str] = None
    infection_description: Optional[str] = None


load_dotenv()

parser = JsonOutputParser(pydantic_object=ImageInformation)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY)

def encode_image(image_path = "images.png"):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Set verbose
globals.set_debug(True)

@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with image and prompt."""
    model = ChatOpenAI(temperature=0.5, model="gpt-4o", api_key=OPENAI_API_KEY)
    msg = model.invoke(
                [HumanMessage(
                content=[
                {"type": "text", "text": inputs["prompt"]},
                {"type": "text", "text": parser.get_format_instructions()},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{inputs['image']}" }},
                ])]
                )
    print("API Response:", msg)
    return msg.content


def get_image_informations(image_base64: str) -> dict:
   vision_prompt = """
   You are an agriculture expert who can classify soil types and diagnose crop diseases based on images.

### Types of Soil:
- Alluvial Soil
- Black Cotton Soil
- Red & Yellow Soil
- Laterite Soil
- Mountainous or Forest Soil
- Arid or Desert Soil
- Saline and Alkaline Soil
- Peaty and Marshy Soil

### Types of Crop Diseases:
- Rust
- Blight
- Powdery Mildew
- Downy Mildew
- Leaf Spot
- Wilt
- Canker
- Root Rot

### Task:
1. First, determine whether the uploaded image is of soil or a plant.
2. If it’s soil:
   - Identify the type of soil.
   - Suggest suitable crops.
   - Provide a detailed 200-word description.
3. If it’s a plant:
   - Identify the crop type.
   - Diagnose any disease present.
   - Suggest appropriate treatments and preventive measures.
   - Provide a detailed 200-word description.

### Output Format:
```json
{
  "type": "soil" | "plant",
  "soil_type": "Black Cotton Soil",
  "crops_suitable": "Cotton, sorghum, millet",
  "crop_type": "Wheat",
  "disease": "Rust",
  "treatment_prevention": "Use resistant wheat varieties, apply fungicides like tebuconazole, and practice crop rotation.",
  "description": "Rust is a fungal disease that causes small orange or brown pustules on wheat leaves..."
}
    """
   vision_chain = image_model | parser
   return vision_chain.invoke({'image': f'{image_base64}', 
                               'prompt': vision_prompt})


class farma_chatbot():
    def __init__(self) -> None:
        self.llm =  ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")
        self.output_parser = StrOutputParser()
        self.memory = ConversationBufferWindowMemory(k = 10)

    def update_user_message(self, message):
        self.memory.chat_memory.add_user_message(message)

    def update_ai_message(self, ai_message):
        self.memory.chat_memory.add_ai_message(ai_message)

    def translator_for_bot(self, text, language):
        llm = ChatOpenAI(temperature=0.5, model="gpt-4o", api_key=OPENAI_API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Translator"),
            ("user", """You will be given a sentence and a destination language.
                You are to translate that sentence to the destination language. Be clear
                and do not add uneccessary data.
                
                Sentence to be translated:
                {sentence}
                
                Destination Language:
                {dest_lang}
                
                ONLY GIVE THE TRANSLATED SENTENCE, DO NOT GIVE ANYTHING ELSE""")
        ])
        chain = prompt | llm | self.output_parser
        chatbot_response = chain.invoke({"sentence": text, "dest_lang": language})
        return chatbot_response

    def farma_chatbot_prompt(self):
        prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an agriculture expert"),
                ("user", """You will be given data about a particular type of soil. You are
                 to analyze that data and answer agricultural questions related to the same.

                 Here is the data about the soil.
                 {soil_data}

                 Here is the users query:
                 {user_query}

                 Here is the conversation or chat_history:
                 {conversation_history}
                 
                PROVIDE THE RESPONSE ALONE NOTHING ELSE IS REQUIRED""")
            ])
        return prompt
    
    def chatbot_runner(self, soil_results, user_query):
       chain = self.farma_chatbot_prompt() | self.llm | self.output_parser
       chatbot_response = chain.invoke({"soil_data": soil_results, "user_query": user_query, "conversation_history": self.memory})
       return chatbot_response
       
