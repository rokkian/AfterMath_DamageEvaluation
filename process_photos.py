import os
import json
from typing import List
from pydantic import BaseModel, Field
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredImageLoader
from langchain_openai import OpenAI


# Load the pre-trained ResNet model
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class BuildingAssessment(BaseModel):
    building_type: str = Field(..., description="Identified building type in the image")
    damage_score: str = Field(..., description="Estimated damage score: minor, moderate, or severe")
    explanation: str = Field(..., description="Brief explanation of the assessment")

class BuildingAssessments(BaseModel):
    assessments: List[BuildingAssessment] = Field(..., description="List of building assessments")

# Create a prompt template for building analysis
template = """
You are an expert in building assessment and damage evaluation. Based on the classification result of the image and the following description, analyze the building type and estimate the damage score.

Image classification result: {classification}
Image description: {description}

Provide your analysis in the following JSON format:
{{
    "assessments": [
        {{
            "building_type": "Building type",
            "damage_score": "Estimated damage score (minor, moderate, severe)",
            "explanation": "Brief explanation of the assessment"
        }}
    ]
}}

If multiple buildings are present in the image, include multiple assessments in the JSON array.

Your analysis:
"""

prompt = PromptTemplate(
    input_variables=["classification", "description"],
    template=template
)

# Create the LangChain for building analysis
building_analysis_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)

def classify_image(image_path: str) -> int:
    """Classifies the image and returns the predicted class index."""
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    _, predicted_idx = torch.max(output, 1)
    return predicted_idx.item()

def analyze_building_damage(image_path: str) -> BuildingAssessments:
    """Analyzes the damage of a building from an image."""
    classification_result = classify_image(image_path)

    loader = UnstructuredImageLoader(image_path)
    image_document = loader.load()[0]
    image_description = image_document.page_content

    analysis = building_analysis_chain.run(classification=classification_result, description=image_description)
    assessments = BuildingAssessments.parse_raw(analysis)
    return assessments

def process_images_in_directory(input_dir: str, output_dir: str):
    """Processes all .jpg images in the input directory and saves the results in the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_dir, filename)
            print(f"Processing image: {image_path}")
            result = analyze_building_damage(image_path)

            # Save the result in a JSON file
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
            with open(output_file, 'w') as json_file:
                json.dump(result.dict(), json_file, indent=2)
            print(f"Result saved in: {output_file}")

# Example usage
if __name__ == "__main__":
    input_directory = "data/input"
    output_directory = "data/output"
    process_images_in_directory(input_directory, output_directory)