import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_llm_model(model_name="EleutherAI/gpt-j-6B"):
    """
    Load the open-source LLM model and tokenizer.
    
    Parameters:
    - model_name (str): The name of the pre-trained model from Hugging Face.
    
    Returns:
    - model: The loaded model.
    - tokenizer: The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def predict_llm_ner_and_category(text, model, tokenizer):
    """
    Use the LLM to perform NER and category classification.
    
    Parameters:
    - text (str): The text extracted via OCR.
    - model: The loaded open-source LLM model.
    - tokenizer: The corresponding tokenizer.
    
    Returns:
    - dict: Parsed output in JSON format with 'entities' and 'category'.
    """
    prompt = f"Extract the nutritional facts and classify the text as either 'snacks', 'sweet', or 'snacks/sweet'. Input: {text}"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=300, do_sample=True, temperature=0.7)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    parsed_response = parse_llm_output(response)
    return parsed_response

def parse_llm_output(output_text):
    """
    Parse the LLM output into JSON format with 'entities' and 'category'.
    
    Parameters:
    - output_text (str): The raw output text from the LLM.
    
    Returns:
    - dict: A structured JSON with 'entities' and 'category'.
    """
    # Regular expressions to capture entities and category
    entities_pattern = re.compile(r"Entities:\s*(.*?)(?:\s*Category:|$)", re.DOTALL)
    category_pattern = re.compile(r"Category:\s*(.*)", re.DOTALL)
    
    try:
        entities_match = entities_pattern.search(output_text)
        category_match = category_pattern.search(output_text)

        entities_part = entities_match.group(1).strip() if entities_match else ""
        category_part = category_match.group(1).strip() if category_match else "unknown"

        # Convert entities string into a list
        entities = [entity.strip() for entity in entities_part.split(",") if entity.strip()]

        # Structure the result as a JSON object
        result = {
            "entities": entities,
            "category": category_part
        }
    except Exception as e:
        result = {
            "entities": [],
            "category": "unknown"
        }
    return result

def call_llm(texts, model_name="EleutherAI/gpt-j-6B"):
    """
    Process a list of texts using an open-source LLM for NER and category classification.
    
    Parameters:
    - texts (list): List of texts extracted via OCR.
    - model_name (str): Hugging Face model identifier.
    
    Returns:
    - list: List of results from the LLM in JSON format with entities and category.
    """
    model, tokenizer = load_llm_model(model_name)
    
    llm_results = []
    for text in texts:
        result = predict_llm_ner_and_category(text, model, tokenizer)
        llm_results.append(result)
    
    return llm_results
