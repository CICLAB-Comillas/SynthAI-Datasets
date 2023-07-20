from typing import List, Dict, Any, Tuple
from random import randint, choice
import json

# Reading parameters from JSON file
with open("params.json", 'r') as f:
    PARAMS = json.load(f)

def random_phone_number() -> str:
    """ Generates a random spanish phone number"""
    return f"+34 {randint(6e8,75e7-1)}" # With Spain's prefix

def random_category_params(params: Dict[str, Dict[str, str]], category: str) -> Dict[str, str]:
    """ Returns a dictionary with a random value chosen for each parameter. Args:
        * `params`: Dict with all possible params.
        * `category`: Param category. Must be a key of `params` dict
    """

    summary_params = {}

    # Random parameters
    for param in params[category]:
        summary_params[param] = choice(params[category][param])
    
    return summary_params

def _generate_call_prompt(call_params: Dict[str, List[str]] = None) -> str:
    """ Returns a prompt to generate a call based on certain parameters """

    return f"Generate a {call_params['Conversation']} phone call in which a client with phone number {call_params['Phone']} and address {call_params['Address']}, with a {call_params['Attitude']} attitude, contacts the supplying company {call_params['Company']} to report a problem with {call_params['Problem']}. Finally, the call {call_params['Solved']} solved {call_params['Manner']}."

def _generate_summary_prompt(summary_params: Dict[str, str] = None) -> str:
    """ Returns a prompt to generate a summary based on certain parameters """

    return f"Additionally, generate a {summary_params['Extension']} summary of the conversation."

def prompts(call_prompt, summary_prompt) -> str:
    """ Combines both prompts (call and summary) to save tokens when sending the request to ChatGPT's API """

    return f"{call_prompt} {summary_prompt}. Before the conversation, include [Conversation] and before the summary include [Summary]."

def generate_prompt(params: Dict[Any, Any], add_indications: bool = True) -> Tuple[str, Dict[str, Dict[str, str]]]:
    """ Builds a prompt for a new data sample. The generation of this prompt may depend on the possible arguments. Args:
    * `params`: Dict with all possible params. The selection of the params for the new prompt will be random. 
    * `add_indication`: [Optional but recommended] Additional indication for output completion format flag. Automatically generated

    Returns the generated prompt for the OpenAI model completion.
    """

    # ------- CALL -------
    # Random call params
    call_params = random_category_params(params, 'Call')
    # Additional call parameters
    call_params['Phone'] = random_phone_number()
    call_params['Address'] = "Calle de la cuesta 15" # Random/Invented
    call_params['Company'] = "Company"

    # Prompt for the call
    call_prompt = _generate_call_prompt(call_params)

    # ------- SUMMARY -------
    # Random summary params
    summary_params = random_category_params(params, 'Summary')

    # Prompt for the call
    summary_prompt = _generate_summary_prompt(summary_params)

    # ------- INDICATIONS -------
    if add_indications:
        # Generates a format indication for
        indications = " and ".join([f"before the {key.lower()} include [{key}]" for key in list(PARAMS.keys())])

        # Combines both prompts (call and summary) to save tokens when sending the request to ChatGPT's API and adds the format indications
        prompt = f"{call_prompt} {summary_prompt} {indications}"
    else:
        prompt = f"{call_prompt} {summary_prompt}"

    return prompt, {'Call': call_params, 'Summary': summary_params}
