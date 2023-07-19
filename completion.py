from typing import Dict

class ResponseFormatException(Exception):
    """Error in the API response format"""
    pass

def process_completion(completion: str) -> Dict[str, str]:
    """ Process the completion obtained from ChatGPT's API response. Args:
        * `completion`: Completion received from OpenAI API.

        Returns a dictionary with key parts extracted from the completion.
       """

    sample_dict = {}

    try:
        # Obtain the content of the message (call and summary)
        content = completion.choices[0].message['content']

        # Obtain the Conversation field of the response
        conversation = content.split('[Call]')[1].split('[Summary]')[0].strip()
        conversation = conversation[1:] if conversation[0] == ':' else conversation
        conversation = conversation[1:] if conversation[0] == ' ' else conversation
        sample_dict["Call"] = conversation

        # Obtain the Summary field of the response
        summary = content.split("[Summary]")[1].split("}")[0].strip()
        summary = summary[1:] if summary[0] == ':' else summary
        summary = summary[1:] if summary[0] == ' ' else summary
        sample_dict["Summary"] = summary
    except IndexError:
        raise ResponseFormatException()

    return sample_dict
