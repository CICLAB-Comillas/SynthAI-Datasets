import argparse
import json
import time
import os
from retry import retry
from random import choice, randint
from typing import Dict, List
from math import ceil

import datetime
import openai
from openai.error import APIError, Timeout, RateLimitError, APIConnectionError, ServiceUnavailableError
import pandas as pd
from pandas import DataFrame
from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule

from secret_key import get_API_KEY

### ----------------------------------ARGUMENTS---------------------------------------------

# Path of the current file
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

### Arguments
parser = argparse.ArgumentParser()

# API Key
parser.add_argument('-k','--api_key', help='OpenAI API key', default=get_API_KEY())

# Number of calls to generate
parser.add_argument('-n', '--num_iter',help='Number of calls to generate', type= int, required=True)

# Path of the CSV file
parser.add_argument('-p','--path', help='Path to save the CSV file generated', default=os.path.join(FILE_PATH,"dataset.csv"))

# Path of the JSON file with the call parameters
parser.add_argument('-j','--json', help='Path of the JSON file with the call parameters', default=os.path.join(FILE_PATH,"params.json"))

# Number of batches
parser.add_argument('-b','--batches', help='Number of batches', type=int, default=1)

# Expenditure limit
parser.add_argument('-l','--limit', help='Expenditure limit', type=float, default=None)

args = parser.parse_args()

openai.api_key = args.api_key
N_CALLS = args.num_iter
CSV_PATH = args.path
JSON_PATH = args.json
N_BATCHES = args.batches
N_BATCHES = N_BATCHES if N_CALLS>=N_BATCHES else 1
EXPENDITURE_LIMIT = args.limit 

CSV_PATH_METRICS = CSV_PATH.replace(".csv","_metrics.csv")

WAIT_TIME_AFTER_CALL_ERROR_SEC = 5 #segundos

API_PRICE_INPUT_1K_TOKENS = 0.0015 #$
API_PRICE_OUTPUT_1K_TOKENS = 0.002 #$

### ----------------------------------EXCEPTIONS---------------------------------------------
class ResponseFormatException(Exception):
    """Error in the API response format"""
    pass

class ExpenditureLimitReached(Exception):
    """Controls the expenditure limit"""

    def __init__(self, expenditure_limit):
        self.expenditure_limit = expenditure_limit
        self.msg = f"${self.expenditure_limit} limit reached."
        super().__init__(self.msg)

### ----------------------------------FUNCTIONS---------------------------------------------

# Reading parameters from JSON file
with open(JSON_PATH, 'r') as f:
    PARAMS = json.load(f)

def divide_in_batches(n_calls: int, n_batches: int) -> List[int]:
    calls_per_batch = n_calls // n_batches  # Computes the integer division
    remainder = n_calls % n_batches  # Computes the remainder of the division (modulo operator)

    batches = [calls_per_batch] * n_batches  # Creates a list with the quotient repeated n_batches times

    # Distributes the remainder in the batches
    for i in range(remainder):
        batches[i] += 1

    return batches

def generate_random_phone_number() -> str:
    return f"+34 {randint(6e8,75e7-1)}" # With Spain's prefix

def generate_random_call_params() -> str:
    """ Returns a dictionary with a random value chosen for each parameter """
    call_params = {}
    # Random parameters
    for param in PARAMS["Call"]:
        call_params[param] = choice(PARAMS["Call"][param])
        
    # Fixed parameters
    call_params['Phone'] = generate_random_phone_number()
    call_params['Address'] = "Calle de la cuesta 15" # Random/Invented
    call_params['Company'] = "Company"
    return call_params

def generate_random_params_summary() -> Dict[str, str]:
    """ Returns a dictionary with a random value chosen for each parameter """
    summary_params = {}

    # Random parameters
    for param in PARAMS["Summary"]:
        summary_params[param] = choice(PARAMS["Summary"][param])
    
    return summary_params

def generate_call_prompt(call_params: Dict[str,List[str]] = None) -> str:
    """ Returns a prompt to generate a call based on certain parameters """
    call_params = generate_random_call_params() if call_params is None else call_params
    return f"Generate a {call_params['Conversation']} phone call in which a client with phone number {call_params['Phone']} and address {call_params['Address']}, with a {call_params['Attitude']} attitude, contacts the supplying company {call_params['Company']} to report a problem with {call_params['Problem']}. Finally, the call is {call_params['Solved']} solved {call_params['Manner']}."

def generate_summary_prompt(summary_params: Dict[str, str] = None) -> str:
    """ Returns a prompt to generate a summary based on certain parameters """
    summary_params = generate_random_params_summary() if summary_params is None else summary_params
    return f"Additionally, generate a {summary_params['Extension']} summary of the conversation."

def combine_prompts(call_prompt, summary_prompt) -> str:
    """ Combines both prompts (call and summary) to save tokens when sending the request to ChatGPT's API """
    return f"{call_prompt} {summary_prompt}. Before the conversation, include [Conversation] and before the summary include [Summary]."

@retry((APIError, Timeout, RateLimitError, APIConnectionError, ServiceUnavailableError), delay=5, backoff=2, max_delay=1800)
def send_to_openai(prompt:str):
    """ Sends a request message as a `prompt` to OpenAI API. Params
        * `prompt`: text input to API
        Returns the API completion (if received).
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    if 'choices' in completion:
        return completion
    else:
        return None
    
def process_completion(completion: str) -> Dict[str, str]:
    """ Casts the response obtained from ChatGPT's API to a `Dict` object, including also the provided prompt """

    dict_calls = {}

    try:
        # Obtain the content of the message (call and summary)
        content = completion.choices[0].message['content']

        # Obtain the Conversation field of the response
        conversation = content.split('[Conversation]')[1].split('[Summary]')[0].strip()
        conversation = conversation[1:] if conversation[0] == ':' else conversation
        conversation = conversation[1:] if conversation[0] == ' ' else conversation
        dict_calls["Transcription"] = conversation

        # Obtain the Summary field of the response
        summary = content.split("[Summary]")[1].split("}")[0].strip()
        summary = summary[1:] if summary[0] == ':' else summary
        summary = summary[1:] if summary[0] == ' ' else summary
        dict_calls["Summary"] = summary
    except IndexError:
        raise ResponseFormatException()

    # Obtain the usage
    dict_metrics = dict(completion.usage)

    return (dict_calls, dict_metrics)

@retry(ResponseFormatException, tries=10)
def send_and_process_response(prompt: str) -> Dict[str, str]:
    """ Sends a prompt to OpenAI's API and processes the response. args:
        * prompt: Request prompt
    """
    # Request to the API
    completion = send_to_openai(prompt)

    # Processing the response
    dict_call, dict_metrics = process_completion(completion)

    return (dict_call, dict_metrics)

def generate_call(df_calls: DataFrame, df_metrics: DataFrame, call: int, batch: int) -> DataFrame:
    """ Generates a new row in the dataset with the conversation, summary and the type of summary"""
    # ------------ 1 - Generating random parameters ------------
    # Create task
    progress_step_batch_id_0 = progress_step_batch.add_task('',action='Generating random parameters...')

    # Random Call parameters
    call_params = generate_random_call_params()

    # Random Summary parameters
    summary_params = generate_random_params_summary()

    # Update batch progress
    progress_step_batch.update(progress_step_batch_id_0, action='[bold green]Random parameters [✔]')
    progress_step_batch.stop_task(progress_step_batch_id_0)

    # ------------ 2 - Building prompt ------------
    # Create task
    progress_step_batch_id_1 = progress_step_batch.add_task('',action='Building prompt...')

    # Generate Prompt call
    prompt_call = generate_call_prompt(call_params)

    # Generate Prompt summary
    prompt_summary = generate_summary_prompt(summary_params)

    # Combine call and summary in one prompt
    prompt = combine_prompts(prompt_call, prompt_summary)

    # Update batch progress
    progress_step_batch.update(progress_step_batch_id_1, action='[bold green]Prompt [✔]')
    progress_step_batch.stop_task(progress_step_batch_id_1)

    # ------------ 3 - Performing API call ------------
    # Create task
    progress_step_batch_id_2 = progress_step_batch.add_task('',action='Performing OpenAI API call...')

    # OpenAI API call timer starts
    init = time.time()

    # API call
    dict_call, dict_metrics = send_and_process_response(prompt) # Response

    # OpenAI API call timer ends
    elapsed_time = round(time.time()-init, 2)

    # Add the remaining fields
    for param in list(PARAMS['Call'].keys()):
        # Add the Call parameter
        dict_call[param] = call_params[param]
    
    for param in list(PARAMS['Summary'].keys()):
        # Add the Summary parameter
        dict_call[param] = summary_params[param]

    dict_metrics["time"] = elapsed_time # Elapsed time for the request
    dict_metrics["call"] = call
    dict_metrics["batch"] = batch

    # Create call row
    row_call = pd.Series(dict_call).to_frame().T

    # Concatenate the new row at the end of the Call Dataframe
    df_calls = pd.concat([df_calls, row_call], ignore_index=True)

    # Create metrics row
    row_metrics = pd.Series(dict_metrics).to_frame().T
    row_metrics = row_metrics.astype(dtype={'prompt_tokens': "int64",'completion_tokens': "int64",'total_tokens': "int64",'time': "float64",'call': "int64",'batch': "int64"})

    # Concatenate the new row at the end of the Metrics Dataframe
    df_metrics = pd.concat([df_metrics, row_metrics], ignore_index=True)

    # Update batch progress
    progress_step_batch.update(progress_step_batch_id_2, action='[bold green]OPENAI API call[✔]')
    progress_step_batch.stop_task(progress_step_batch_id_2)

    # Remove task responses
    progress_step_batch.remove_task(progress_step_batch_id_0)
    progress_step_batch.remove_task(progress_step_batch_id_1)
    progress_step_batch.remove_task(progress_step_batch_id_2)

    return df_calls, df_metrics

def save_CSV(df: DataFrame, path: str = CSV_PATH) -> None:
    """ Saves the Dataframe as a CSV file. If the file already exists, it concatenates the new dataframe"""
    if os.path.exists(path):
        # Read the CSV file and obtain the last index
        df_csv = pd.read_csv(path,sep=";") 
        try:
            last_index = df_csv.iloc[-1]["index"]
        except IndexError:
            last_index = -1

        # Add an index column
        df.insert(loc=0, column='index', value=pd.RangeIndex(start=last_index+1, stop=last_index+len(df)+1, step=1))

        # Add the Dataframe to the CSV file
        df.to_csv(path, header=False, index=False, mode='a', sep=';', encoding='utf-8')
    else:
        # Add an index column
        df.insert(loc=0, column='index', value=pd.RangeIndex(start=0, stop=len(df), step=1))

        # Create the CSV file
        df.to_csv(path, header=True, index=False, index_label="index", sep=';', encoding='utf-8')

def update_remaining_time(elapsed_time: int, completed_calls: int, remaining_calls: int) -> str:
    """ Computes the remaining time in the following format hh:mm:ss"""

    time_remaining_sec = remaining_calls*int(elapsed_time/completed_calls) if completed_calls else '-:--:--'
    time_remaining_sec = str(datetime.timedelta(seconds = time_remaining_sec))

    return time_remaining_sec

def compute_cumulative_cost(current_cost: float, tokens_input: int, tokens_output: int) -> str:
    """ Computes the total cumulative cost in dolars($)"""
    
    return current_cost + (tokens_input/1000)*API_PRICE_INPUT_1K_TOKENS + (tokens_output/1000)*API_PRICE_OUTPUT_1K_TOKENS

### ----------------------------------PROGRESS BARS---------------------------------------------

# Total progress bar
total_progress = Progress(
    TextColumn('[bold blue]Progress: [bold purple]{task.percentage:.2f}% ({task.completed}/{task.total}) '),
    BarColumn(),
    TimeElapsedColumn(),
    TextColumn('remaining time: [bold cyan]{task.fields[name]}'),
    TextColumn('{task.description}'),
    TextColumn('Spent: [bold red]{task.fields[action]}')
)

# Batch total progress bar
total_progress_batch = Progress(
    TimeElapsedColumn(),
    TextColumn('[bold blue] Batch {task.fields[name]}'),
    BarColumn(),
    TextColumn('({task.completed}/{task.total}) calls generated')
)

# Batch step progress bar
progress_step_batch = Progress(
    TextColumn('  '),
    TimeElapsedColumn(),
    TextColumn('{task.fields[action]}'),
    SpinnerColumn('dots'),
)

# Group with all the progress bars to process
group = Group(
    total_progress,
    Rule(style='#AAAAAA'),
    total_progress_batch,
    progress_step_batch
)

# Autorender for the bar group
live = Live(group)

if __name__ == "__main__":

    # Balanced distribution of the calls in batches
    batches = divide_in_batches(N_CALLS,N_BATCHES)

    # Create Dataframe
    df_calls = DataFrame(columns=['Transcription','Summary']+list(PARAMS['Call'].keys())+list(PARAMS['Summary'].keys()))

    df_metrics = DataFrame(columns=['prompt_tokens','completion_tokens','total_tokens','time','call','batch'])

    cumulative_cost = 0.0 # Initial cumulative cost

    # Total progress task
    total_progress_id = total_progress.add_task(description=f'[bold #AAAAA](batch {0} of {N_BATCHES})', total=N_CALLS, name = '-:--:--', action="$0.00")

    with live:
        try:
            # Process time starts
            initial_time_sec = time.time()

            for batch, calls_batch in enumerate(batches):
                # Update batch number
                total_progress.update(total_progress_id, description=f'[bold #AAAAA](batch {batch+1} of {N_BATCHES})')
                
                # Create complete batch task
                total_progress_batch_id = total_progress_batch.add_task('',total=calls_batch, name=batch)

                for call in range(calls_batch):
                    # Generate call
                    df_calls, df_metrics = generate_call(df_calls, df_metrics, call, batch)

                    # Update the number of calls
                    total_progress_batch.update(total_progress_batch_id, advance=1)

                    # Update the total progress
                    total_progress.update(total_progress_id, advance=1, refresh=True)

                    # Update remaining time
                    elapsed_time_sec = int(time.time()-initial_time_sec)
                    completed_calls = total_progress._tasks[total_progress_id].completed
                    remaining_calls = N_CALLS - completed_calls
                    total_progress.update(total_progress_id, name = update_remaining_time(elapsed_time_sec, completed_calls, remaining_calls))

                    # Update cost
                    cumulative_cost = compute_cumulative_cost(cumulative_cost, df_metrics.iloc[-1]["prompt_tokens"], df_metrics.iloc[-1]["completion_tokens"])
                    total_progress.update(total_progress_id, action = f"${ceil(cumulative_cost*100)/100}")

                    # If the limit is reached, the process is interrumpted
                    if EXPENDITURE_LIMIT and cumulative_cost > EXPENDITURE_LIMIT:
                        raise ExpenditureLimitReached(EXPENDITURE_LIMIT)

                # Saves Calls Dataframe in a CSV file
                save_CSV(df_calls.iloc[-calls_batch:])
                save_CSV(df_metrics.iloc[-calls_batch:], CSV_PATH_METRICS) # Save metrics

        except Exception as e: # work on python 3.x
            print(f"An error occurred: {e}")
            # Saves Calls Dataframe in a CSV file
            save_CSV(df_calls.iloc[-calls_batch:])
            save_CSV(df_metrics.iloc[-calls_batch:], CSV_PATH_METRICS)  # Save metrics