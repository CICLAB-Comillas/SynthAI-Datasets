import argparse
import json
import time
from os.path import dirname, join, abspath, exists
from retry import retry
from random import choice, randint
from typing import Dict, List, Tuple
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

from prompt import generate_prompt
from completion import process_completion, ResponseFormatException

# Path of the current file
FILE_DIR = dirname(abspath(__file__))

### ----------------------------------ARGUMENTS---------------------------------------------

parser = argparse.ArgumentParser()

# API Key
parser.add_argument('-k','--api_key', help='OpenAI API key')

# Number of data samples to generate
parser.add_argument('-n', '--samples',help='Number of data samples to generate', type= int, required=True)

# Path of the CSV file
parser.add_argument('-p','--path', help='Output CSV Path. By default output CSV is generated at this file level (same directory)', default=join(FILE_DIR,"dataset.csv"))

# Path of the JSON file with the parameters
parser.add_argument('-j','--json', help='Path of prompt parameters JSON', default=join(FILE_DIR,"params.json"))

# Number of batches
parser.add_argument('-b','--batches', help='Number of batches', type=int, default=1)

# Budget limit
parser.add_argument('-l','--limit', help='Budget limit for dataset generation', type=float, default=None)

# Dataset generation metrics flag (Default==True -> saves metrics as .csv, "--no-m"==False -> no metrics)
parser.add_argument("-m","--save_metrics", help="Creates a .csv with the synthetic dataset generation metrics (`prompt_tokens`,`completion_tokens`,`total_tokens`,`time`,`sample`,`batch`)", action=argparse.BooleanOptionalAction, default=True)

args = parser.parse_args()

openai.api_key = args.api_key
SAMPLES = args.samples
CSV_PATH = args.path
JSON_PATH = args.json
BATCHES = args.batches
BATCHES = BATCHES if SAMPLES>=BATCHES else 1
BUDGET_LIMIT = args.limit 
METRICS = args.save_metrics

CSV_PATH_METRICS = CSV_PATH.replace(".csv","_metrics.csv")

### ---------------------- OpenAI Params & prices -------------------------------------------

API_PRICE_INPUT_1K_TOKENS = 0.0015 #$
API_PRICE_OUTPUT_1K_TOKENS = 0.002 #$

### ----------------------------------EXCEPTIONS---------------------------------------------

class BudgetLimitReached(Exception):
    """Controls the budget limit"""

    def __init__(self, budget_limit):
        self.budget_limit = budget_limit
        self.msg = f"${self.budget_limit} limit reached."
        super().__init__(self.msg)

### ----------------------------------FUNCTIONS---------------------------------------------

# Reading parameters from JSON file
with open(JSON_PATH, 'r') as f:
    PARAMS = json.load(f)

def distribute_in_samples(total_samples: int, batches: int) -> List[int]:
    """ Tries to equally distribute the total samples in the number of indicated batches. Args:
        * `total_samples`: Total number of samples
        * `batches`: Number of output batches

        Returns a List with the amount of samples asigned to each batch. Output list len is always equal to `batches`.
    """

    samples_per_batch = total_samples // batches  # Computes the integer division
    remainder = total_samples % batches  # Computes the remainder of the division (modulo operator)

    batches = [samples_per_batch] * batches  # Creates a list with the quotient repeated BATCHES times

    # Distributes the remainder in the batches
    for i in range(remainder):
        batches[i] += 1

    return batches

@retry((APIError, Timeout, RateLimitError, APIConnectionError, ServiceUnavailableError), delay=5, backoff=2, max_delay=1800)
def send_to_openai(prompt: str) -> Dict[str, str] or None:
    """ Sends a request message with `prompt` body to OpenAI API. Params
        * `prompt`: text input to API
        Returns the API completion (if received).
    """
    completion = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ],
        temperature=0.7,
        max_tokens=1024,
    )

    return completion if 'choices' in completion else None
    
@retry(ResponseFormatException, tries=10)
def send_and_process_response(prompt: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """ Sends a prompt to OpenAI's API and processes the completion (if) received. 
        If a format error happens while handling response (was not generated correctly with the expected format), rules out this completion and generates a new one. This process is done over and over up to 10 times. 
        Args:
        * `prompt`: Request body for completion.

        Returns the processed completion as dictionary as well as the metrics (dict). Output may vary depending on completion processing function.
        
        In the case that max number of tries is reached, then it is supposed that either the completion processing function or prompt generation are not as tune as expected (code review recommended).
    """

    # Request to API
    completion = send_to_openai(prompt)

    # Processing completion
    sample_dict = process_completion(completion)

    # Get the usage (metrics) from completion response
    metrics_dict = dict(completion.usage)

    return sample_dict, metrics_dict

def generate_sample(df_samples: DataFrame, sample: int, batch: int,  df_metrics: DataFrame = None) -> DataFrame:
    """ Generates a new row in the dataset with a new sample. Args:
        * `df_samples`: Samples DataFrame. New generated sample will be appended last
        * `sample`: This new sample number over the total
        * `batch`: Batch of this new sample
        * `df_metrics`: [Optional] Metrics DataFrame. The metrics of this new sample with be appended last
    """

    # ------------ Step 1 Building prompt ------------
    # Create task
    progress_step_batch_id_0 = progress_step_batch.add_task('',action='Generating prompt...')

    # Prompt generation
    prompt, random_params = generate_prompt(PARAMS)

    # Update batch progress
    progress_step_batch.update(progress_step_batch_id_0, action='[bold green]Prompt Generated[✔]')
    progress_step_batch.stop_task(progress_step_batch_id_0)

    # ------------ Step 2 Performing API request ------------
    # Create task
    progress_step_batch_id_1 = progress_step_batch.add_task('',action='Performing OpenAI API request...')

    # OpenAI API sample timer starts
    init = time.time()

    # API sample
    sample_dict, metrics_dict = send_and_process_response(prompt) # Response

    # OpenAI API sample timer ends
    elapsed_time = round(time.time()-init, 2)

    # Adds the params randomly selected in each category to the sample dict
    for category in list(random_params.keys()):
        # Add the remaining fields
        sample_dict.update(random_params[category])

    metrics_dict["time"] = elapsed_time # Elapsed time for the request
    metrics_dict["sample"] = sample
    metrics_dict["batch"] = batch

    # Create sample row
    row_sample = pd.Series(sample_dict).to_frame().T

    # Concatenate the new row at the end of the sample Dataframe
    df_samples = pd.concat([df_samples, row_sample], ignore_index=True)

    
    # Create metrics row
    row_metrics = pd.Series(metrics_dict).to_frame().T
    row_metrics = row_metrics.astype(dtype={'prompt_tokens': "int64",'completion_tokens': "int64",'total_tokens': "int64",'time': "float64",'sample': "int64",'batch': "int64"})

    # Concatenate the new row at the end of the Metrics Dataframe
    df_metrics = pd.concat([df_metrics, row_metrics], ignore_index=True)

    # Update batch progress
    progress_step_batch.update(progress_step_batch_id_1, action='[bold green]OpenAI API completion[✔]')
    progress_step_batch.stop_task(progress_step_batch_id_1)

    # Remove task responses
    progress_step_batch.remove_task(progress_step_batch_id_0)
    progress_step_batch.remove_task(progress_step_batch_id_1)

    return df_samples, df_metrics

def save_CSV(df: DataFrame, path: str = CSV_PATH) -> None:
    """ Saves the Dataframe as a CSV file. If the file already exists, it concatenates the new dataframe"""
    if exists(path):
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

def update_remaining_time(elapsed_time: int, completed_samples: int, remaining_samples: int) -> str:
    """ Computes the remaining time in the following format hh:mm:ss"""

    time_remaining_sec = remaining_samples*int(elapsed_time/completed_samples) if completed_samples else '-:--:--'
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
    TextColumn('Spent: [bold red]{task.fields[accumulated_cost]} [/bold red](limit: ${task.fields[limit]:.2f})')
)

# Batch total progress bar
total_progress_batch = Progress(
    TimeElapsedColumn(),
    TextColumn('[bold blue] Batch {task.fields[name]}'),
    BarColumn(),
    TextColumn('({task.completed}/{task.total}) samples generated')
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

# Autorender for bar group
live = Live(group)

if __name__ == "__main__":

    # Balanced distribution of the samples in batches
    batches = distribute_in_samples(SAMPLES,BATCHES)

    _df_samples_columns = list(PARAMS.keys())
    for param in list(PARAMS.keys()):
        _df_samples_columns += list(PARAMS[param].keys())

    # Create Dataframe
    df_samples = DataFrame(columns=_df_samples_columns)

    df_metrics = DataFrame(columns=['prompt_tokens','completion_tokens','total_tokens','time','sample','batch'])

    cumulative_cost = 0.0 # Initial cumulative cost

    # Total progress task
    total_progress_id = total_progress.add_task(description=f'[bold #AAAAA](batch 0 of {BATCHES})', total=SAMPLES, name = '-:--:--', accumulated_cost="$0.00", limit=BUDGET_LIMIT)

    with live:
        try:
            # Process time starts
            initial_time_sec = time.time()

            for batch, samples_batch in enumerate(batches):
                # Update batch number
                total_progress.update(total_progress_id, description=f'[bold #AAAAA](batch {batch+1} of {BATCHES})')
                
                # Create complete batch task
                total_progress_batch_id = total_progress_batch.add_task('',total=samples_batch, name=batch)

                for sample in range(samples_batch):
                    # Generate sample
                    df_samples, df_metrics = generate_sample(df_samples, sample, batch, df_metrics)

                    # Update the number of samples
                    total_progress_batch.update(total_progress_batch_id, advance=1)

                    # Update the total progress
                    total_progress.update(total_progress_id, advance=1, refresh=True)

                    # Update remaining time
                    elapsed_time_sec = int(time.time()-initial_time_sec)
                    completed_samples = total_progress._tasks[total_progress_id].completed
                    remaining_samples = SAMPLES - completed_samples
                    total_progress.update(total_progress_id, name = update_remaining_time(elapsed_time_sec, completed_samples, remaining_samples))

                    # Update cost
                    cumulative_cost = compute_cumulative_cost(cumulative_cost, df_metrics.iloc[-1]["prompt_tokens"], df_metrics.iloc[-1]["completion_tokens"])
                    total_progress.update(total_progress_id, accumulated_cost = f"${ceil(cumulative_cost*100)/100}")

                    # If the limit is reached, the process is interrumpted
                    if BUDGET_LIMIT and cumulative_cost > BUDGET_LIMIT:
                        raise BudgetLimitReached(BUDGET_LIMIT)

                # Saves samples Dataframe in a CSV file
                save_CSV(df_samples.iloc[-samples_batch:])
                if METRICS:
                    save_CSV(df_metrics.iloc[-samples_batch:], CSV_PATH_METRICS) # Save metrics

        except Exception as e: # work on python 3.x
            print(f"An error occurred: {e}")
            # Saves samples Dataframe in a CSV file
            save_CSV(df_samples.iloc[-samples_batch:])
            if METRICS:
                save_CSV(df_metrics.iloc[-samples_batch:], CSV_PATH_METRICS)  # Save metrics