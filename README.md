# SynthAI-Datasets 🤖
SynthAI-Datasets is a repository for synthetic dataset generation using the OpenAI API.

<!-- TABLE OF CONTENTS -->
- [SynthAI-Datasets 🤖](#synthai-datasets-)
  - [About the project ℹ️](#about-the-project-ℹ️)
  - [How to use ⏩](#how-to-use-)
    - [Libraries and dependencies 📚](#libraries-and-dependencies-)
    - [Example 🪄](#example-)
  - [Fine tune 🎨](#fine-tune-)
    - [Parameter setting 🎲](#parameter-setting-)
    - [Prompt creation ⌨️](#prompt-creation-️)
    - [Completion processing 📦](#completion-processing-)
  - [Developers 🔧](#developers-)

## About the project ℹ️

***SynthAI-Dataset*** provides an *easy* and *ready-to-use tool* for generating synthetic datasets with the OpenAI API.


Here is an example of how to create a dataset of 10 samples with *SynthAI-Dataset* by just setting a few args.

https://github.com/CICLAB-Comillas/SynthAI-Datasets/assets/59868153/7df8aebb-8b4a-4133-8d90-987e4ebd634c


## How to use ⏩

To make things easier, here is a brief guide on how to get the *SynthAI-Dataset* generator working.


### Libraries and dependencies 📚

 > It is recommended to use Python version 3.10.11 to avoid possible incompatibilities with dependencies and libraries

The first step is to install the required dependencies. Fortunately, the `requirements.txt` file contains all the necessary libraries to run the code without errors. 

```bash
pip install -r requirements.txt
```

### Example 🪄

Once you have all the requirements installed, you can run the base example with the following command in a console.

```console
python generate_dataset.py -k <your_openai_api_key> -n <samples> -b <batches>
```

As you might be wondering, the `-k` argument stands for the *OpenAI* **API Key token**. For obvious reasons 😉, we have hidden ours in the [demo](#about-the-project-ℹ️). The `-n` arg indicates the number of synthetic samples to generate and the `-b` the number of batches.

 > 🚨 Obvious reminder: Be careful not to upload your keys, you never know who may use them without permission. 

 Here is a detailed list of all possible input arguments:

  * `-k`: **OpenAI API key 🔑**. You can create this token in the OpenAI [platform](https://platform.openai.com/account/api-keys).
  * `-n`: **Number of synthetic samples** to generate.
  * `-b`: **Number of batches**. Splitting the total amount of samples in a reasonable number of batches may improve the program's efficiency. In addition, the program automatically saves the generated data each time a batch is finished.
  * `-p`: **Output CSV path** for the generated dataset. By default, a file `dataset.csv` is created in the project directory. If the path provided already contains an existing CSV file with the same name (from a previous generation), the program appends the new samples to the end of that same file (😎).
  * `-j`: **Parameter JSON Path**. This file contains all the possible params for each category used for the prompt construction. By default, it will take the `params.json` file located at the project directory. 
  * `-l`: **Budget limit** 💰 for the dataset generation run. The generator will stop when either the total number of samples is reached or the accumulated cost of the requests made gets to this limit. In development 🏗️: If you are planning to use another OpenAI model for the requests, be aware that prices may vary.  
  * `-m`: **Flag for metrics saving**. Some of these metrics include the consumed tokens for the prompt, completion and the total number for each generated sample, as well as the API response time and the sample and batch numbers. By default is True. To not save metrics include `--no-m` in the command.

 > 💡 Tip: We strongly recommend to use the **budget limit** arg in order to prevent unexpected charges in your OpenAI account.


 ## Fine tune 🎨

 The base example is fine, but we suppose that you are here to create your own custom dataset, relax, you are only a few changes away from getting it.

To make it easier, this section details how to create your own dataset in a few simple steps, following an example case: *generate a dataset of customer service calls of an electricity supply company, including the **conversation** held and its **summary***.


 ### Parameter setting 🎲

First, you need to edit the `params.json` file for your own use case. Another option is to create your own JSON file, but remember to indicate this path in the `-j` arg.

In short, this JSON has to contain every possible param value of each category, in order to generate diverse prompts by the combinations of randomly selected params.

The `params.json` file for the example case looks like this:

```json
{
    "Call": {
        "Attitude": [
            "positive",
            "neutral",
            "negative",
            "aggresive",
            "assertive"
        ],
        "Conversation": [
            "long",
            "brief",
            "normal"
        ],
        "Problem": [
            "an electricity supply cut off",
            "a gas supply cut-off",
            "the meter",
            "an unexpected charge on the electricity bill",
            "an unexpected charge on gas bill",
            "a change of electricity tariff",
            "a change of gas tariff",
            "a rate recommendation",
            "the solar panel installation"
        ],
        "Solved": [
            "is",
            "is not"
        ],
        "Manner": [
            "successfully",
            "kindly",
            "patiently",
            "quickly"
        ]
    },
    "Summary": {
        "Extension": [
            "brief",
            "concise",
            "extensive",
            "in depth",
            "detailed"
        ]
    }
}
```

The JSON file has two main keys, each corresponding to a part of the dataset to be generated: conversation (`Call`) and summary (`Summary`). 

Each main key might contain one or more keys, which are in fact the parameters of that field. For example, the conversation has 5 parameters, such as the *client mood* (`Attitude`), the *reason for calling* (`Problem`) and whether it was solved or not (`Solved`). 

The *summary* also has parameters, which in this case is only one, the *extension* (`Extension`).  

 ### Prompt creation ⌨️

 The second step is to create your *prompt*, which is the instruction provided to the OpenAI API to generate a completion. The main objective is to create different prompts to achieve randomness in the generated data samples. To do so, it is necessary to define a function that randomly selects a value for each parameters defined in the previous step. 

 `generate_dataset.py` imports the function ***generate_prompt*** from `prompt.py`. So, in order to code your own implementation for the function, be sure to edit this last file.

To build the prompt, three steps must be followed (see `prompt.py`). The first one is to select a random set of params each time the function is called. For some params it may not be feasible to define each possible value (ex. *phone number*), so it is necessary to code functions that randomly generate this data.

The second step is to set up the prompt body for each field. For the client service dataset example, the prompt body `Call` is presented below, in which `call_params` is a dict with the randomly selected params:

```python
f"Generate a {call_params['Conversation']} phone call in which a client with phone number {call_params['Phone']} and address {call_params['Address']}, with a {call_params['Attitude']} attitude, contacts the supplying company {call_params['Company']} to report a problem with {call_params['Problem']}. Finally, the call {call_params['Solved']} solved {call_params['Manner']}."
```

Depending on the parameters, the prompt will be different, so as the response from OpenAI. However, the body of the prompt remains the same.
  > 💡 We strongly recommend testing different versions of the prompt with a few number of samples before running the program for the total target samples. This will allow you to check if the result is what you wanted and, thus, it will avoid wasting unnecessary credits from your OpenAI account.

Last but not least, it is convenient to include a brief indication for OpenAI to generate the sample with a desired format. By default, the `generate_prompt` example tells the API to include [*key] before each part, where *key is the name of that part (Ex. [Call] or [Summary])

### Completion processing 📦

After sending the request with the prompt created in the last step, you will receive a response from the API. If an exception occurs, don't panic, `generate_dataset.py` has mechanisms to prevent sample generation from aborting in case of API connection issues or errors in the response format. 

Similarly to the prompt logic, the response processing logic is separated from the main code, it can be found in the `completion.py` file, which is called through the `process_completion` method. As its name indicates, this function receives the API response and processes it, returning a dictionary with the different fields of the sample (ex. conversation & summary).

Again, to generate your own dataset it is necessary that you set up your own implementation of the `completion.py` function.

  > 🚨 In case the response is not generated in the expected format, the program will try to generate a new one from scratch. If this happens more than 10 times, the program will be interrupted, since the problem will most likely be due to some inconsistency in your code.

## Developers 🔧

We would like to thank you for taking the time to read about this project.

If you have any suggestions💡 or doubts❔ **please do not hesitate to contact us**, kind regards from the developers:
  * [Jaime Mohedano](https://github.com/Jatme26)
  * [David Egea](https://github.com/David-Egea)
  * [Ignacio de Rodrigo](https://github.com/nachoDRT)