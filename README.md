<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


# SynthAI-Datasets 🤖
SynthAI-Datasets is a repository for synthetic dataset generation using OpenAI API.

<!-- TABLE OF CONTENTS -->
- [SynthAI-Datasets 🤖](#synthai-datasets-)
  - [About the project ℹ️](#about-the-project-ℹ️)
  - [How to use ⏩](#how-to-use-)
    - [Libraries and dependencies 📚](#libraries-and-dependencies-)
    - [Example 🪄](#example-)
  - [Fine tune 🎨](#fine-tune-)
    - [Declaring Parameters](#declaring-parameters)

## About the project ℹ️

***SynthAI-Dataset*** provides an *easy* and *ready-to-use tool* to generate synthetic datasets with the OpenAI API.


Here is an example of how to create a dataset of 10 samples with *SynthAI-Dataset* by the only input of a few args.

https://github.com/CICLAB-Comillas/SynthAI-Datasets/assets/59868153/7df8aebb-8b4a-4133-8d90-987e4ebd634c


## How to use ⏩

To make things easier, here a brief guide on how to get the*SynthAI-Dataset* generator working.


### Libraries and dependencies 📚

 > It is recommended to use Python version 3.10.11 to avoid possible incompatibilities with dependencies and libraries

The first step is to you will need to install the required dependencies. Fortunately, the `requirements.txt` contains all the necessary libraries to run the code without errors. 

```bash
pip install -r requirements.txt
```

### Example 🪄

Once you have all the requirements installed, you can now run the base example with the following command in a console.

```console
python generate_dataset.py -k <your_openai_api_key> -n <samples> -b <batches>
```

As you might be wondering, the `-k` argument stands for the *OpenAI* **API Key token**. For obvious reasons 😉, we have hidden ours in the [demo](#about-the-project-ℹ️). The `-n` arg indicates the number of synthetic samples to generate and the `-b` the number of batches.

 > 🚨 Obvious reminder: Be careful not to upload your keys, you never know who may use them without permission. 

 Here is a detailed list of all possible input arguments:

  * `-k`: **OpenAI API key 🔑**. You can create this token at their OpenAI [platform](https://platform.openai.com/account/api-keys).
  * `-n`: **Number of synthetic samples** to generate.
  * `-b`: **Number of batches**. Splitting the total amount of samples in an reasonable number of batches may improve the program efficiency. In addition, the program automatically saves the generated data each time a batch is finished.
  * `-p`: **Output CSV path** for the generated dataset. By default, a file `dataset.csv` is created in the project directory. If the path provided is from an existing CSV (from a previous generation), then add the new samples to the end of that same CSV (😎).
  * `-j`: **Parameter JSON Path**. This file contains all the possible params for each category used in prompt construction. By default it will take the `params.json` located at project directory. 
  * `-l`: **Budget limit** 💰 for the dataset generation run. The generator will stop when either total number of samples is finished or the accumulated cost of the requests made reaches this limit. In development 🏗️: If you are planning to change to other OpenAI model, be aware that prices may vary.  
  * `-m`: **Flag for metrics saving**. Some of this metrics are the consumed tokens for prompt, completion and in total for each generated sample, as well as the API response time, the sample and batch number. By default is True. To not save metrics include `--no-m` in command.

 > 💡 Tip: We strongly recommend you to use the **budget limit** arg in order to prevent unexpected charges in your OpenAI account.


 ## Fine tune 🎨

 The base example is fine, but I suppose that you are here to create your own custom dataset, relax, you are only a few changes away from getting it.

 To make things simpler, this section will detail the necessary steps to make a custom dataset with an example. In this case, we will prepare the code to generate a synthetic dataset of **dice rolls** 🎲.


 ### Declaring Parameters

First, you need to edit the `params.json` file for your own case, or if you prefer so, you can create your own JSON file, but remember to indicate this path in the `-j` arg.

In short, this JSON has to contain avery possible param value of each category, in order to generate diverse prompts by the combinations of randomly selected params. 

For our example, the `params.json` should look like this:

```json
{
    "Dice1": {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6"       
    },
    "Dice2": {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6"       
    }
}
```

The JSON has two params, each of one

