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

## About the project ℹ️

***SynthAI-Dataset*** provides an *easy* and *ready-to-use tool* to generate synthetic datasets with the OpenAI API.


Here is an example of how to create a dataset of 10 samples with *SynthAI-Dataset* by the only input of a few args.

https://github.com/CICLAB-Comillas/SynthAI-Datasets/assets/59868153/7df8aebb-8b4a-4133-8d90-987e4ebd634c


## How to use ⏩

*SynthAI-Dataset* was designed

Below you can see a demo.


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

TODO: Add a brief definition of each argument

 ## Fine tune 🎨

 The base example is fine, but I suppose that you are here for your own custom dataset, so a few changes must be done. The are basically **2 main things** you must change in other to accomplish your objective:

 - Declare your own random parameters for the prompt
 - 
