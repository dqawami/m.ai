# Welcome to Project m.AI

m.AI is an open source machine learning library whose goal is to optimize AI algorithms for graphics cards. Unlike other AI libraries, m.AI only needs the current layer of a machine learning model to be loaded into a GPU's memory instead of the entire model. This change allows for anyone to train practically any model on a graphics card instead of being limited by the maximum memory size of a GPU. This project is still in early stages, so all help and feedback would be greatly appreciated!

## Features
Currently this library only supports either single-GPU training or CPU training. This library also supports basic linear and image recognition models for training. Exact supported layers can be found in the mAI folder.

## Feature Requests
Please request for any features in the Discussions tab, and they'll be added to the list for development. If any bugs are discovered, please report them in Issues. 

## System Requirements
Linux OS needed
- Mac and Windows support coming soon

NVIDIA graphics card is required to run on GPU
- Unsure if GPU/NVCC is needed for CPU only code, would appreciate feedback on this

## Next Steps
- [ ] Library installation and installation instructions
- [ ] Layer and model verification
- [ ] Adding unit tests and error checking
- [ ] Adding distributed GPU option
- [ ] Fleshing out linear and image recognition libraries
- [ ] Adding more machine learning layers such as NLP, LLMs, and Time-Series
- [ ] Adding more model and model types
