# Welcome to Project m.AI

m.AI is an open source machine learning library whose goal is to optimize AI algorithms for graphics cards. m.AI uses heterogeneous memory so that only the current layer of a machine learning model needs to be loaded into a GPU's memory instead of the entire model. This change allows for anyone to train practically any model on any graphics card instead of being limited by the maximum memory size of a GPU. This project is still in early stages, so all help and feedback would be greatly appreciated!

## Features
Currently this library supports single-GPU training and CPU training. This library also supports basic linear and image recognition models for training. Exact supported layers can be found in the mAI folder.

## Feature Requests
If any bugs are discovered, please report them in the Issues tab. If you have any suggestions for features to be added, please also add them to Issues and they'll be added to the dev list.

## System Requirements
Ubuntu supported
- Mac and Windows support coming soon
-Other Linux distros need to be checked

NVIDIA graphics card is required to run on GPU
- Unsure if GPU/NVCC is needed for CPU only code, would appreciate feedback on this

## Next Steps
- [ ] Library installation and installation instructions
- [ ] Layer and model verification
- [ ] Adding unit tests and error checking
- [ ] Adding distributed GPU option
- [ ] Fleshing out linear and image recognition libraries
- [ ] Adding more machine learning layers such as NLP and LLMs
- [ ] Adding more model and model types
