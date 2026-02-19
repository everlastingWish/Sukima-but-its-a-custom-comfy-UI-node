# Sukima Custom Node for ComfyUI
This repository provides a custom node for ComfyUI based on [Sukima](https://github.com/hitomi-team/sukima). [Sukima](https://github.com/hitomi-team/sukima) was originally designed as a ready-to-deploy container implementing a REST API for Language Models with a focus on scalability.

This implementation adapts [Sukima](https://github.com/hitomi-team/sukima) into a ComfyUI node, featuring updated code to ensure compatibility with modern PyTorch and Transformers libraries (as of early 2026).

# Features
- Modern Compatibility: Fully patched to work with the latest versions of PyTorch and Hugging Face Transformers.
- Lightweight: Contains the logic and node structure without forcing a specific model download.

# Installation
1. Clone this repo into `custom_nodes` folder.
2. Install dependencies: `pip install -r requirements.txt`

# Usage:
Once installed, the node will be available within the ComfyUI menu.

Note that this repository does not contain pre-trained models. You must provide your own compatible model path as required by the setting.
<img width="767" height="797" alt="Example Workflow" src="https://github.com/user-attachments/assets/fbdfe8a5-54f9-41f0-afa9-eac3aacbe8c9" />

# Project History
- Original Completion: December 5th, 2025
- Public Release: February 20th, 2026
- Latest Requirements Update: February 20th, 2026

# License
This project is an integration wrapper. All original logic and code from Sukima remain under its original [GPL-2.0 License](LICENSE). Modifications and ComfyUI node integration logic provided in this repo are also distributed under [GPL-2.0](LICENSE) to maintain compatibility.
