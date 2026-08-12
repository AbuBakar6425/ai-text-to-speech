---
title: AI Text-to-Speech Generator
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
python_version: "3.10"
---

# AI Text-to-Speech Generator

## Project Overview

This project is an Artificial Intelligence based Text-to-Speech
(TTS) application developed using Python, Gradio, and the
Kokoro-82M open-weight generative AI model.

The application converts user-provided text into natural-sounding
speech and provides the generated audio as a WAV file.

## Objectives

The main objectives of this project are:

- Implement a Generative AI application.
- Convert natural language text into speech.
- Use a pretrained open-weight TTS model.
- Develop a simple graphical user interface.
- Deploy the application online using Hugging Face Spaces.
- Provide generated audio as the final output.

## Technologies Used

- Python
- PyTorch
- Kokoro-82M
- Gradio
- NumPy
- SoundFile
- Hugging Face Spaces

## Model

The application uses:

Kokoro-82M

Model repository:

https://huggingface.co/hexgrad/Kokoro-82M

Kokoro is an open-weight Text-to-Speech model with
approximately 82 million parameters.

## Application Workflow

```text
User Input Text
       |
       v
Gradio Interface
       |
       v
Kokoro-82M TTS Model
       |
       v
Generated Audio
       |
       v
WAV Audio Output
