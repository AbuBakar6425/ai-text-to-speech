import os
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

APP_TITLE = "AI Text-to-Speech Generator"
MAX_CHARACTERS = 1000

# Kokoro English voices
VOICES = {
    "American Female - Heart": "af_heart",
    "American Female - Bella": "af_bella",
    "American Female - Nicole": "af_nicole",
    "American Male - Adam": "am_adam",
    "American Male - Michael": "am_michael",
    "British Female - Emma": "bf_emma",
    "British Female - Isabella": "bf_isabella",
    "British Male - George": "bm_george",
    "British Male - Lewis": "bm_lewis",
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("======================================")
print("AI Text-to-Speech Generator")
print("Kokoro-82M")
print("Device:", DEVICE)
print("======================================")


# ============================================================
# LOAD KOKORO PIPELINE
# ============================================================

# 'a' = American English
# 'b' = British English

pipeline_a = KPipeline(lang_code="a")
pipeline_b = KPipeline(lang_code="b")


# ============================================================
# TEXT TO SPEECH FUNCTION
# ============================================================

def generate_speech(text, voice_name, speed):
    """
    Convert input text into speech using Kokoro-82M.

    Parameters
    ----------
    text : str
        User's input text.

    voice_name : str
        Selected Kokoro voice.

    speed : float
        Speech generation speed.

    Returns
    -------
    str
        Path to generated WAV file.
    """

    # --------------------------------------------------------
    # Validate text
    # --------------------------------------------------------

    if text is None or not text.strip():
        raise gr.Error("Please enter some text.")

    text = text.strip()

    if len(text) > MAX_CHARACTERS:
        raise gr.Error(
            f"Text is too long. Please keep it under "
            f"{MAX_CHARACTERS} characters."
        )

    # --------------------------------------------------------
    # Get voice ID
    # --------------------------------------------------------

    voice_id = VOICES.get(voice_name)

    if voice_id is None:
        raise gr.Error("Please select a valid voice.")

    # --------------------------------------------------------
    # Select language pipeline
    # --------------------------------------------------------

    if voice_id.startswith("a"):
        pipeline = pipeline_a
    else:
        pipeline = pipeline_b

    # --------------------------------------------------------
    # Generate speech
    # --------------------------------------------------------

    audio_chunks = []

    try:

        for _, _, audio in pipeline(
            text,
            voice=voice_id,
            speed=float(speed)
        ):

            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()

            audio_chunks.append(np.asarray(audio))

    except Exception as error:

        print("Generation error:", error)

        raise gr.Error(
            f"Speech generation failed: {str(error)}"
        )

    # --------------------------------------------------------
    # Validate generated audio
    # --------------------------------------------------------

    if not audio_chunks:
        raise gr.Error(
            "No audio was generated. Please try another text."
        )

    # --------------------------------------------------------
    # Combine audio chunks
    # --------------------------------------------------------

    final_audio = np.concatenate(audio_chunks)

    # --------------------------------------------------------
    # Create temporary WAV file
    # --------------------------------------------------------

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    )

    output_file.close()

    sf.write(
        output_file.name,
        final_audio,
        24000
    )

    return output_file.name


# ============================================================
# CLEAR FUNCTION
# ============================================================

def clear_text():
    return ""


# ============================================================
# GRADIO USER INTERFACE
# ============================================================

with gr.Blocks(
    title=APP_TITLE,
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # 🎙️ AI Text-to-Speech Generator

        Convert written text into natural-sounding speech using
        the **Kokoro-82M open-weight Text-to-Speech model**.

        Enter your text, select a voice, adjust the speed,
        and generate an audio file.
        """
    )

    gr.Markdown(
        """
        **Model:** Kokoro-82M  
        **Task:** Text-to-Speech  
        **Framework:** Python + Gradio  
        **Deployment:** Hugging Face Spaces
        """
    )

    # --------------------------------------------------------
    # Text Input
    # --------------------------------------------------------

    text_input = gr.Textbox(
        label="Enter Text",
        placeholder=(
            "Type your text here...\n\n"
            "Example: Artificial intelligence is transforming "
            "healthcare by helping doctors analyze large amounts "
            "of medical information."
        ),
        lines=8,
        max_length=MAX_CHARACTERS
    )

    character_counter = gr.Markdown(
        f"Maximum characters: **{MAX_CHARACTERS}**"
    )

    # --------------------------------------------------------
    # Voice and Speed
    # --------------------------------------------------------

    with gr.Row():

        voice_dropdown = gr.Dropdown(
            choices=list(VOICES.keys()),
            value="American Female - Heart",
            label="Select Voice"
        )

        speed_slider = gr.Slider(
            minimum=0.5,
            maximum=2.0,
            value=1.0,
            step=0.1,
            label="Speech Speed",
            info="1.0 = normal speed"
        )

    # --------------------------------------------------------
    # Buttons
    # --------------------------------------------------------

    with gr.Row():

        generate_button = gr.Button(
            "🔊 Generate Speech",
            variant="primary"
        )

        clear_button = gr.Button(
            "🗑️ Clear"
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    audio_output = gr.Audio(
        label="Generated Speech",
        type="filepath",
        autoplay=False
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    gr.Markdown(
        """
        ### How it works

        1. Enter text into the input box.
        2. Select an AI voice.
        3. Adjust speech speed if required.
        4. Click **Generate Speech**.
        5. Listen to the generated audio.
        6. Download the WAV file.

        ### About the Model

        This application uses **Kokoro-82M**, an open-weight
        text-to-speech model containing approximately 82 million
        parameters.

        Model:
        https://huggingface.co/hexgrad/Kokoro-82M

        The model is released under the Apache-2.0 license.
        """
    )

    # --------------------------------------------------------
    # Button Events
    # --------------------------------------------------------

    generate_button.click(
        fn=generate_speech,
        inputs=[
            text_input,
            voice_dropdown,
            speed_slider
        ],
        outputs=audio_output
    )

    clear_button.click(
        fn=clear_text,
        inputs=[],
        outputs=text_input
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    demo.launch()
