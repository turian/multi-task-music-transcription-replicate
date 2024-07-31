from beam import Image, endpoint, function


def _load_model(model_type):
    from inference_model import InferenceModel

    # Set model based on the choice
    checkpoint_path = f"/checkpoints/{model_type}/"
    model = InferenceModel(checkpoint_path, model_type)
    return model


image = Image(
    python_version="python3.10",
    # python_packages=[ ],
    commands=[
        "pip3 install setuptools==70.3.0",
        "python3 -m pip install gsutil",
        "gsutil -q -m cp -r gs://mt3/checkpoints .",
        "gsutil -q -m cp gs://magentadata/soundfonts/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2 .",
        "apt-get update -y && apt-get install libfluidsynth3 build-essential libasound2-dev libjack-dev -y",
        "git clone --branch=main https://github.com/magenta/mt3 && cd mt3 && python3 -m pip install jax[cuda12_local] nest-asyncio pyfluidsynth==1.3.0 -e . -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    ],
    base_image="docker.io/nvidia/cuda:12.2.2-runtime-ubuntu22.04",
)


@function(cpu=1.0, memory=128, gpu="T4", image=image)
def predict(
    audio_file: str,
    model_type: str,
) -> str:
    import os
    import shutil
    import tempfile

    import note_seq
    import numpy as np
    import tensorflow as tf

    # Make a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            model = _load_model(model_type)
            # audio, sr = note_seq.audio_io.wav_data_to_samples_librosa(
            audio = note_seq.audio_io.load_audio(audio_file, sample_rate=16000)
            est_ns = model(audio)
            midi_file = os.path.join(temp_dir, os.path.basename(audio_file) + ".mid")
            note_seq.sequence_proto_to_midi_file(est_ns, midi_file)
            return midi_file
        except:
            raise
        finally:
            # Delete the temporary directory
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    predict.remote("file_example_WAV_1MG.wav", "mt3")
