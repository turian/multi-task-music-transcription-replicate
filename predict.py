import cog
import note_seq
import numpy as np
import tensorflow as tf
from google.colab import files
from mt3 import InferenceModel


class Model(cog.Model):
    def setup(self, model_type):
        # Set model based on the choice
        checkpoint_path = f"/models/checkpoints/{model_type}/"
        self.model = InferenceModel(checkpoint_path, model_type)

    @cog.input("audio_file", type=cog.Path, help="Input audio file")
    @cog.input(
        "model_type",
        type=str,
        options=["ismir2021", "mt3"],
        default="mt3",
        help="Model type: ismir2021 for piano, mt3 for multi-instrument",
    )
    def predict(self, audio_file, model_type):
        self.setup(model_type)
        audio, sr = note_seq.audio_io.wav_data_to_samples_librosa(
            audio_file.read(), sample_rate=16000
        )
        est_ns = self.model(audio)
        midi_file = "/tmp/transcribed.mid"
        note_seq.sequence_proto_to_midi_file(est_ns, midi_file)
        return cog.File(midi_file)
