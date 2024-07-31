import note_seq
import numpy as np
import tensorflow as tf
from cog import BasePredictor, Input, Path

from inference_model import InferenceModel


class Predictor(BasePredictor):
    def setup(self):
        pass

    def _load_model(self, model_type):
        # Set model based on the choice
        checkpoint_path = f"/checkpoints/{model_type}/"
        model = InferenceModel(checkpoint_path, model_type)
        return model

    def predict(
        self,
        audio_file: Path = Input(description="Input audio file"),
        model_type: str = Input(
            "Model type: ismir2021 for piano, mt3 for multi-instrument",
            choices=["ismir2021", "mt3"],
        #    default="mt3",
        ),
    ) -> Path:
        model = self._load_model(model_type)
        audio, sr = note_seq.audio_io.wav_data_to_samples_librosa(
            audio_file.read(), sample_rate=16000
        )
        est_ns = model(audio)
        midi_file = "/tmp/transcribed.mid"
        note_seq.sequence_proto_to_midi_file(est_ns, midi_file)
        return midi_file
