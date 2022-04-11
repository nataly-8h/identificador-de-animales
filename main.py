from HMMTrainer import HMMTrainer
from librosa.feature import mfcc
import librosa
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


def get_hmm_models(hmm_models, folder, num_mfcc_features):
    for dirname in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, dirname)

        if not os.path.isdir(subfolder_path):
            continue

        label = subfolder_path[subfolder_path.rfind("/") + 1:]
        X = np.array([])
        y_words = []

        for filename in [i for i in os.listdir(subfolder_path) if i.endswith(".wav")]:
            filepath = os.path.join(subfolder_path, filename)
            sampling_freq, audio = librosa.load(filepath)
            mfcc_features = mfcc(y=sampling_freq, sr=audio)

            if len(X) == 0:
                X = mfcc_features[:, :num_mfcc_features]
            else:
                X = np.append(X, mfcc_features[:, :num_mfcc_features], axis=0)
            y_words.append(label)

        print('X.shape =', X.shape)

        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X)
        hmm_models.append((hmm_trainer, label))
        hmm_trainer = None


def test_input(input_tests):
    for test in os.listdir(input_tests):
        filepath = os.path.join(input_tests, test)

        if not os.path.isfile(filepath) or not test.endswith(".wav"):
            continue

        sampling_freq, audio = librosa.load(filepath)

        # Extract MFCC features
        mfcc_features = mfcc(y=sampling_freq, sr=audio)
        mfcc_features = mfcc_features[:, : num_mfcc_features]

        scores = []

        for item in hmm_models:
            hmm_model, label = item
            score = hmm_model.get_score(mfcc_features)
            scores.append(score)

        index = np.array(scores).argmax()
        print("\nInput: ", filepath)
        print("Predicted:", hmm_models[index][1])


if __name__ == "__main__":
    hmm_models = []
    input_folder = "audio"
    input_tests = "tests"
    num_mfcc_features = 15

    get_hmm_models(hmm_models, input_folder, num_mfcc_features)
    test_input(input_tests)
