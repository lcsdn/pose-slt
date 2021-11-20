# coding: utf-8
"""
Data module
"""
import pickle
import gzip

import torch
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Optional, Tuple


def save_dataset_file(saved_object, filename):
    with gzip.open(filename, "wb") as f:
        pickle.dump(saved_object, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        keypoints_path: Optional[str],
        keypoints_fields: Optional[list],
        keypoints_dimension: Optional[int],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            fields: A tuple containing the fields that will be used for data
                in each language.
            keypoints_path: String of the path to the data file of body
                keypoints extracted from the dataset.
            keypoints_fields: A tuple containing the fields for each body part
                keypoints.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        include_keypoints = keypoints_path is not None and keypoints_fields is not None
        
        fields = [
            ("sequence", fields[0]),
            ("signer", fields[1]),
            ("sgn", fields[2]),
            ("gls", fields[3]),
            ("txt", fields[4]),
        ]
        if include_keypoints:
            fields += keypoints_fields
            if keypoints_dimension is None:
                dimension_selection = list(range(5))
            elif keypoints_dimension == 2:
                dimension_selection = [0, 1]
            elif keypoints_dimension == 3:
                dimension_selection = [2, 3, 4]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                    }
        
        if include_keypoints:
            keypoints_data = load_dataset_file(keypoints_path)

        examples = []
        for s in samples:
            sample = samples[s]
            
            if include_keypoints:
                sequence_keypoints = keypoints_data[sample["name"]]
                keypoints_features = [
                    sequence_keypoints["body_keypoints"][:, :, dimension_selection],
                    sequence_keypoints["hand_keypoints"][:, :, :, dimension_selection],
                    sequence_keypoints["face_keypoints"][:, :, dimension_selection],
                ]
            else:
                keypoints_features = []
            
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        # Removing left and right whitespaces
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                        *keypoints_features,
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)