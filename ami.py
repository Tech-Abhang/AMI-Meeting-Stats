# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The AMI Meeting Corpus consists of 100 hours of meeting recordings. The recordings use a range of signals
synchronized to a common timeline. These include close-talking and far-field microphones, individual and
room-view video cameras, and output from a slide projector and an electronic whiteboard. During the meetings,
the participants also have unsynchronized pens available to them that record what is written. The meetings
were recorded in English using three different rooms with different acoustic properties, and include mostly
non-native speakers.
"""

import os

import datasets

_CITATION = """\
@inproceedings{10.1007/11677482_3,
author = {Carletta, Jean and Ashby, Simone and Bourban, Sebastien and Flynn, Mike and Guillemot, Mael and Hain, Thomas and Kadlec, Jaroslav and Karaiskos, Vasilis and Kraaij, Wessel and Kronenthal, Melissa and Lathoud, Guillaume and Lincoln, Mike and Lisowska, Agnes and McCowan, Iain and Post, Wilfried and Reidsma, Dennis and Wellner, Pierre},
title = {The AMI Meeting Corpus: A Pre-Announcement},
year = {2005},
isbn = {3540325492},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/11677482_3},
doi = {10.1007/11677482_3},
abstract = {The AMI Meeting Corpus is a multi-modal data set consisting of 100 hours of meeting
recordings. It is being created in the context of a project that is developing meeting
browsing technology and will eventually be released publicly. Some of the meetings
it contains are naturally occurring, and some are elicited, particularly using a scenario
in which the participants play different roles in a design team, taking a design project
from kick-off to completion over the course of a day. The corpus is being recorded
using a wide range of devices including close-talking and far-field microphones, individual
and room-view video cameras, projection, a whiteboard, and individual pens, all of
which produce output signals that are synchronized with each other. It is also being
hand-annotated for many different phenomena, including orthographic transcription,
discourse properties such as named entities and dialogue acts, summaries, emotions,
and some head and hand gestures. We describe the data set, including the rationale
behind using elicited material, and explain how the material is being recorded, transcribed
and annotated.},
booktitle = {Proceedings of the Second International Conference on Machine Learning for Multimodal Interaction},
pages = {28–39},
numpages = {12},
location = {Edinburgh, UK},
series = {MLMI'05}
}
"""

_DESCRIPTION = """\
The AMI Meeting Corpus consists of 100 hours of meeting recordings. The recordings use a range of signals
synchronized to a common timeline. These include close-talking and far-field microphones, individual and
room-view video cameras, and output from a slide projector and an electronic whiteboard. During the meetings,
the participants also have unsynchronized pens available to them that record what is written. The meetings
were recorded in English using three different rooms with different acoustic properties, and include mostly
non-native speakers. \n
"""

_HOMEPAGE = "https://groups.inf.ed.ac.uk/ami/corpus/"

_LICENSE = "CC BY 4.0"

_TRAIN_SAMPLE_IDS = [
    "EN2001a",
    "EN2001b",
    "EN2001d",
    "EN2001e",
    "EN2003a",
    "EN2004a",
    "EN2005a",
    "EN2006a",
    "EN2006b",
    "EN2009b",
    "EN2009c",
    "EN2009d",
    "ES2002a",
    "ES2002b",
    "ES2002c",
    "ES2002d",
    "ES2003a",
    "ES2003b",
    "ES2003c",
    "ES2003d",
    "ES2005a",
    "ES2005b",
    "ES2005c",
    "ES2005d",
    "ES2006a",
    "ES2006b",
    "ES2006c",
    "ES2006d",
    "ES2007a",
    "ES2007b",
    "ES2007c",
    "ES2007d",
    "ES2008a",
    "ES2008b",
    "ES2008c",
    "ES2008d",
    "ES2009a",
    "ES2009b",
    "ES2009c",
    "ES2009d",
    "ES2010a",
    "ES2010b",
    "ES2010c",
    "ES2010d",
    "ES2012a",
    "ES2012b",
    "ES2012c",
    "ES2012d",
    "ES2013a",
    "ES2013b",
    "ES2013c",
    "ES2013d",
    "ES2014a",
    "ES2014b",
    "ES2014c",
    "ES2014d",
    "ES2015a",
    "ES2015b",
    "ES2015c",
    "ES2015d",
    "ES2016a",
    "ES2016b",
    "ES2016c",
    "ES2016d",
    "IB4005",
    "IN1001",
    "IN1002",
    "IN1005",
    "IN1007",
    "IN1008",
    "IN1009",
    "IN1012",
    "IN1013",
    "IN1014",
    "IN1016",
    "IS1000a",
    "IS1000b",
    "IS1000c",
    "IS1000d",
    "IS1001a",
    "IS1001b",
    "IS1001c",
    "IS1001d",
    "IS1002b",
    "IS1002c",
    "IS1002d",
    "IS1003a",
    "IS1003b",
    "IS1003c",
    "IS1003d",
    "IS1004a",
    "IS1004b",
    "IS1004c",
    "IS1004d",
    "IS1005a",
    "IS1005b",
    "IS1005c",
    "IS1006a",
    "IS1006b",
    "IS1006c",
    "IS1006d",
    "IS1007a",
    "IS1007b",
    "IS1007c",
    "IS1007d",
    "TS3005a",
    "TS3005b",
    "TS3005c",
    "TS3005d",
    "TS3006a",
    "TS3006b",
    "TS3006c",
    "TS3006d",
    "TS3007a",
    "TS3007b",
    "TS3007c",
    "TS3007d",
    "TS3008a",
    "TS3008b",
    "TS3008c",
    "TS3008d",
    "TS3009a",
    "TS3009b",
    "TS3009c",
    "TS3009d",
    "TS3010a",
    "TS3010b",
    "TS3010c",
    "TS3010d",
    "TS3011a",
    "TS3011b",
    "TS3011c",
    "TS3011d",
    "TS3012a",
    "TS3012b",
    "TS3012c",
    "TS3012d",
]

_TRAIN_SAMPLES_BROKEN_SDM = ["IS1003b", "IS1007d"]

_VALIDATION_SAMPLE_IDS = [
    "ES2011a",
    "ES2011c",
    "IB4001",
    "IB4003",
    "IB4010",
    "IS1008a",
    "IS1008c",
    "TS3004a",
    "TS3004c",
    "ES2011b",
    "ES2011d",
    "IB4002",
    "IB4004",
    "IB4011",
    "IS1008b",
    "IS1008d",
    "TS3004b",
    "TS3004d",
]

_EVAL_SAMPLE_IDS = [
    "EN2002a",
    "EN2002b",
    "EN2002c",
    "EN2002d",
    "ES2004a",
    "ES2004b",
    "ES2004c",
    "ES2004d",
    "IS1009a",
    "IS1009b",
    "IS1009c",
    "IS1009d",
    "TS3003a",
    "TS3003b",
    "TS3003c",
    "TS3003d",
]

_SAMPLE_IDS = {
    "train": _TRAIN_SAMPLE_IDS,
    "dev": _VALIDATION_SAMPLE_IDS,
    "eval": _EVAL_SAMPLE_IDS,
}

_SUBSETS = ("ihm", "sdm")

_BASE_DATA_URL = "https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/"

_AUDIO_ARCHIVE_URL = _BASE_DATA_URL + "audio/{subset}/{split}/{_id}.tar.gz"

_ANNOTATIONS_ARCHIVE_URL = _BASE_DATA_URL + "annotations/{split}/text"

logger = datasets.utils.logging.get_logger(__name__)


class AMIConfig(datasets.BuilderConfig):
    """BuilderConfig for AMI."""

    def __init__(self, name, *args, **kwargs):
        """BuilderConfig for AMI"""
        super().__init__(name=name, *args, **kwargs)


class AMI(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        AMIConfig(name=subset) for subset in _SUBSETS
    ]

    DEFAULT_WRITER_BATCH_SIZE = 128

    def _info(self):
        features = datasets.Features(
            {
                "meeting_id": datasets.Value("string"),
                "audio_id": datasets.Value("string"),
                "text": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000),
                "begin_time": datasets.Value("float32"),
                "end_time": datasets.Value("float32"),
                "microphone_id": datasets.Value("string"),
                "speaker_id": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        splits = ["train", "dev", "eval"]

        audio_archives_urls = {}
        for split in splits:
            split_ids = _SAMPLE_IDS[split]

            if self.config.name == "sdm" and split == "train":
                split_ids = [s for s in split_ids if s not in _TRAIN_SAMPLES_BROKEN_SDM]

            audio_archives_urls[split] = [
                _AUDIO_ARCHIVE_URL.format(subset=self.config.name, split=split, _id=m) for m in split_ids
            ]

        audio_archives = dl_manager.download(audio_archives_urls)
        local_extracted_archives_paths = dl_manager.extract(audio_archives) if not dl_manager.is_streaming else {
            split: [None] * len(audio_archives[split]) for split in splits
        }

        annotations_urls = {split: _ANNOTATIONS_ARCHIVE_URL.format(split=split) for split in splits}
        annotations = dl_manager.download(annotations_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "audio_archives": [dl_manager.iter_archive(archive) for archive in audio_archives["train"]],
                    "local_extracted_archives_paths": local_extracted_archives_paths["train"],
                    "annotation": annotations["train"],
                    "split": "train"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "audio_archives": [dl_manager.iter_archive(archive) for archive in audio_archives["dev"]],
                    "local_extracted_archives_paths": local_extracted_archives_paths["dev"],
                    "annotation": annotations["dev"],
                    "split": "dev"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "audio_archives": [dl_manager.iter_archive(archive) for archive in audio_archives["eval"]],
                    "local_extracted_archives_paths": local_extracted_archives_paths["eval"],
                    "annotation": annotations["eval"],
                    "split": "eval"
                },
            ),
        ]

    def _generate_examples(self, audio_archives, local_extracted_archives_paths, annotation, split):
        # open annotation file
        assert len(audio_archives) == len(local_extracted_archives_paths)

        with open(annotation, "r", encoding="utf-8") as f:
            transcriptions = {}
            for line in f.readlines():
                line_items = line.strip().split()
                _id = line_items[0]
                text = " ".join(line_items[1:])
                _, meeting_id, microphone_id, speaker_id, begin_time, end_time = _id.split("_")

                # sdm specific
                if self.config.name == "sdm" and meeting_id in _TRAIN_SAMPLES_BROKEN_SDM:
                    continue
                elif self.config.name == "sdm":
                    microphone_id = "SDM1"
                    _id_items = _id.split("_")
                    _id = "_".join(_id_items[:2] + ["sdm"] + _id_items[3:])

                audio_filename = "_".join([split, _id.lower()]) + ".wav"

                transcriptions[audio_filename] = {
                    "audio_id": _id,
                    "meeting_id": meeting_id,
                    "text": text,
                    "begin_time": int(begin_time) / 100,
                    "end_time": int(end_time) / 100,
                    "microphone_id": microphone_id,
                    "speaker_id": speaker_id,
                }

        features = ["meeting_id", "audio_id", "text", "begin_time", "end_time", "microphone_id", "speaker_id"]
        for archive, local_archive_path in zip(audio_archives, local_extracted_archives_paths):
            for audio_path, audio_file in archive:
                # adapt for sdm
                if self.config.name == "sdm":
                    _audio_path_items = audio_path.split("_")
                    audio_path = "_".join(_audio_path_items[:3] + ["sdm"] + _audio_path_items[4:])

                # audio_path is like 'EN2001a/train_ami_en2001a_h00_mee068_0414915_0415078.wav'
                audio_meta = transcriptions[audio_path.split("/")[-1]]

                yield audio_path, {
                    "audio": {
                        "path": os.path.join(local_archive_path, audio_path) if local_archive_path else audio_path,
                        "bytes": audio_file.read(),
                    },
                    **{feature: audio_meta[feature] for feature in features}
                }
