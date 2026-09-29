# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
GigaSpeech is an evolving, multi-domain English speech recognition corpus with 10,000 hours of high quality
labeled audio suitable for supervised training, and 40,000 hours of total audio suitable for semi-supervised
and unsupervised training. Around 40,000 hours of transcribed audio is first collected from audiobooks, podcasts
and YouTube, covering both read and spontaneous speaking styles, and a variety of topics, such as arts, science,
sports, etc. A new forced alignment and segmentation pipeline is proposed to create sentence segments suitable
for speech recognition training, and to filter out segments with low-quality transcription. For system training,
GigaSpeech provides five subsets of different sizes, 10h, 250h, 1000h, 2500h, and 10000h.
For our 10,000-hour XL training subset, we cap the word error rate at 4% during the filtering/validation stage,
and for all our other smaller training subsets, we cap it at 0%. The DEV and TEST evaluation sets, on the other hand,
are re-processed by professional human transcribers to ensure high transcription quality.
"""

import csv
import os

import datasets

_CITATION = """\
@article{DBLP:journals/corr/abs-2106-06909,
  author    = {Guoguo Chen and
               Shuzhou Chai and
               Guanbo Wang and
               Jiayu Du and
               Wei{-}Qiang Zhang and
               Chao Weng and
               Dan Su and
               Daniel Povey and
               Jan Trmal and
               Junbo Zhang and
               Mingjie Jin and
               Sanjeev Khudanpur and
               Shinji Watanabe and
               Shuaijiang Zhao and
               Wei Zou and
               Xiangang Li and
               Xuchen Yao and
               Yongqing Wang and
               Yujun Wang and
               Zhao You and
               Zhiyong Yan},
  title     = {GigaSpeech: An Evolving, Multi-domain {ASR} Corpus with 10, 000 Hours
               of Transcribed Audio},
  journal   = {CoRR},
  volume    = {abs/2106.06909},
  year      = {2021},
  url       = {https://arxiv.org/abs/2106.06909},
  eprinttype = {arXiv},
  eprint    = {2106.06909},
  timestamp = {Wed, 29 Dec 2021 14:29:26 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2106-06909.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """\
GigaSpeech is an evolving, multi-domain English speech recognition corpus with 10,000 hours of high quality
labeled audio suitable for supervised training, and 40,000 hours of total audio suitable for semi-supervised
and unsupervised training. Around 40,000 hours of transcribed audio is first collected from audiobooks, podcasts
and YouTube, covering both read and spontaneous speaking styles, and a variety of topics, such as arts, science,
sports, etc. A new forced alignment and segmentation pipeline is proposed to create sentence segments suitable
for speech recognition training, and to filter out segments with low-quality transcription. For system training,
GigaSpeech provides five subsets of different sizes, 10h, 250h, 1000h, 2500h, and 10000h.
For our 10,000-hour XL training subset, we cap the word error rate at 4% during the filtering/validation stage,
and for all our other smaller training subsets, we cap it at 0%. The DEV and TEST evaluation sets, on the other hand,
are re-processed by professional human transcribers to ensure high transcription quality.
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

_SUBSETS = ("ihm",)

_BASE_DATA_URL = "https://huggingface.co/datasets/patrickvonplaten/ami-ihm-kaldi-chunked/resolve/main/"

_AUDIO_ARCHIVE_URL = _BASE_DATA_URL + "audio/{subset}/{split}/{_id}.tar.gz"

_ANNOTATIONS_ARCHIVE_URL = _BASE_DATA_URL + "annotations/{split}/text"

logger = datasets.utils.logging.get_logger(__name__)


class AMIConfig(datasets.BuilderConfig):
    """BuilderConfig for AMI."""

    def __init__(self, name, *args, **kwargs):
        """BuilderConfig for AMI"""
        super().__init__(name=name, *args, **kwargs)


class AMI(datasets.GeneratorBasedBuilder):
    """
    GigaSpeech is an evolving, multi-domain English speech recognition corpus with 10,000 hours of high quality
    labeled audio suitable for supervised training, and 40,000 hours of total audio suitable for semi-supervised
    and unsupervised training (this implementation contains only labelled data for now).
    Around 40,000 hours of transcribed audio is first collected from audiobooks, podcasts
    and YouTube, covering both read and spontaneous speaking styles, and a variety of topics, such as arts, science,
    sports, etc. A new forced alignment and segmentation pipeline is proposed to create sentence segments suitable
    for speech recognition training, and to filter out segments with low-quality transcription. For system training,
    GigaSpeech provides five subsets of different sizes, 10h, 250h, 1000h, 2500h, and 10000h.
    For our 10,000-hour XL training subset, we cap the word error rate at 4% during the filtering/validation stage,
    and for all our other smaller training subsets, we cap it at 0%. The DEV and TEST evaluation sets, on the other hand,
    are re-processed by professional human transcribers to ensure high transcription quality.
    """

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        AMIConfig(name=subset) for subset in _SUBSETS
    ]

    DEFAULT_WRITER_BATCH_SIZE = 128

    def _info(self):
        features = datasets.Features(
            {
                "segment_id": datasets.Value("string"),
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
        train_audio_files = {m: _AUDIO_ARCHIVE_URL.format(subset=self.config.name, split="train", _id=m) for m in _TRAIN_SAMPLE_IDS}
        dev_audio_files = {m: _AUDIO_ARCHIVE_URL.format(subset=self.config.name, split="dev", _id=m) for m in _VALIDATION_SAMPLE_IDS}
        eval_audio_files = {m: _AUDIO_ARCHIVE_URL.format(subset=self.config.name, split="eval", _id=m) for m in _EVAL_SAMPLE_IDS}

        train_audio_archives = dl_manager.download_and_extract(train_audio_files)
        dev_audio_archives = dl_manager.download_and_extract(dev_audio_files)
        eval_audio_archives = dl_manager.download_and_extract(eval_audio_files)

        train_annotation = dl_manager.download_and_extract(_ANNOTATIONS_ARCHIVE_URL.format(split="train"))
        dev_annotation = dl_manager.download_and_extract(_ANNOTATIONS_ARCHIVE_URL.format(split="dev"))
        eval_annotation = dl_manager.download_and_extract(_ANNOTATIONS_ARCHIVE_URL.format(split="eval"))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"audio": train_audio_archives, "annotation": train_annotation, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"audio": dev_audio_archives, "annotation": dev_annotation, "split": "dev"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"audio": eval_audio_archives, "annotation": eval_annotation, "split": "eval"},
            ),
        ]

    def _generate_examples(self, audio, annotation, split):
        # open annotation file
        with open(annotation, "r", encoding="utf-8") as f:
            transcriptions = {}
            for line in f.readlines():
                line_items = line.strip().split()
                _id = line_items[0]
                text = " ".join(line_items[1:])
                _, segment_id, microphone_id, speaker_id, begin_time, end_time = _id.split("_")

                transcriptions[_id] = {
                    "audio_id": _id,
                    "segment_id": segment_id,
                    "text": text,
                    "begin_time": int(begin_time) / 100,
                    "end_time": int(end_time) / 100,
                    "microphone_id": microphone_id,
                    "speaker_id": speaker_id,
                }

        for _audio_id, (transcription_id, result) in enumerate(transcriptions.items()):
            folder_id = result["segment_id"]
            file_name = "_".join([split, transcription_id.lower()]) + ".wav"
            audio_file = os.path.join(audio[folder_id], folder_id, file_name)
            result["audio"] = audio_file
            yield _audio_id, result
