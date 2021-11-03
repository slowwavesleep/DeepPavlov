# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

from abc import abstractmethod
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Optional

import torch
from overrides import overrides

from deeppavlov.core.common.errors import ConfigError

from deeppavlov.core.models.torch_model import TorchModel

logger = getLogger(__name__)


class TransformerModel(TorchModel):

    @abstractmethod
    def train_on_batch(self, x: list, y: list):
        ...


    @abstractmethod
    def _load_hf_model(self, model: str, with_weights: bool):
        ...


    @overrides
    def load(self, fname):
        if fname is not None:
            self.load_path = fname

        if self.pretrained_bert:
            ...

