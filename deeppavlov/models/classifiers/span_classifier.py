
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math 
import copy
import json
import six
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
import _pickle as cPickle
from deeppavlov.core.models.torch_model import TorchModel

from deeppavlov.models.torch_bert.torch_transformers_classifier import TorchTransformersClassifierModel
from transformers import AutoConfig, AutoModelForSequenceClassification


from typing import Tuple, List, Optional, Union, Dict, Set, Any
import numpy as np

from transformers import AutoTokenizer, BertTokenizer
from transformers.data.processors.utils import InputFeatures
from transformers.tokenization_utils_base import BatchEncoding

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register, register_model
import torch.nn.functional as F
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor


@register_model("span_classifier")
class SpanClassifier(nn.Module):
    """
        Build span classifier components as a sub-module.
        Classifier that allows for spans and text as input.
        Use same classifier code as build_single_sentence_module,
        except we'll use span indices to extract span representations,
        and use these as input to the classifier.
    """


    def _make_cnn_layer(self, d_inp):
        """
        Make a CNN layer as a projection of local context.
        CNN maps [batch_size, max_len, d_inp]
        to [batch_size, max_len, proj_dim] with no change in length.
        """
        k = 1 + 2 * self.cnn_context
        padding = self.cnn_context
        return nn.Conv1d(
            d_inp,
            self.proj_dim,
            kernel_size=k,
            stride=1,
            padding=padding,
            dilation=1,
            groups=1,
            bias=True,
        )

    def __init__(self, hidden_size, task_params={}, num_spans=2, num_classes=2, dropout=0.2):
        assert num_spans > 0, "Please set num_spans to be more than 0"
        super(SpanClassifier, self).__init__()
        # Set config options needed for forward pass.
        self.loss_type = task_params.get("cls_loss_fn", "softmax")
        self.cnn_context = task_params.get("cnn_context", 0)
        self.num_spans = num_spans
        self.proj_dim = task_params.get("d_hid", hidden_size)
        self.projs = torch.nn.ModuleList()

        for i in range(num_spans):
            # create a word-level pooling layer operator
            proj = self._make_cnn_layer(hidden_size)
            self.projs.append(proj)
        self.span_extractors = torch.nn.ModuleList()

        # Lee's self-pooling operator (https://arxiv.org/abs/1812.10860)
        for i in range(num_spans):
            span_extractor = SelfAttentiveSpanExtractor(self.proj_dim)
            self.span_extractors.append(span_extractor)

        # Classifier gets concatenated projections of spans.
        clf_input_dim = self.span_extractors[1].get_output_dim() * num_spans
        self.classifier =nn.Sequential(
                nn.Linear(clf_input_dim, hidden_size),
                nn.Tanh(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes))

    def forward(
        self,
        batch: Dict
    ) -> Dict:
        """
        Run forward pass.
        Expects batch to have the following entries:
            'input' : [batch_size, max_len, emb_size]
            'labels' : [batch_size, num_targets] of label indices
            'span1s' : [batch_size, 1, 2], span indices
            'span2s' : [batch_size, 1, 2], span indices
                .
                .
                .
            'span_ts': [batch_size, 1, 2], span indices

        Parameters
        -------------------------------
            batch: dict(str -> Tensor) with entries described above.
            sent_embs: [batch_size, max_len, repr_dim] Tensor
            sent_mask: [batch_size, max_len, 1] Tensor of {0,1}
            predict: whether or not to generate predictions
        This learns different span pooling operators for each span.

        Returns
        -------------------------------
            out: dict(str -> Tensor)
        """
        sent_embs = batch['input']
        sent_mask = batch['attention_mask']
        out = {}

        # Apply projection CNN layer for each span of the input sentence
        sent_embs_t = sent_embs.transpose(1, 2)  # needed for CNN layer
        se_projs = []
        for i in range(self.num_spans):
            se_proj = self.projs[i](sent_embs_t).transpose(2, 1).contiguous()
            se_projs.append(se_proj)

        span_embs = torch.Tensor([]).cuda() if torch.cuda.is_available() else torch.Tensor([])
        _kw = dict(sequence_mask=sent_mask.long())
        for i in range(self.num_spans):
            # spans are [batch_size, num_targets, span_modules]
            span_emb = self.span_extractors[i](se_projs[i], batch["span" + str(i + 1) + "s"], **_kw)
            span_embs = torch.cat([span_embs, span_emb], dim=2)

        # [batch_size, num_targets, n_classes]
        logits = self.classifier(span_embs)
        out["logits"] = logits

        # Compute loss if requested.
        if "labels" in batch:
            logits = logits.squeeze(dim=1)
            out["logits"] = logits
            out["loss"] = self.compute_loss(logits, batch["labels"])
        return out

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Paramters
        -------------------------------
            logits: [total_num_targets, n_classes] Tensor of float scores
            labels: [total_num_targets, n_classes] Tensor of sparse binary targets

        Returns
         -------------------------------
            loss: scalar Tensor
        """
        if self.loss_type == "sigmoid":
            return F.binary_cross_entropy(torch.sigmoid(logits), labels.float())
        elif self.loss_type == "softmax":
            return F.cross_entropy(logits, labels.long())
        else:
            raise ValueError(
                "Unsupported loss type '%s' " "for span classification." % self.loss_type
            )
