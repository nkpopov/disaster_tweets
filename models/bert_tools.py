#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K
from typing import Any
from typing import Tuple
from typing import List


# Path to BERT module
BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


def _multiply_by_mask(x, mask):
    return x * tf.expand_dims(mask, axis=-1)


def _masked_reduce_mean(x, mask):
    reduced_x = tf.reduce_sum(_multiply_by_mask(x, mask), axis=1)
    reduced_m = tf.reduce_sum(mask, axis=1, keepdims=True)
    return reduced_x / (reduced_m + 1e-10)


def _layer_name(n: int) -> str:
    return "encoder/layer_" + str(11 - n)


class BertLayer(tf.keras.layers.Layer):
    def __init__(self, n_fine_tune_layers: int=3, pooling: str='first', **kwargs):
        if pooling not in ["first", "mean"]:
            raise ValueError("Undefined pooling type: " + self.pooling)

        self.pooling = pooling
        self.trainable = True
        self.output_size = 768
        self.n_fine_tune_layers = n_fine_tune_layers
        self.fine_tune_layers = [_layer_name(i) for i in range(n_fine_tune_layers)]
        super(BertLayer, self).__init__(**kwargs)

    def _filter_trainable_variables(self, variables, layers):
        variables = [v for v in variables if any([l in v.name for l in layers])]
        return variables

    def _get_trainable_vars_first(self):
        predicate = lambda x: not "/cls/" in x.name
        variables = [v for v in self.bert.variables if predicate(v)]
        layers = self.fine_tune_layers + ["pooler/dense"]
        return self._filter_trainable_variables(variables, layers)

    def _get_trainable_vars_mean(self):
        predicate = lambda x: not "/cls/" in x.name and not "/pooler/" in x.name
        variables = [v for v in self.bert.variables if predicate(v)]
        layers = self.fine_tune_layers
        return self._filter_trainable_variables(variables, layers)

    def _get_pooled_first(self, bert_inputs):
        pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)
        pooled = pooled["pooled_output"]
        return pooled

    def _get_pooled_mean(self, bert_inputs):
        mask = tf.cast(bert_inputs["input_mask"], tf.float32)        
        pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)
        pooled = pooled["sequence_output"]
        pooled = _masked_reduce_mean(pooled, mask)
        return pooled

    GET_TRAINABLE_VARS_FUNC = {
        "first": _get_trainable_vars_first,
        "mean":  _get_trainable_vars_mean
    }
        
    GET_POOLED_FUNC = {
        "first": _get_pooled_first,
        "mean": _get_pooled_mean
    }

    def build(self, input_shape: Tuple[int, int]) -> None:
        self.bert = hub.Module(BERT_PATH, trainable=self.trainable, name=self.name + "_module")
        variables = self.bert.variables
        variables_tr = self.GET_TRAINABLE_VARS_FUNC[self.pooling](self)
        variables_not_tr = [v for v in variables if v not in variables_tr]

        print("-------------------------")
        print("Bert trainable variables:")
        for v in variables_tr:
            print(v.name)
        print("-------------------------")

        self._trainable_weights.extend(variables_tr)
        self._non_trainable_weights.extend(variables_not_tr)
        super(BertLayer, self).build(input_shape)

    def call(self, inputs: Tuple) -> Any:
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        bert_inputs = dict(input_ids=inputs[0], input_mask=inputs[1], segment_ids=inputs[2])
        return self.GET_POOLED_FUNC[self.pooling](self, bert_inputs)

    def compute_output_shape(self, input_shape) -> Tuple[int, int]:
        return (input_shape[0], self.output_size)
                                                                
    
if __name__ == "__main__":
    pass

