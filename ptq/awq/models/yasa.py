from ptq.awq.models.base import BaseAWQForCausalLM
from ptq.yasa.yasa_model import *

class YasaAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "GPTNeoXDecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def get_model_layers(model: YasaCausalLM):
        return model.gpt_neox.layers

    @staticmethod
    def get_act_for_scaling(module: YasaLayer):
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.dense_h_to_4h.out_features,
        )

    @staticmethod
    def move_embed(model: YasaCausalLM, device: str):
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)

    @staticmethod
    def get_layers_for_scaling(module: YasaLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[module.attention.query_key_value],
                inp=input_feat["attention.query_key_value"],
            )
        )

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/issues/2#issuecomment-1606297469
        """
        layers.append(dict(
            prev_op=module.attention.query_key_value,
            layers=[module.attention.dense],
            inp=input_feat['attention.dense'],
        ))
        """

        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat["mlp.dense_h_to_4h"],
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.act,
                layers=[module.mlp.dense_4h_to_h],
                inp=input_feat["mlp.dense_4h_to_h"],
                is_swiglu=True
            )
        )

        return layers
