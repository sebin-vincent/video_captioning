"""Functional interface."""

from typing import Optional, Union

import math
import torch
import torch.nn as nn
from collections.abc import Callable


from .additional_utils.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from .additional_utils.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from .modeling_utils import RobertaConfig
from .additional_utils.masking_utils import create_bidirectional_mask, create_causal_mask
from .additional_utils.pytorch_utils import apply_chunking_to_forward
from .modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from .additional_utils.modeling_layers import GradientCheckpointingLayer
from .additional_utils.processing_utils import Unpack
from .utils import TransformersKwargs, auto_docstring, logging


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


logger = logging.get_logger(__name__)

LayerNormClass = torch.nn.LayerNorm

class RobertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, self.padding_idx)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # NOTE: We assume either pos ids to have bsz == 1 (broadcastable) or bsz == effective bsz (input_shape[0])
                buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
                buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
                token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    @staticmethod
    def create_position_ids_from_inputs_embeds(inputs_embeds, padding_idx):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            padding_idx + 1, sequence_length + padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    @staticmethod
    def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:

        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx



class RoBertaImgEmbeddings(nn.Module):
    """ RoBERTa Language - Image Embedding
    Construct the embeddings from word & Images, position and token_type embeddings.
    """
    def __init__(self, config):
        super(RoBertaImgEmbeddings, self).__init__()
        self.img_dim = 565

        self.img_embeddings = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        img_embeddings = self.img_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = img_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights



class RobertaSelfAttention(nn.Module):
    def __init__(self, config, is_causal=False, layer_idx=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        # get all proj
        query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            # decoder-only roberta can have a simple dynamic cache for example
            current_past_key_values = past_key_values
            if isinstance(past_key_values, EncoderDecoderCache):
                current_past_key_values = past_key_values.self_attention_cache

            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            key_layer, value_layer = current_past_key_values.update(
                key_layer,
                value_layer,
                self.layer_idx,
                {"cache_position": cache_position},
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn_output, attn_weights


class RobertaCrossAttention(nn.Module):
    def __init__(self, config, is_causal=False, layer_idx=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.is_causal = is_causal
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        # determine input shapes
        bsz, tgt_len = hidden_states.shape[:-1]
        src_len = encoder_hidden_states.shape[1]

        q_input_shape = (bsz, tgt_len, -1, self.attention_head_size)
        kv_input_shape = (bsz, src_len, -1, self.attention_head_size)

        # get query proj
        query_layer = self.query(hidden_states).view(*q_input_shape).transpose(1, 2)

        is_updated = past_key_values.is_updated.get(self.layer_idx) if past_key_values is not None else False
        if past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_layer = past_key_values.cross_attention_cache.layers[self.layer_idx].keys
            value_layer = past_key_values.cross_attention_cache.layers[self.layer_idx].values
        else:
            key_layer = self.key(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)
            value_layer = self.value(encoder_hidden_states).view(*kv_input_shape).transpose(1, 2)

            if past_key_values is not None:
                # save all states to the cache
                key_layer, value_layer = past_key_values.cross_attention_cache.update(
                    key_layer, value_layer, self.layer_idx
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                past_key_values.is_updated[self.layer_idx] = True

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        return attn_output, attn_weights



class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaAttention(nn.Module):
    def __init__(self, config, is_causal=False, layer_idx=None, is_cross_attention=False):
        super().__init__()
        self.is_cross_attention = is_cross_attention
        attention_class = RobertaCrossAttention if is_cross_attention else RobertaSelfAttention
        self.self = attention_class(config, is_causal=is_causal, layer_idx=layer_idx)
        self.output = RobertaSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        attention_mask = attention_mask if not self.is_cross_attention else encoder_attention_mask
        attention_output, attn_weights = self.self(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        attention_output = self.output(attention_output, hidden_states)
        return attention_output, attn_weights


class RobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config, is_causal=config.is_decoder, layer_idx=layer_idx)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RobertaAttention(
                config,
                is_causal=False,
                layer_idx=layer_idx,
                is_cross_attention=True,
            )
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        self_attention_output, _ = self.attention(
            hidden_states,
            attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        attention_output = self_attention_output

        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            cross_attention_output, _ = self.crossattention(
                self_attention_output,
                None,  # attention_mask
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            attention_output = cross_attention_output

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class RobertaPreTrainedModel(PreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": RobertaLayer,
        "attentions": RobertaSelfAttention,
        "cross_attentions": RobertaCrossAttention,
    }

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights with BertLMPredictionHead->RobertaLMHead
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, RobertaLMHead):
            module.bias.data.zero_()

class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


@auto_docstring(
    custom_intro="""
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://huggingface.co/papers/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """
)
class RobertaModel(RobertaPreTrainedModel):
    _no_split_modules = ["RobertaEmbeddings", "RobertaLayer"]

    def __init__(self, config, add_pooling_layer=True):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.config = config
        self.gradient_checkpointing = False

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = (
                EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))
                if encoder_hidden_states is not None or self.config.is_encoder_decoder
                else DynamicCache(config=self.config)
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            device = input_ids.device
            seq_length = input_ids.shape[1]
        else:
            device = inputs_embeds.device
            seq_length = inputs_embeds.shape[1]

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past_key_values_length, past_key_values_length + seq_length, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        attention_mask, encoder_attention_mask = self._create_attention_masks(
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            embedding_output=embedding_output,
            encoder_hidden_states=encoder_hidden_states,
            cache_position=cache_position,
            past_key_values=past_key_values,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_ids=position_ids,
            **kwargs,
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
        )

    def _create_attention_masks(
        self,
        attention_mask,
        encoder_attention_mask,
        embedding_output,
        encoder_hidden_states,
        cache_position,
        past_key_values,
    ):
        if self.config.is_decoder:
            attention_mask = create_causal_mask(
                config=self.config,
                input_embeds=embedding_output,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
            )
        else:
            attention_mask = create_bidirectional_mask(
                config=self.config,
                input_embeds=embedding_output,
                attention_mask=attention_mask,
            )

        if encoder_attention_mask is not None:
            encoder_attention_mask = create_bidirectional_mask(
                config=self.config,
                input_embeds=embedding_output,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        return attention_mask, encoder_attention_mask


class RobertaImgModel(RobertaPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        >>> config = BertConfig.from_pretrained('bert-base-uncased')
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>> model = BertModel(config)
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        >>> outputs = model(input_ids)
        >>> last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(RobertaImgModel, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config)

        self.img_dim = config.img_feature_dim #2054 #565
        logger.info('RoBertaImgModel Image Dimension: {}'.format(self.img_dim))
        self.img_feature_type = config.img_feature_type
        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        if config.img_feature_type == 'dis_code':
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_t': # transpose
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_size, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_scale': # scaled
            self.input_embeddings = nn.Linear(config.code_dim, config.code_size, bias=True)
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        else:
            self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            if self.use_img_layernorm:
                self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.img_layer_norm_eps)

        self.apply(self.init_weights)
        self.model_type = getattr(config, 'model_type', 'bert')


        # re-initialize img_embedding weight
        # self.img_embedding.weight.data.normal_(mean=0.0, std=config.img_initializer_range)

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None, img_feats=None,
            encoder_history_states=None):

        if attention_mask is None:
            if img_feats is not None:
                attention_mask = torch.ones((input_ids.shape[0], input_ids.shape[1] + img_feats.shape[1]), device=input_ids.device)
            else:
                attention_mask = torch.ones_like(input_ids)
            #if img_feats is not None: attention_mask = torch.ones_like((input_ids.shape[0], input_ids.shape[1]+img_feats.shape[1]))
            #else: attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                token_type_ids=token_type_ids)
        # add img_embedding_output and sum with embedding_output
        #logger.info('embedding_output: %s' % str(embedding_output.shape))
        if encoder_history_states is not None:
            if encoder_history_states[0].shape[1] != 0:
                assert img_feats is None or img_feats.shape[1]==0, "Cannot take image features while using encoder history states"

        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_t': # transpose
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_scale': # left scaled
                code_emb = self.code_embeddings(img_feats)
                #scale_output =
                # add scale ouput
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'e2e' and self.model_type == 'TIMM_vit':
                img_embedding_output = img_feats
            else:
                if torch._C._get_tracing_state():
                    # Ugly workaround to make this work for ONNX.
                    #  It is also valid for PyTorch bu I keep this path separate to remove once fixed in ONNX
                    img_embedding_output = self.img_embedding(img_feats.squeeze(0)).unsqueeze(0)
                else:
                    img_embedding_output = self.img_embedding(img_feats)
                #logger.info('img_embedding_output: %s' % str(img_embedding_output.shape))
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)

                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)

                # sum two embeddings
                #padding_matrix = torch.zeros((embedding_output.shape[0], embedding_output.shape[1]-img_embedding_output.shape[1], embedding_output.shape[2])).cuda()
                #img_embedding_output = torch.cat((padding_matrix, img_embedding_output), 1)
                #embedding_output = embedding_output + img_embedding_output

            # concatenate two embeddings
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)

        encoder_outputs = self.encoder(embedding_output,
                extended_attention_mask, head_mask=head_mask,
                encoder_history_states=encoder_history_states)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs



class RobertaCaptioningHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RobertaLMHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class RobertaCaptioningLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.label_smoothing = getattr(config, 'label_smoothing', 0)
        self.drop_worst_ratio = getattr(config, 'drop_worst_ratio', 0)
        self.drop_worst_after = getattr(config, 'drop_worst_after', 0)
        self.log_soft = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction='none')
        self.iter = 0

    def forward(self, logits, target):
        self.iter += 1
        eps = self.label_smoothing
        n_class = logits.size(1)
        one_hot = torch.zeros_like(logits).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = self.log_soft(logits)
        loss = self.kl(log_prb, one_hot).sum(1)

        if self.drop_worst_ratio > 0 and self.iter > self.drop_worst_after:
            loss, _ = torch.topk(loss,
                    k=int(loss.shape[0] * (1-self.drop_worst_ratio)),
                    largest=False)

        loss = loss.mean()

        return loss


# cclin
class RobertaImgFeatureLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.label_smoothing = getattr(config, 'label_smoothing', 0)
        self.drop_worst_ratio = getattr(config, 'drop_worst_ratio', 0)
        self.drop_worst_after = getattr(config, 'drop_worst_after', 0)
        self.log_soft = nn.LogSoftmax(dim=1)
        # self.kl = nn.KLDivLoss(reduction='none')
        self.cri = nn.MSELoss()
        # self.cri = nn.SmoothL1Loss()
        self.iter = 0

    def forward(self, logits, target):
        self.iter += 1
        # log_prb = self.log_soft(logits)
        target = target.view(-1, target.shape[-1])
        # loss = self.cri(logits, target).sum(1)
        loss = self.cri(logits, target)

        # if self.drop_worst_ratio > 0 and self.iter > self.drop_worst_after:
        #     loss, _ = torch.topk(loss,
        #             k=int(loss.shape[0] * (1-self.drop_worst_ratio)),
        #             largest=False)

        # loss = loss.mean()

        return loss


class RobertaForImageCaptioning(RobertaPreTrainedModel):
    r"""
    Bert for Image Captioning.
    """
    def __init__(self, config):
        super(RobertaForImageCaptioning, self).__init__(config)
        self.config = config
        self.bert = RobertaImgModel(config)
        self.cls = RobertaCaptioningHeads(config)
        self.loss = RobertaCaptioningLoss(config)
        # cclin
        self.cls_img_feat = RobertaCaptioningHeads(config)
        self.loss_img_feat = RobertaImgFeatureLoss(config)

        self.apply(self.init_weights)
        self.tie_weights()

        self.model_type = getattr(config, 'model_type', 'bert')
        if self.model_type == 'TIMM_vit':
            self.bert = RobertaImgModel(config)

    def tie_weights(self):
        if hasattr(self.config, 'tie_weights') and self.config.tie_weights:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)
        freeze = False
        if hasattr(self.config, 'freeze_embedding'):
            freeze = self.config.freeze_embedding
        self.bert.embeddings.word_embeddings.weight.requires_grad = not freeze

    def forward(self, *args, **kwargs):
        is_decode = kwargs.get('is_decode', False)
        inference_mode = kwargs.get('inference_mode', '')
        if inference_mode:
            kwargs.pop('inference_mode')
            if inference_mode == 'prod':
                return self.prod_generate(*args, **kwargs)
            if inference_mode == 'prod_no_hidden':
                return self.prod_no_hidden_generate(*args, **kwargs)
            assert False, 'unknown inference_mode: {}'.format(inference_mode)
        if is_decode:
            return self.generate(*args, **kwargs)
        else:
            return self.encode_forward(*args, **kwargs)

    def encode_forward(self, input_ids, img_feats, attention_mask,
            masked_pos=None, masked_ids=None,
            masked_pos_img=None, masked_token_img=None,
            token_type_ids=None, position_ids=None, head_mask=None,
            is_training=True, encoder_history_states=None):
        outputs = self.bert(input_ids, img_feats=img_feats, attention_mask=attention_mask,
                position_ids=position_ids, token_type_ids=token_type_ids,
                head_mask=head_mask,
                encoder_history_states=encoder_history_states)
        if is_training:
            sequence_output = outputs[0][:, :masked_pos.shape[-1], :]
            # num_masks_in_batch * hidden_size
            # a_start_time = time.time()
            sequence_output_masked = sequence_output[masked_pos==1, :] # it is slow, but don't have better solution now
            # print('bert forward (after sequence_output_masked original):', time.time() - a_start_time)
            # a_start_time = time.time()
            # sequence_output = sequence_output.cpu()
            # print('bert forward (after sequence_output_masked 1):', time.time() - a_start_time)
            # a_start_time = time.time()
            # sequence_output_masked = sequence_output[index,:]
            # print('bert forward (after sequence_output_masked 2):', time.time() - a_start_time)
            # a_start_time = time.time()
            # sequence_output_masked = sequence_output_masked.cuda()
            # # sequence_output_masked = sequence_output.cpu()[index,:].cuda()
            # print('bert forward (after sequence_output_masked 3):', time.time() - a_start_time)
            # index = masked_pos.bool()
            # a_start_time = time.time()
            # new_index = index.unsqueeze(-1).expand(-1, -1, sequence_output.shape[-1])
            # sequence_output_masked = torch.masked_select(sequence_output, new_index).view(-1,sequence_output.shape[-1])
            # print('bert forward (after sequence_output_masked v2):', time.time() - a_start_time)
            class_logits = self.cls(sequence_output_masked)
            masked_ids = masked_ids[masked_ids != -1]   # remove padded target
            masked_loss = self.loss(class_logits.float(), masked_ids)
            if masked_pos_img is not None:
                sequence_output_img = outputs[0][:, masked_pos.shape[-1]:, :]
                # num_masks_in_batch * hidden_size
                sequence_output_masked_img = sequence_output_img[masked_pos_img==1, :]
                # img cclin
                img_feat = self.cls_img_feat(sequence_output_masked_img)
                masked_loss_img = self.loss_img_feat(img_feat.float(), masked_token_img)
                masked_loss += 0.1 * masked_loss_img  # TODO
            outputs = (masked_loss, class_logits,) + outputs[2:]
        else:
            sequence_output = outputs[0][:, :input_ids.shape[-1], :]
            class_logits = self.cls(sequence_output)
            outputs = (class_logits,) + outputs[2:]
        return outputs

    def prepare_inputs_for_generation(self, curr_ids, past=None):
        # NOTE: if attention is on, it should be the token used to mask words in training
        mask_token_id = self.mask_token_id
        batch_size = curr_ids.shape[0]
        mask_ids = torch.full(
            (batch_size, 1), mask_token_id, dtype=torch.long, device=curr_ids.device
        )

        def _slice(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len + self.od_labels_len)
            return t[:, start: end]

        def _remove_elements(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len + self.od_labels_len)
            return torch.cat([t[:, :start], t[:, end:]], dim=1)

        if past is None:
            input_ids = torch.cat([curr_ids, mask_ids], dim=1)

            curr_len = input_ids.shape[1]
            full_len = self.max_seq_len + self.od_labels_len + self.img_seq_len
            assert self.full_attention_mask.shape == (batch_size,
                    full_len, full_len)

            def _remove_rows_cols(t, row_start, row_end, col_start, col_end):
                t00 = t[:, :row_start, :col_start]
                t01 = t[:, :row_start, col_end:]
                t10 = t[:, row_end:, :col_start]
                t11 = t[:, row_end:, col_end:]
                res = torch.cat([torch.cat([t00, t01], dim=2), torch.cat([t10, t11],
                            dim=2)], dim=1)
                assert res.shape == (t.shape[0], t.shape[1]-row_end+row_start,
                        t.shape[2]-col_end+col_start)
                return res

            seq_start = curr_len
            seq_end = self.max_seq_len
            attention_mask = _remove_rows_cols(self.full_attention_mask, seq_start,
                    seq_end, seq_start, seq_end)

            masked_pos = _remove_elements(self.full_masked_pos, seq_start, seq_end)
            token_type_ids = _remove_elements(self.full_token_type_ids, seq_start, seq_end)
            position_ids = _remove_elements(self.full_position_ids, seq_start, seq_end)
            img_feats = self.img_feats

            if self.add_od_labels:
                assert self.od_label_ids.shape[1] == self.od_labels_len
                input_ids = torch.cat([input_ids, self.od_label_ids], dim=1)
        else:
            last_token = curr_ids[:, -1:]
            # The representation of last token should be re-computed, because
            # it depends on both self-attention context and input tensor
            input_ids = torch.cat([last_token, mask_ids], dim=1)
            start_pos = curr_ids.shape[1] - 1
            end_pos = start_pos + input_ids.shape[1]
            masked_pos = _slice(self.full_masked_pos, start_pos, end_pos)
            token_type_ids = _slice(self.full_token_type_ids, start_pos, end_pos)
            position_ids = _slice(self.full_position_ids, start_pos, end_pos)

            img_feats = None
            assert past[0].shape[0] == batch_size
            if self.prev_encoded_layers is None:
                assert start_pos == 1  # the first token after BOS
                assert past[0].shape[1] == 2 + self.od_labels_len + self.img_seq_len
                # reorder to [od_labels, img_feats, sentence]
                self.prev_encoded_layers = [
                        torch.cat([x[:, 2:, :], x[:, :start_pos,:]], dim=1)
                        for x in past]
                s2s = self.full_attention_mask[:, :self.max_seq_len,
                        :self.max_seq_len]
                s2i = self.full_attention_mask[:, :self.max_seq_len,
                        self.max_seq_len:]
                i2s = self.full_attention_mask[:, self.max_seq_len:,
                        :self.max_seq_len]
                i2i = self.full_attention_mask[:, self.max_seq_len:,
                        self.max_seq_len:]
                self.full_attention_mask = torch.cat(
                        [torch.cat([i2i, i2s], dim=2),
                        torch.cat([s2i, s2s], dim=2)],
                        dim=1)
            else:
                assert start_pos > 1
                assert past[0].shape[1] == 2
                self.prev_encoded_layers = [torch.cat([x, p[:, :-1, :]], dim=1)
                        for x, p in zip(self.prev_encoded_layers, past)]

            attention_mask = self.full_attention_mask[:,
                self.od_labels_len+self.img_seq_len+start_pos: self.od_labels_len+self.img_seq_len+end_pos,
                :self.od_labels_len+self.img_seq_len+end_pos]

        return {'input_ids': input_ids, 'img_feats': img_feats,
            'masked_pos': masked_pos, 'attention_mask': attention_mask,
            'token_type_ids': token_type_ids, 'position_ids': position_ids,
            'is_training': False,
            'encoder_history_states': self.prev_encoded_layers}

    def get_output_embeddings(self):
        return self.decoder

    def generate(self, img_feats, attention_mask, masked_pos, token_type_ids=None,
            position_ids=None, head_mask=None, input_ids=None, max_length=None,
            do_sample=None, num_beams=None, temperature=None, top_k=None, top_p=None,
            repetition_penalty=None, bos_token_id=None, pad_token_id=None,
            eos_token_ids=None, mask_token_id=None, length_penalty=None,
            num_return_sequences=None,
            num_keep_best=1, is_decode=None,
            add_od_labels=False, od_labels_start_posid=None,
            use_cbs=False, fsm=None, num_constraints=None,
            min_constraints_to_satisfy=None, use_hypo=False,
            decoding_constraint_flag=None, bad_ending_ids=None,
            ):
        """ Generates captions given image features
        """
        assert is_decode
        batch_size = img_feats.shape[0]
        self.img_seq_len = img_feats.shape[1]
        self.max_seq_len = max_length
        self.mask_token_id = mask_token_id
        self.prev_encoded_layers = None
        # NOTE: num_keep_best is not equavilant to num_return_sequences
        # num_keep_best is the number of hypotheses to keep in beam search
        # num_return_sequences is the repeating times of input, coupled with
        # do_sample=True can generate more than one samples per image
        self.num_keep_best = num_keep_best

        vocab_size = self.config.vocab_size
        if not use_cbs:
            num_fsm_states = 1
        else:
            b, num_fsm_states, f1, v = fsm.shape
            assert b==batch_size and v==vocab_size and f1==num_fsm_states

        self.add_od_labels = add_od_labels
        # avoid position_ids collision of caption and od labels
        self.od_labels_start_posid = max(od_labels_start_posid, self.max_seq_len)
        if self.add_od_labels:
            # get od labels part from input_ids
            assert input_ids.shape[0] == batch_size
            od_label_ids = input_ids[:, self.max_seq_len:]
            self.od_labels_len = input_ids.shape[1] - self.max_seq_len
            input_ids = None
        else:
            self.od_labels_len = 0
            od_label_ids = None
            assert input_ids.shape == (batch_size, self.max_seq_len)
            input_ids = None

        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."
            assert input_ids.shape[0] == batch_size, "Input batch size must match image features"

        cur_len = input_ids.shape[1]
        if  num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = self._expand_for_beams(input_ids, num_return_sequences)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        if position_ids is None:
            position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=input_ids.device)
            posids_len = self.max_seq_len
            if self.add_od_labels:
                od_labels_posids = torch.arange(
                        self.od_labels_start_posid,
                        self.od_labels_start_posid + self.od_labels_len, dtype=torch.long, device=input_ids.device)
                position_ids = torch.cat([position_ids, od_labels_posids])
                posids_len += self.od_labels_len
            position_ids = position_ids.unsqueeze(0).expand([batch_size, posids_len])

        num_expand = num_beams * num_fsm_states * num_return_sequences
        self.od_label_ids = self._expand_for_beams(od_label_ids, num_expand)
        self.img_feats = self._expand_for_beams(img_feats, num_expand)
        self.full_attention_mask = self._expand_for_beams(attention_mask, num_expand)
        self.full_masked_pos = self._expand_for_beams(masked_pos, num_expand)
        self.full_token_type_ids = self._expand_for_beams(token_type_ids, num_expand)
        self.full_position_ids = self._expand_for_beams(position_ids, num_expand)
        self.full_head_mask = self._expand_for_beams(head_mask, num_expand)

        if not use_cbs:
            if num_beams > 1:
                output = self._generate_beam_search(
                    input_ids,
                    cur_len,
                    max_length,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    pad_token_id,
                    eos_token_ids,
                    effective_batch_size,
                    length_penalty,
                    num_beams,
                    vocab_size,
                )
            else:
                output = self._generate_no_beam_search(
                    input_ids,
                    cur_len,
                    max_length,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    pad_token_id,
                    eos_token_ids,
                    effective_batch_size,
                )
        else:
            from src.modeling.utils_cbs import (ConstrainedBeamSearch,
                    select_best_beam_with_constraints)
            assert self.num_keep_best == 1, 'not supported n_best > 1 for CBS'
            searcher = ConstrainedBeamSearch(eos_token_ids, max_length,
                    num_beams, use_hypo=use_hypo,
                    decoding_constraint_flag=decoding_constraint_flag,
                    bad_ending_ids=bad_ending_ids)
    
            curr_ids, sum_logprobs = searcher.search(
                    input_ids,
                    None,
                    self._decode_step,
                    fsm,
            )

            curr_ids, logprobs = select_best_beam_with_constraints(
                curr_ids,
                sum_logprobs,
                num_constraints,
                min_constraints_to_satisfy,
                eos_token_ids,
            )
            # (batch_size, n_best, max_len), (batch_size, n_best)
            output = (curr_ids.unsqueeze(1), logprobs.unsqueeze(1))

        return output

    def _expand_for_beams(self, x, num_expand):
        if x is None or num_expand == 1:
            return x

        input_shape = list(x.shape)
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_expand, ...)
        x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
        return x

    def _do_output_past(self, outputs):
        return len(outputs) > 1

    def prod_generate(self, img_feats, od_label_ids, max_length,
            bos_token_id, eos_token_ids, mask_token_id, od_labels_start_posid,
            add_od_labels=True, cls_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1,
            ):
        """ Generates captions for PROD, batch size=1, num_beams=1.
            Use faster generation where output_hidden_states = True
        """
        batch_size = img_feats.shape[0]
        assert batch_size == 1
        device = img_feats.device
        assert od_label_ids.shape[0] == batch_size
        od_labels_len = od_label_ids.shape[1]
        img_seq_len = img_feats.shape[1]

        mask_ids = torch.full(
            (1, 1), mask_token_id, dtype=torch.long, device=device
        )

        # prepare inputs
        cur_ids = torch.full((1, 1), bos_token_id,
                dtype=torch.long, device=device)

        input_ids = torch.cat([cur_ids, mask_ids, od_label_ids], dim=1)
        token_type_ids = torch.cat([
                torch.tensor([[cls_token_segment_id, sequence_a_segment_id]],
                    dtype=torch.long, device=device),
                torch.full((1, od_labels_len), sequence_b_segment_id,
                    dtype=torch.long, device=device)
                ], dim=1)

        position_ids = torch.arange(2, dtype=torch.long, device=device)
        od_labels_start_posid = max(od_labels_start_posid, max_length)
        if add_od_labels:
            od_labels_posids = torch.arange(
                    od_labels_start_posid, od_labels_start_posid + od_labels_len,
                    dtype=torch.long, device=device)
            position_ids = torch.cat([position_ids, od_labels_posids])
        posids_len = 2 + od_labels_len
        position_ids = position_ids.unsqueeze(0).expand([1, posids_len])

        attention_mask = torch.ones(
                (1, 2+od_labels_len+img_seq_len, 2+od_labels_len+img_seq_len),
                dtype=torch.long, device=device)
        attention_mask[:, 0, 1] = 0   # words in sentence can not see words after itself
        attention_mask[:, 2:, :2] = 0 # od_label, img_feat can not see sentence

        # make empty history states for the first step
        encoder_history_states = tuple(
            torch.empty([1, 0, self.config.hidden_size], device=device)
            for _ in range(self.config.num_hidden_layers)
        )

        # prepare inputs for >1 steps
        token_type_ids_after_first = torch.full([1, 2], sequence_a_segment_id,
                dtype=torch.long, device=device)
        img_feats_after_first = torch.empty([1, 0, self.config.img_feature_dim],
                device=device)  # place holder to avoid None

        # initial model inputs for the first step
        model_inputs = {'input_ids': input_ids, 'img_feats': img_feats,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids, 'position_ids': position_ids,
            'is_training': False,
            'encoder_history_states': encoder_history_states}
        cur_len = cur_ids.shape[1]
        sum_logprob = 0
        while True:
            outputs = self(**model_inputs)

            assert self._do_output_past(outputs)
            if cur_len == 1:
                assert outputs[0].shape[1] == 2 + od_labels_len
            else:
                assert cur_len > 1
                assert outputs[0].shape[1] == 2

            # greedy decoding
            next_token_idx = 1
            next_token_logits = outputs[0][:, next_token_idx, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            # Compute scores
            _scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
            sum_logprob += _scores[:, next_token].item()

            if next_token in eos_token_ids:
                break
            cur_ids = torch.cat([cur_ids, next_token.unsqueeze(-1)], dim=-1)
            cur_len = cur_ids.shape[1]
            if cur_len == max_length:
                break

            # prepare model inputs for the next step
            past = outputs[1]
            last_token = cur_ids[:, -1:]
            input_ids = torch.cat([last_token, mask_ids], dim=1)
            position_ids = torch.arange(cur_len - 1, cur_len + 1,
                    dtype=torch.long, device=device)
            attention_mask = torch.ones([1, 2, od_labels_len+img_seq_len+cur_len+1],
                    dtype=torch.long, device=device)
            attention_mask[:, 0, -1] = 0
            assert past[0].shape[0] == batch_size
            # special handle for the first step
            if cur_len == 2:
                assert past[0].shape[1] == 2 + od_labels_len + img_seq_len
                # remove the first token after BOS
                # reorder to [od_labels, img_feats, sentence]
                encoder_history_states = [
                        torch.cat([x[:, 2:, :], x[:, :1,:]], dim=1)
                        for x in past]
            else:
                assert cur_len > 2
                assert past[0].shape[1] == 2
                encoder_history_states = [torch.cat([x, p[:, :-1, :]], dim=1)
                        for x, p in zip(encoder_history_states, past)]

            model_inputs = {'input_ids': input_ids,
                'img_feats': img_feats_after_first,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids_after_first,
                'position_ids': position_ids,
                'is_training': False,
                'encoder_history_states': encoder_history_states}

        logprob = sum_logprob / cur_ids.shape[1]

        # (batch_size, max_len), (batch_size)
        return cur_ids, torch.full((1,), logprob, device=device)

    def prod_no_hidden_generate(self, img_feats, od_label_ids, max_length,
            bos_token_id, eos_token_ids, mask_token_id, od_labels_start_posid,
            add_od_labels=True, cls_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1,
            ):
        """ Generates captions for PROD, batch size=1, num_beams=1.
            Use output_hidden_states = False
        """
        batch_size = img_feats.shape[0]
        assert batch_size == 1
        device = img_feats.device
        assert od_label_ids.shape[0] == batch_size
        od_labels_len = od_label_ids.shape[1]
        img_seq_len = img_feats.shape[1]

        mask_ids = torch.full(
            (1, 1), mask_token_id, dtype=torch.long, device=device
        )

        # prepare inputs
        cur_ids = torch.full((1, 1), bos_token_id,
                dtype=torch.long, device=device)
        od_labels_start_posid = max(od_labels_start_posid, max_length)
        triangle_mask = torch.tril(torch.ones([max_length, max_length],
                dtype=torch.long, device=device))

        def _prepare_inputs(cur_ids):
            cur_len = cur_ids.shape[1]
            input_ids = torch.cat([cur_ids, mask_ids, od_label_ids], dim=1)
            token_type_ids = torch.cat([
                    torch.tensor([[cls_token_segment_id]],
                        dtype=torch.long, device=device),
                    torch.full((1, cur_len), sequence_a_segment_id,
                        dtype=torch.long, device=device),
                    torch.full((1, od_labels_len), sequence_b_segment_id,
                        dtype=torch.long, device=device)
                    ], dim=1)

            token_len = cur_len + 1
            position_ids = torch.arange(token_len, dtype=torch.long, device=device)
            if add_od_labels:
                od_labels_posids = torch.arange(
                        od_labels_start_posid, od_labels_start_posid + od_labels_len,
                        dtype=torch.long, device=device)
                position_ids = torch.cat([position_ids, od_labels_posids])
            posids_len = token_len + od_labels_len
            position_ids = position_ids.unsqueeze(0).expand([1, posids_len])

            attention_mask = torch.ones(
                    (1, token_len+od_labels_len+img_seq_len, token_len+od_labels_len+img_seq_len),
                    dtype=torch.long, device=device)
            attention_mask[:, :token_len,
                    :token_len].copy_(triangle_mask[:token_len, :token_len])
            attention_mask[:, token_len:, :token_len] = 0 # od_label, img_feat can not see sentence
            return input_ids, token_type_ids, position_ids, attention_mask

        # initial model inputs for the first step
        input_ids, token_type_ids, position_ids, attention_mask = \
                _prepare_inputs(cur_ids)
        model_inputs = {'input_ids': input_ids, 'img_feats': img_feats,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids, 'position_ids': position_ids,
            'is_training': False,
            }
        cur_len = cur_ids.shape[1]
        sum_logprob = 0
        while True:
            outputs = self(**model_inputs)

            assert not self._do_output_past(outputs)

            # greedy decoding
            next_token_idx = cur_len
            next_token_logits = outputs[0][:, next_token_idx, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            # Compute scores
            _scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
            sum_logprob += _scores[:, next_token].item()

            if next_token in eos_token_ids:
                break
            cur_ids = torch.cat([cur_ids, next_token.unsqueeze(-1)], dim=-1)
            cur_len = cur_ids.shape[1]
            if cur_len == max_length:
                break

            # prepare model inputs for the next step
            input_ids, token_type_ids, position_ids, attention_mask = \
                    _prepare_inputs(cur_ids)
            model_inputs = {'input_ids': input_ids,
                'img_feats': img_feats,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'position_ids': position_ids,
                'is_training': False,
                }

        logprob = sum_logprob / cur_ids.shape[1]

        # (batch_size, max_len), (batch_size)
        return cur_ids, torch.full((1,), logprob, device=device)