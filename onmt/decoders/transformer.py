"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

from onmt.decoders.decoder import DecoderBase
from onmt.modules import MultiHeadedAttention, AverageAttention, JointMultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.gpt_mlp import MLP

## Unconditional version
class TransformerGPTUnconditionalDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, attn_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0):
        super(TransformerGPTUnconditionalDecoderLayer, self).__init__()

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=attn_dropout,
                max_relative_positions=max_relative_positions)
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(d_model, dropout=attn_dropout)

        self.feed_forward = MLP(d_model, d_model*4, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-5)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                layer_cache=None, step=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attn ``(batch_size, 1, src_len)``

        """
        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8)
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            future_mask = future_mask.bool()
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        input_norm = self.layer_norm_1(inputs)

        if isinstance(self.self_attn, MultiHeadedAttention):
            query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         type="self")
        elif isinstance(self.self_attn, AverageAttention):
            query, attn = self.self_attn(input_norm, mask=dec_mask,
                                         layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)

        output = self.feed_forward(query_norm)
        output = output + query

        return output, attn


class TransformerGPTDecoderLayerCtxattn(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, attn_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0,
                 ctx_weight_param=False):
        super(TransformerGPTDecoderLayerCtxattn, self).__init__()

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=attn_dropout,
                max_relative_positions=max_relative_positions)
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(d_model, dropout=attn_dropout)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = MLP(d_model, d_model*4, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-5)
        self.context_layer_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.drop = nn.Dropout(dropout)

        if ctx_weight_param:
            print('using ctx_weight_param')
            self.ctx_weight = Parameter(torch.zeros(1))
        self.ctx_weight_param = ctx_weight_param

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                layer_cache=None, step=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attn ``(batch_size, 1, src_len)``

        """
        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8)
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            future_mask = future_mask.bool()
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        input_norm = self.layer_norm_1(inputs)

        if isinstance(self.self_attn, MultiHeadedAttention):
            query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         type="self")
        elif isinstance(self.self_attn, AverageAttention):
            query, attn = self.self_attn(input_norm, mask=dec_mask,
                                         layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)

        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      type="context")
        mid = self.drop(mid)
        if self.ctx_weight_param:
            mid = mid*self.ctx_weight
        mid += query
        mid_norm = self.context_layer_norm(mid)

        output = self.feed_forward(mid_norm)
        #output = self.feed_forward(query_norm)
        output = output + mid

        return output, attn

class TransformerGPTDecoderLayerPSA(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, attn_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0,
                 ctx_weight_param=False):
        super(TransformerGPTDecoderLayerPSA, self).__init__()
        
        # This is called self for easier loading of gpt params
        self.self_attn = JointMultiHeadedAttention(
            heads, d_model, dropout=attn_dropout,
            max_relative_positions=max_relative_positions,
            ctx_weight_param=ctx_weight_param)

        self.feed_forward = MLP(d_model, d_model*4, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-5)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-5)
        self.drop = nn.Dropout(dropout)
        #self.ctx_weight = Parameter(torch.zeros(1))

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                layer_cache=None, step=None, evaluate_attns=False):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attn ``(batch_size, 1, src_len)``

        """
        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8)
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            future_mask = future_mask.bool()
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        input_norm = self.layer_norm_1(inputs)

        query, attn, all_attn_probs = self.self_attn(input_norm, memory_bank, self_mask=dec_mask,
                                                     ctx_mask=src_pad_mask, layer_cache=layer_cache)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)

        output = self.feed_forward(query_norm)
        output = output + query

        if evaluate_attns:
            return output, attn, all_attn_probs
        else:
            return output, attn


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, attn_dropout,
                 self_attn_type="scaled-dot", max_relative_positions=0):
        super(TransformerDecoderLayer, self).__init__()

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads, d_model, dropout=attn_dropout,
                max_relative_positions=max_relative_positions)
        elif self_attn_type == "average":
            self.self_attn = AverageAttention(d_model, dropout=attn_dropout)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                layer_cache=None, step=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, 1, model_dim)``
            * attn ``(batch_size, 1, src_len)``

        """
        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8)
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            future_mask = future_mask.bool()
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        input_norm = self.layer_norm_1(inputs)

        if isinstance(self.self_attn, MultiHeadedAttention):
            query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                         mask=dec_mask,
                                         layer_cache=layer_cache,
                                         type="self")
        elif isinstance(self.self_attn, AverageAttention):
            query, attn = self.self_attn(input_norm, mask=dec_mask,
                                         layer_cache=layer_cache, step=step)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      type="context")
        output = self.feed_forward(self.drop(mid) + query)

        return output, attn


class TransformerDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       copy_attn (bool): if using a separate copy attention
       self_attn_type (str): type of self-attention scaled-dot, average
       dropout (float): dropout parameters
       embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attn_dropout, embeddings,
                 max_relative_positions, use_GPT_version_psa,
                 use_GPT_version_unconditional, use_GPT_version_ctxattn,
                 ctx_weight_param):
        super(TransformerDecoder, self).__init__()

        self.embeddings = embeddings

        # Decoder State
        self.state = {}
        
        kwargs = {}
        if use_GPT_version_ctxattn:
            layer_cls = TransformerGPTDecoderLayerCtxattn
        elif use_GPT_version_psa:
            layer_cls = TransformerGPTDecoderLayerPSA
            kwargs['ctx_weight_param'] = ctx_weight_param
        elif use_GPT_version_unconditional:
            layer_cls = TransformerGPTUnconditionalDecoderLayer
        else:
            layer_cls = TransformerDecoderLayer

        self.transformer_layers = nn.ModuleList(
            [layer_cls(d_model, heads, d_ff, dropout, attn_dropout,
             self_attn_type=self_attn_type,
             max_relative_positions=max_relative_positions,
             **kwargs)
             for i in range(num_layers)])

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.dec_heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout,
            opt.attn_dropout if hasattr(opt, 'attn_dropout') else opt.dropout,
            embeddings,
            opt.max_relative_positions,
            opt.use_GPT_version_psa,
            opt.use_GPT_version_unconditional,
            opt.use_GPT_version_ctxattn,
            opt.ctx_weight_param)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)
        
        if self.state["src"] is not None:
            self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach() if self.state["src"] is not None else None

    def forward(self, tgt, memory_bank, step=None, **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_bank)

        # emb = self.embeddings(tgt, step=step)
        # assert emb.dim() == 3  # len x batch x embedding_dim
        inp_t = tgt

        sample_prob = getattr(self, '_decoder_sampling', 0.0)
        decoder_sampling_k = getattr(self, '_parallel_sampling_k', 1)
        decoder_sampling_sequence = getattr(self, '_decoder_sampling_sequence', False)

        if sample_prob != 0.0 and decoder_sampling_k == 0:
            raise ValueError("parallel_sampling_k can't be 0 w/ decoder_sampling != 0.0 (%f)" % sample_prob)

        # output = emb.transpose(0, 1).contiguous()
        pad_idx = self.embeddings.word_padding_idx
        
        src = self.state["src"]
        if src is not None:
            src_words = src[:, :, 0].transpose(0, 1)
            src_batch, src_len = src_words.size()
            src_pad_mask = src_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_src]
        else:
            src_pad_mask = None

        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        tgt_words = tgt[:, :, 0].transpose(0, 1)
        tgt_batch, tgt_len = tgt_words.size()
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        save_all_attns = kwargs.get('evaluate_attns', False)
        all_attns_full = []

        while True:
            emb = self.embeddings(inp_t, step=step)
            assert emb.dim() == 3  # len x batch x embedding_dim
            output = emb.transpose(0, 1).contiguous()
            src_memory_bank = memory_bank.transpose(0, 1).contiguous()


            for i, layer in enumerate(self.transformer_layers):
                layer_cache = self.state["cache"]["layer_{}".format(i)] \
                    if step is not None else None
                if save_all_attns:
                    output, attn, all_attns = layer(
                        output,
                        src_memory_bank,
                        src_pad_mask,
                        tgt_pad_mask,
                        layer_cache=layer_cache,
                        step=step,
                        evaluate_attns=True)
                    all_attns_full.append(all_attns)
                else:
                    output, attn = layer(
                        output,
                        src_memory_bank,
                        src_pad_mask,
                        tgt_pad_mask,
                        layer_cache=layer_cache,
                        step=step)

            output = self.layer_norm(output)
            dec_outs = output.transpose(0, 1).contiguous()
            attn = attn.transpose(0, 1).contiguous()

            attns = {"std": attn}
            if self._copy:
                attns["copy"] = attn
            if save_all_attns:
                attns['full_all_layers'] = all_attns_full

            if sample_prob == 0.0 or decoder_sampling_k == 0:
                break
            else:
                decoder_sampling_k -= 1

                _output = output
                _copy_attn = attn
                _loss = self._loss
                _batch = self._batch
                import onmt
                if isinstance(_loss.generator, onmt.modules.CopyGenerator):
                    _scores, _p_copy = _loss.generator(
                        _loss._bottle(_output), _loss._bottle(_copy_attn), _batch.src_map)
                    from onmt.modules.copy_generator import collapse_copy_scores
                    scores_data = collapse_copy_scores(
                        _loss._unbottle(_scores, _batch.batch_size),
                        _batch, 
                        _loss.tgt_vocab,
                        _batch.dataset.src_vocabs)
                    scores_data = scores_data[:, :, :len(_loss.tgt_vocab)]
                else:
                    _scores = _loss.generator(
                        _loss._bottle(_output))
                    scores_data = torch.exp(_loss._unbottle(_scores, _batch.batch_size))

                # we ignore last prediction scores (supposedly <eos>)
                scores_data = scores_data[:-1, :, :]
                scores_data = _loss._bottle(scores_data)

                if not self._decoder_greedy:
                    pred_t = torch.multinomial(scores_data, 1).to(tgt.device)
                else:
                    pred_t = (scores_data.max(dim=1)).indices.view(-1, 1).to(tgt.device)
                # print("[k=%d] pred" % decoder_sampling_k, _loss._unbottle(pred_t, _batch.batch_size)[:dsteps, :dinputs])
                pred_prob = torch.rand(pred_t.size()).to(tgt.device)

                if decoder_sampling_sequence:
                    # threshold the probability for the whole sequence
                    # i.e. prob is either 1 or 0 based on pred_prob[0]
                    
                    # we only use the first value to test
                    # 1=>sample the whole sequence ; 0=>do not sample
                    switch = (pred_prob.view(-1)[0] < sample_prob).float()
                    
                    # switch=1 => all probs to 0
                    # (so it's < sample prob, so sampled)
                    pred_prob.gt_(switch)
                    

                # print("sample prob: %f" % sample_prob)
                # print(pred_prob)
                pred_mask = (pred_prob < sample_prob).long()


                # we ignore first ref input for now (<bos>)
                _tgt_no_bos = tgt[1:, :, 0]
                bottled_tgt = _loss._bottle(tgt[1:, :, :])
                pred_t = pred_t * pred_mask + bottled_tgt * (1-pred_mask)
                pred_t = _loss._unbottle(pred_t, _batch.batch_size)

                # shift predictions to get next input 
                # i.e. put <bos> back and remove <eos>
                inp_t = torch.cat([inp_t[0:1], pred_t], dim=0)

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth))
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache
