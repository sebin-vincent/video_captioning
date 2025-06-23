import unittest
import torch
from src.layers.bert.modeling_bert import BertConfig, BertSelfAttention, BertAttention, BertLayer, BertEncoder, BertImgModel, BertForImageCaptioning
# from src.modeling.video_captioning_e2e_vid_swin_bert import VideoTransformer # Deferred
# from src.modeling.load_swin import get_swin_model # Or a mock for VideoTransformer

# Configure logger to avoid issues if it's used by models during init and not configured
import logging
logging.basicConfig(level=logging.INFO)


class TestAttentionOutput(unittest.TestCase):

    def setUp(self):
        self.bert_config = BertConfig(
            hidden_size=12, # Small for testing
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=24,
            output_attentions=True, # Key setting for these tests
            output_hidden_states=False, # Keep False to simplify output tuple lengths
            img_feature_dim=10,
            max_position_embeddings=30, # Increased to cover text + img
            vocab_size=100,
            img_layer_norm_eps=1e-12, # for BertImgModel
            type_vocab_size=2, # default
        )
        self.batch_size = 1
        self.seq_length = 5 # Text sequence length
        self.img_seq_len = 3  # Image feature sequence length
        self.hidden_size = self.bert_config.hidden_size
        self.num_heads = self.bert_config.num_attention_heads
        self.num_layers = self.bert_config.num_hidden_layers

        self.dummy_input_ids = torch.randint(0, self.bert_config.vocab_size, (self.batch_size, self.seq_length))
        self.dummy_attention_mask = torch.ones((self.batch_size, self.seq_length), dtype=torch.long)
        # For BertImgModel, attention_mask needs to cover text + image features
        self.full_attention_mask = torch.ones((self.batch_size, self.seq_length + self.img_seq_len), dtype=torch.long)

        self.dummy_img_feats = torch.randn(self.batch_size, self.img_seq_len, self.bert_config.img_feature_dim)

        self.dummy_hidden_states = torch.randn(self.batch_size, self.seq_length, self.hidden_size)
        # Extended attention mask for self-attention layers (original seq_length, not text+img)
        extended_mask_for_bert_internal = self.dummy_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_mask_for_bert_internal = extended_mask_for_bert_internal.to(dtype=torch.float32) # For (1.0 - mask)
        self.extended_attention_mask_bert_internal = (1.0 - extended_mask_for_bert_internal) * -10000.0


    def test_bert_self_attention_output(self):
        layer = BertSelfAttention(self.bert_config)
        outputs = layer(self.dummy_hidden_states, self.extended_attention_mask_bert_internal)
        self.assertTrue(len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}")
        attention_probs = outputs[1]
        self.assertEqual(attention_probs.shape, (self.batch_size, self.num_heads, self.seq_length, self.seq_length))

    def test_bert_attention_output(self):
        layer = BertAttention(self.bert_config)
        outputs = layer(self.dummy_hidden_states, self.extended_attention_mask_bert_internal)
        self.assertTrue(len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}")
        attention_probs = outputs[1]
        self.assertEqual(attention_probs.shape, (self.batch_size, self.num_heads, self.seq_length, self.seq_length))

    def test_bert_layer_output(self):
        layer = BertLayer(self.bert_config)
        outputs = layer(self.dummy_hidden_states, self.extended_attention_mask_bert_internal)
        self.assertTrue(len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}")
        attention_probs = outputs[1]
        self.assertEqual(attention_probs.shape, (self.batch_size, self.num_heads, self.seq_length, self.seq_length))

    def test_bert_encoder_output(self):
        encoder = BertEncoder(self.bert_config)
        # BertConfig output_hidden_states=False, output_attentions=True
        # Expected output: (final_hidden_states, all_attentions_tuple)
        outputs = encoder(self.dummy_hidden_states, self.extended_attention_mask_bert_internal)
        self.assertTrue(len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}")
        all_attentions = outputs[1]
        self.assertIsInstance(all_attentions, tuple)
        self.assertEqual(len(all_attentions), self.num_layers)
        for attention_probs in all_attentions:
            self.assertEqual(attention_probs.shape, (self.batch_size, self.num_heads, self.seq_length, self.seq_length))

    def test_bert_img_model_output(self):
        model = BertImgModel(self.bert_config)
        # BertConfig output_hidden_states=False, output_attentions=True
        # Expected output: (sequence_output, pooled_output, all_attentions_tuple)
        outputs = model(input_ids=self.dummy_input_ids, img_feats=self.dummy_img_feats, attention_mask=self.full_attention_mask)
        self.assertTrue(len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}")
        all_attentions = outputs[2]
        self.assertIsInstance(all_attentions, tuple)
        self.assertEqual(len(all_attentions), self.num_layers)
        expected_full_seq_len = self.seq_length + self.img_seq_len
        for attention_probs in all_attentions:
            self.assertEqual(attention_probs.shape, (self.batch_size, self.num_heads, expected_full_seq_len, expected_full_seq_len))

    def test_bert_for_image_captioning_encode_forward(self):
        model = BertForImageCaptioning(self.bert_config)
        # BertConfig output_hidden_states=False, output_attentions=True
        # encode_forward (is_training=False) returns: (class_logits,) + bert_img_model_outputs[2:]
        # bert_img_model_outputs[2:] is (all_attentions_tuple)
        # Expected: (class_logits, all_attentions_tuple)
        outputs = model.encode_forward(
            input_ids=self.dummy_input_ids,
            img_feats=self.dummy_img_feats,
            attention_mask=self.full_attention_mask, # Mask for text+img
            is_training=False
        )
        self.assertTrue(len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}")
        all_attentions = outputs[1]
        self.assertIsInstance(all_attentions, tuple)
        self.assertEqual(len(all_attentions), self.num_layers)
        expected_full_seq_len_for_att = self.seq_length + self.img_seq_len
        for attention_probs in all_attentions:
            self.assertEqual(attention_probs.shape, (self.batch_size, self.num_heads, expected_full_seq_len_for_att, expected_full_seq_len_for_att))

    def test_bert_for_image_captioning_prod_generate(self):
        # This config will be used by the model internally
        self.bert_config.output_attentions = True
        self.bert_config.output_hidden_states = False # To match other tests easily

        model = BertForImageCaptioning(self.bert_config)
        # Ensure the model instance itself reflects this, if it caches config values,
        # or if its internal components don't re-read from self.bert_config directly.
        # BertForImageCaptioning uses self.bert.config, which is self.bert_config.
        # The set_output_attentions on encoder should be handled by BertImgModel init based on config.

        img_feats_prod = torch.randn(1, self.img_seq_len, self.bert_config.img_feature_dim)
        od_label_ids_prod = torch.randint(0, self.bert_config.vocab_size, (1, 2))

        max_gen_len = 3
        bos_token_id = self.bert_config.vocab_size - 1 # Ensure it's a valid ID
        eos_token_id = self.bert_config.vocab_size - 2
        mask_token_id = self.bert_config.vocab_size - 3

        outputs = model.prod_generate(
            img_feats=img_feats_prod,
            od_label_ids=od_label_ids_prod,
            max_length=max_gen_len, # Max length of generated sequence (BOS + N tokens)
            bos_token_id=bos_token_id,
            eos_token_ids=[eos_token_id], # Must be a list
            mask_token_id=mask_token_id,
            od_labels_start_posid=self.seq_length + 10
        )

        self.assertEqual(len(outputs), 3, f"Expected 3 outputs, got {len(outputs)}")
        generated_ids, logprob, all_step_attentions = outputs

        self.assertIsInstance(all_step_attentions, list)

        # Number of generation steps = number of tokens after BOS.
        # If generated_ids is (1, L), then L-1 tokens were generated after BOS.
        num_generated_tokens_after_bos = generated_ids.shape[1] - 1

        if num_generated_tokens_after_bos < 0: # Should not happen, at least BOS
            num_generated_tokens_after_bos = 0

        self.assertEqual(len(all_step_attentions), num_generated_tokens_after_bos)

        for step_attentions_tuple in all_step_attentions:
            self.assertIsInstance(step_attentions_tuple, tuple)
            self.assertEqual(len(step_attentions_tuple), self.num_layers)
            for layer_attention_tensor in step_attentions_tuple:
                self.assertEqual(layer_attention_tensor.ndim, 4) # B,H,S,S
                self.assertEqual(layer_attention_tensor.shape[0], 1) # Batch size for prod_generate is 1
                self.assertEqual(layer_attention_tensor.shape[1], self.num_heads)
                # Seq len S can vary per step, so not checking S x S exactly here.
                self.assertTrue(layer_attention_tensor.shape[2] > 0)
                self.assertEqual(layer_attention_tensor.shape[2], layer_attention_tensor.shape[3])


if __name__ == '__main__':
    unittest.main()
