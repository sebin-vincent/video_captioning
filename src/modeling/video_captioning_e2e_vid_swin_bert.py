import torch
from fairscale.nn.misc import checkpoint_wrapper
import random


class VideoTransformer(torch.nn.Module):
    def __init__(self, args, config, swin, transformer_encoder):
        super(VideoTransformer, self).__init__()
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        else:
            self.swin = swin
        self.trans_encoder = transformer_encoder
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)
        self.compute_mask_on_the_fly = False # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length
        # learn soft attention mask
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard', False)
        
        if self.learn_mask_enabled==True:
            self.learn_vid_att = torch.nn.Embedding(args.max_img_seq_length*args.max_img_seq_length,1)
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, *args, **kwargs):
        images = kwargs['img_feats']
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        images = images.permute(0, 2, 1, 3, 4)
        vid_feats = self.swin(images)
        if self.use_grid_feat==True:
            vid_feats = vid_feats.permute(0, 2, 3, 4, 1)
        vid_feats = vid_feats.view(B, -1, self.latent_feat_size)
        vid_feats = self.fc(vid_feats)
        # prepare VL transformer inputs
        kwargs['img_feats'] = vid_feats

        # Ensure output_attentions is False for this forward pass if it was configured to be True.
        encoder_module = None
        is_bert_model = False
        if hasattr(self.trans_encoder, 'bert') and hasattr(self.trans_encoder.bert, 'encoder'):
            encoder_module = self.trans_encoder.bert.encoder
            is_bert_model = True
        elif hasattr(self.trans_encoder, 'roberta') and hasattr(self.trans_encoder.roberta, 'encoder'): # Assuming RobertaForImageCaptioning has a 'roberta' attribute
            encoder_module = self.trans_encoder.roberta.encoder
        
        if encoder_module is not None:
            # Check the configuration if attentions would be outputted by default for this call
            default_output_attentions = False
            # For both BertEncoder and RobertaEncoder (from HF), the 'output_attentions' attribute directly reflects the config.
            if hasattr(encoder_module, 'output_attentions'): 
                default_output_attentions = encoder_module.output_attentions
            elif hasattr(encoder_module, 'config') and hasattr(encoder_module.config, 'output_attentions'): # Fallback if direct attribute not found
                default_output_attentions = encoder_module.config.output_attentions

            if default_output_attentions:
                if is_bert_model and hasattr(encoder_module, 'set_output_attentions') and callable(encoder_module.set_output_attentions):
                    # This will modify the internal state of BertEncoder, including its config.output_attentions
                    encoder_module.set_output_attentions(False)
                else:
                    # For RoBERTa (or if BERT model somehow misses set_output_attentions)
                    # we pass output_attentions=False via kwargs to the model's forward call.
                    # This will override the config for the duration of that call.
                    kwargs['output_attentions'] = False
        
        # learn soft attention mask
        if self.learn_mask_enabled:
            kwargs['attention_mask'] = kwargs['attention_mask'].float()
            vid_att_len = self.max_img_seq_length
            current_device = kwargs['img_feats'].device # Get device from an existing tensor in kwargs
            learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
            learn_att = self.sigmoid(learn_att)
            diag_mask = torch.diag(torch.ones(vid_att_len, device=current_device))
            video_attention = (1. - diag_mask)*learn_att
            learn_att = diag_mask + video_attention
            if self.sparse_mask_soft2hard:
                learn_att = (learn_att>=0.5)*1.0
                # learn_att is already on the correct device if self.learn_vid_att.weight is.
                # If learn_vid_att is on CPU and other parts on GPU, it needs .to(current_device)
                # Assuming learn_vid_att is on the same device or handled by nn.Module's to() method.
                learn_att.requires_grad = False # This should be fine
            kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att
        
        outputs = self.trans_encoder(*args, **kwargs)
        if self.learn_mask_enabled:
            loss_sparsity = self.get_loss_sparsity(video_attention)  
            outputs = outputs + (loss_sparsity, )
        return outputs
    
    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss

    def diag_based_init_attn_mask(self, pretrain_attn_mask):
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        zeros_mask = torch.zeros_like(pretrained_learn_att)
        scale_factor = self.max_img_seq_length/pretrained_num_tokens
        
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 


    def bilinear_init_attn_mask(self, pretrain_attn_mask):
        print('init attn mask with bilinear interpolation')
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        scale_factor = int(self.max_img_seq_length/pretrained_num_tokens)
        sampler = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        with torch.no_grad():
            learn_att = sampler(pretrained_learn_att[None,None,:,:].double())[0,0,:,:].half()

    def random_init_attn_mask(self):
        print('random init attn mask')
        self.learn_vid_att = torch.nn.Embedding(self.max_img_seq_length*self.max_img_seq_length,1)


    def reload_attn_mask(self, pretrain_attn_mask): 
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 

    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad =  not freeze

 