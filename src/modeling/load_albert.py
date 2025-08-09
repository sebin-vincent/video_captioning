from src.layers.albert import AlbertTokenizer, AlbertConfig, AlbertForImageCaptioning
from src.utils.logger import LOGGER as logger

def get_albert_model(args):
    # Load pretrained albert and tokenizer based on training configs
    config_class, model_class, tokenizer_class = AlbertConfig, AlbertForImageCaptioning, AlbertTokenizer
    config = config_class.from_pretrained(args.config_name if args.config_name else \
            args.model_name_or_path, num_labels=2, finetuning_task='image_captioning')

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
            else args.model_name_or_path, do_lower_case=args.do_lower_case)
    config.img_feature_type = 'frcnn'
    config.hidden_dropout_prob = getattr(args, 'drop_out', 0.1)
    config.loss_type = 'classification'
    config.tie_weights = getattr(args, 'tie_weights', True)
    config.freeze_embedding = getattr(args, 'freeze_embedding', False)
    config.label_smoothing = getattr(args, 'label_smoothing', 0.0)
    config.drop_worst_ratio = getattr(args, 'drop_worst_ratio', 0.0)
    config.drop_worst_after = getattr(args, 'drop_worst_after', 0)
    # update model structure if specified in arguments
    update_params = ['img_feature_dim', 'num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
    model_structure_changed = [False] * len(update_params)
    # model_structure_changed[0] = True  # cclin hack
    for idx, param in enumerate(update_params):
        arg_param = getattr(args, param, -1)  # Default to -1 if not present
        # albert-base-v2 do not have img_feature_dim
        config_param = getattr(config, param) if hasattr(config, param) else -1
        if arg_param > 0 and arg_param != config_param:
            logger.info(f"Update config parameter {param}: {config_param} -> {arg_param}")
            setattr(config, param, arg_param)
            model_structure_changed[idx] = True
    if any(model_structure_changed):
        assert config.hidden_size % config.num_attention_heads == 0
        if getattr(args, 'load_partial_weights', False):
            # can load partial weights when changing layer only.
            assert not any(model_structure_changed[2:]), "Cannot load partial weights " \
                "when any of ({}) is changed.".format(', '.join(update_params[2:]))
            model = model_class.from_pretrained(args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
            logger.info("Load partial weights for albert layers.")
        else:
            model = model_class(config=config) # init from scratch
            logger.info("Init model from scratch.")
    else:
        # Try to load pretrained model, fallback to scratch if pytorch_model.bin doesn't exist
        import os
        model_bin_path = os.path.join(args.model_name_or_path, 'pytorch_model.bin')
        if os.path.exists(model_bin_path):
            model = model_class.from_pretrained(args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
            logger.info(f"Load pretrained model: {args.model_name_or_path}")
        else:
            logger.info(f"pytorch_model.bin not found in {args.model_name_or_path}, initializing from scratch")
            model = model_class(config=config)
            logger.info("Init ALBERT model from scratch.")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model total parameters: {total_params}')
    return model, config, tokenizer
