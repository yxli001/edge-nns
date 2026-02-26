import yaml
import models


def yaml_load(config):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param


def load_model(model_config, pretrained_model=None):
    if isinstance(model_config, str):
        config = yaml_load(model_config)
    else:
        config = model_config
    
    dense_widths = config['model']['dense_widths']
    logit_total_bits = config["quantization"]["logit_total_bits"]
    logit_int_bits = config["quantization"]["logit_int_bits"]
    activation_total_bits = config["quantization"]["activation_total_bits"]
    activation_int_bits = config["quantization"]["activation_int_bits"]
    alpha = config["quantization"]["alpha"]
    logit_quantizer = config["quantization"]["logit_quantizer"]
    activation_quantizer = config["quantization"]["activation_quantizer"]
    input_shape = tuple(config['data']['input_shape'])
    
    model = models.qkeras_dense_model(
        in_shape=input_shape,
        dense_widths=dense_widths,
        logit_total_bits=logit_total_bits,
        logit_int_bits=logit_int_bits,
        activation_total_bits=activation_total_bits,
        activation_int_bits=activation_int_bits,
        alpha=alpha,
        logit_quantizer=logit_quantizer,
        activation_quantizer=activation_quantizer
    )
    
    if pretrained_model:
        model.load_weights(pretrained_model)
    
    return model