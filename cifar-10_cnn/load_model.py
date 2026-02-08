import os
if os.system('nvidia-smi') == 0:
    import setGPU
import argparse
import tensorflow as tf
import resnet_v1_eembc
import yaml

def yaml_load(config):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param

def load_model(model_config, pretrained_model=None):
    config = yaml_load(model_config)
    input_shape = [int(i) for i in config['data']['input_shape']]
    num_classes = int(config['data']['num_classes'])
    num_filters = config['model']['filters']
    kernel_sizes = config['model']['kernels']
    strides = config['model']['strides']
    l1p = float(config['model']['l1'])
    l2p = float(config['model']['l2'])
    skip = bool(config['model']['skip'])
    avg_pooling = bool(config['model']['avg_pooling'])
    model_name = config['model']['name']

    # quantization parameters
    if 'quantized' in model_name:
        logit_total_bits = config["quantization"]["logit_total_bits"]
        logit_int_bits = config["quantization"]["logit_int_bits"]
        activation_total_bits = config["quantization"]["activation_total_bits"]
        activation_int_bits = config["quantization"]["activation_int_bits"]
        alpha = config["quantization"]["alpha"]
        use_stochastic_rounding = config["quantization"]["use_stochastic_rounding"]
        logit_quantizer = config["quantization"]["logit_quantizer"]
        activation_quantizer = config["quantization"]["activation_quantizer"]
        final_activation = bool(config['model']['final_activation'])

    kwargs = {
        'input_shape': input_shape,
        'num_classes': num_classes,
        'num_filters': num_filters,
        'kernel_sizes': kernel_sizes,
        'strides': strides,
        'l1p': l1p,
        'l2p': l2p,
        'skip': skip,
        'avg_pooling': avg_pooling
    }
    # pass quantization params
    if 'quantized' in model_name:
        kwargs["logit_total_bits"] = logit_total_bits
        kwargs["logit_int_bits"] = logit_int_bits
        kwargs["activation_total_bits"] = activation_total_bits
        kwargs["activation_int_bits"] = activation_int_bits
        kwargs["alpha"] = None if alpha == 'None' else alpha
        kwargs["use_stochastic_rounding"] = use_stochastic_rounding
        kwargs["logit_quantizer"] = logit_quantizer
        kwargs["activation_quantizer"] = activation_quantizer
        kwargs["final_activation"] = final_activation

    # define model
    model = getattr(resnet_v1_eembc, model_name)(**kwargs)
    # Load pretrained model weights
    if pretrained_model:
        model.load_weights(pretrained_model)
    return model


def main(args):
    model = load_model(args.config, pretrained_model=args.pretrained_model)
    print(model.summary())
    print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', 
        '--config', 
        type=str, 
        default="./model_ckpts/resnet_v1_eembc_quantized_baseline_fkeras/baseline_fkeras.yml", 
        help="specify yaml config"
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="specify pretrained model file path",
    )

    args = parser.parse_args()

    main(args)