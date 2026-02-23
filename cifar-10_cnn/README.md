# CIFAR-10 Edge CNNs
This repo contains a small, medium, and large CNN trained on the CIFAR-10 dataset.
These models were developed using a Bayesian neural architecture search process, except for the large one, which was provided by the Tiny MLPerf Benchmark (`NB: we remove the average pooling layer to improve quantization-aware training accuracy`).

Models provided:
* Small Model
* Medium Model
* Large Model (`NB: This is a ResNet`)

## Dataset
To evaluate the model, download the dataset [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Load the model
We provide a script to load the model for further evaluation: `./scripts/load_model.sh`. 
No need to download the dataset to load the model.

### QKeras versions
To load the Medium model, run:
```
./scripts/load_model.sh 0
```

To load the Small model, run:
```
./scripts/load_model.sh 1
```

To load the Large model, run:
```
./scripts/load_model.sh 2
```
