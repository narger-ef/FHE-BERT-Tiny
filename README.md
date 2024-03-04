# FHE-BERT-Tiny

This repository contains the source code for the Work-in-Progress paper called *Transformer-based Language Models and Homomorphic Encryption: an intersection with BERT-tiny*. In particular, in contains a FHE-based circuit that implements the Transformer Encoder layers of BERT-tiny (available [here](https://huggingface.co/philschmid/tiny-bert-sst2-distilled)), fine-tuned on the SST-2 dataset.

## Prerequisites

Linux or Mac operative system

In order to run the program, you need to install:
- `cmake`
- `g++` or `clang`
- `OpenFHE` ([how to install OpenFHE](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/installation.html))

Plus, since the tokenization process relies on PyTorch:
- `python`
- `pip`

## How to use it
After intalling all the required prerequisites, install the required Python libraries using pip:
```
pip install -r src/requirements.txt
```

Then, it is possible to generate the set of keys for the CKKS scheme. Go to the `build` folder:

```
cd build
```

and run the following command:

```
./FHE-BERT-Tiny --generate_keys
```

This generates the required keys to evaluate the circuit. Optionally, it is possible to generate keys satisfying $\lambda = 128$ bits of security by adding the following flag (notice that this will generate a larger ring, leading to larger runtimes).

```
./FHE-BERT-Tiny --generate_keys --secure
```

This command will generate a `keys` folder in the root of the project folder, containing the serializations of the required keys. Now it is possible to run the FHE circuit by using this command

```
./FHE-BERT-Tiny "Dune part 2 was a mesmerizing experience, movie of the year?"
```

In general, the circuit can be evaluated as follows (after the generation of the keys):

```
./FHE-BERT-Tiny <text> [OPTIONS]
```
where

- `<text>` is the input text to be evaluated

and the optional `[OPTIONS]` parameters are:

- `--verbose` prints information during the evaluation of the network. It can be useful to study the precision of the circuit at the end of each layer
- `--plain` adds the result of the plain circuit at the end of the FHE evaluation


## Authors

- Lorenzo Rovida (`lorenzo.rovida@unimib.it`)
- Alberto Leporati (`alberto.leporati@unimib.it`)

Made with <3  at [Bicocca Security Lab](https://www.bislab.unimib.it), at University of Milan-Bicocca.

<img src="imgs/lab_logo.png" alt="BisLab logo" width=20%>

### Declaration

This is a proof of concept and, even though parameters are created with $\lambda \geq 128$ security bits (according to [Homomorphic Encryption Standards](https://homomorphicencryption.org/standard)), this circuit is intended for educational purposes only.
