# HyperEncoding

Training and evaluation code for spiking neural networks with temporal input encodings. The repo focuses on comparing how different encodings behave under clean training and several robustness settings.

The main training path is:

1. Load an image dataset.
2. Keep image tensors in `[0, 1]`.
3. Encode each image across `T` simulation steps.
4. Train an SNN model with BPTT or BPTR-style temporal backprop.
5. Save checkpoints and logs for clean or adversarial runs.

## What Is Included

- `main_train.py`: training entrypoint.
- `main_test.py`: checkpoint evaluation entrypoint.
- `data_loaders.py`: CIFAR-10, CIFAR-100, SVHN, MNIST, Fashion-MNIST, and ImageNet-100 loaders.
- `models/`: VGG and SEW-ResNet SNN models.
- `models/layers.py`: temporal expansion, spike functions, and input encoders.
- `attacks/`: clean/noisy/adversarial evaluation and training helpers.

Supported input encodings are:

- `const`: repeat the same image over all time steps.
- `rate`: Bernoulli rate encoding.
- `hypergeometric`: sparse sampling without replacement across time.
- `signed`: legacy signed-rate branch; use with care because the current encoder still expects bounded inputs.

For `rate` and `hypergeometric`, inputs must stay in `[0, 1]`. Do not use `--center` for those runs unless you know the selected path still preserves that range.

## Requirements

Basic setup:

```bash
conda create -n hyperencoding python=3.10 -y
conda activate hyperencoding
```

Install PyTorch separately so it matches your GPU/CUDA driver. For example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies used by the code:

- Python 3.10
- PyTorch and torchvision
- spikingjelly
- datasets
- numpy
- auto-attack, mainly for optional robustness evaluation

The training scripts expect a CUDA GPU. They intentionally stop on CPU.

## Datasets

Small datasets such as CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, and SVHN download through torchvision into `data/`.

ImageNet-100 is loaded from the Hugging Face `imagenet-1k` dataset by selecting labels `< 100`. The dataset is gated, so you need to accept the terms on Hugging Face and authenticate first:

```bash
huggingface-cli login
```

Optional helper scripts are available:

```bash
python scripts/download_imagenet100.py
python scripts/download_imagenet1000.py --token $HF_TOKEN
```

## One Simple Training Cycle

A quick sanity run on CIFAR-10 for one epoch:

```bash
python main_train.py \
  --dataset cifar10 \
  --model vgg11 \
  --encoding const \
  --time 4 \
  --epochs 1 \
  --batch_size 128 \
  --device 0 \
  --suffix _smoke
```

This should:

- download CIFAR-10 if needed;
- train for one epoch;
- print the best test accuracy;
- write logs and checkpoints under `cifar10-checkpoints/`.

The checkpoint names are generated from the dataset, model, encoding, time, attack, and suffix. For the command above, look for files similar to:

```text
cifar10-checkpoints/model_vgg11_encoding_const_Time_4_atck_clean_smoke.log
cifar10-checkpoints/model_vgg11_encoding_const_Time_4_atck_clean_smoke.pth
cifar10-checkpoints/model_vgg11_encoding_const_Time_4_atck_clean_smoke_last.pth
```

## Common Training Commands

Clean CIFAR-10 with rate encoding:

```bash
python main_train.py --dataset cifar10 --model vgg11 --encoding rate --time 4 --device 0
```

Clean ImageNet-100 with SEW-ResNet:

```bash
python main_train.py --dataset imagenet100 --model sewresnet --encoding const --time 4 --device 0
```

Run the three ImageNet-100 encodings on separate GPUs:

```bash
python main_train.py --dataset imagenet100 --model sewresnet --encoding const --time 4 --device 1
python main_train.py --dataset imagenet100 --model sewresnet --encoding rate --time 4 --device 2
python main_train.py --dataset imagenet100 --model sewresnet --encoding hypergeometric --time 4 --device 3
```

Resume a matching run:

```bash
python main_train.py --dataset cifar10 --model vgg11 --encoding const --time 4 --device 0 --resume
```

## Evaluation

Evaluate a saved checkpoint with `main_test.py`:

```bash
python main_test.py \
  --dataset cifar10 \
  --model vgg11 \
  --encoding const \
  --time 4 \
  --device 0 \
  --id cifar10-checkpoints/model_vgg11_encoding_const_Time_4_atck_clean_smoke.pth
```

Testing writes result logs under `<dataset>-Results/`.

## Attacks And Robustness

Training and testing support these attack names through `--attack`:

- `fgsm`
- `pgd`
- `gn`
- `sea`
- `retiming_l0`, `retiming_l1`, `retiming_linf`

Example PGD training command:

```bash
python main_train.py \
  --dataset cifar10 \
  --model vgg11 \
  --encoding rate \
  --time 4 \
  --attack pgd \
  --eps 8 \
  --alpha 2 \
  --steps 4 \
  --device 0
```

`fgsm`, `pgd`, and `gn` use image-space epsilon scaled by `/255`. `sea` and retiming attacks use their own temporal/sparse interpretation.

## What To Expect

- A one-epoch smoke run only checks that the pipeline works; accuracy will not be meaningful.
- Full clean training defaults to `200` epochs with SGD, momentum `0.9`, weight decay `5e-4`, and cosine LR scheduling.
- ImageNet-100 runs are much slower than CIFAR runs and require enough GPU memory for `224x224` images and temporal batches.
- Logs are written both to the terminal and to `<dataset>-checkpoints/*.log`.
- Best and latest checkpoints are saved as `.pth` files. The `_last.pth` file is used by `--resume`.

## Notes

- Keep `--center` off for encodings that require `[0, 1]` inputs.
- Use a unique `--suffix` when comparing experiments so old checkpoints are not overwritten.
- If a run seems stuck near chance accuracy, first check that the model is not resuming an old incompatible checkpoint and that the input encoding matches the preprocessing assumptions.
