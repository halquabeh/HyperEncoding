# SignedRateEncoding
Code corresponding to the paper (https://openreview.net/forum?id=qLh6Ufvnuc)

**IMPROVING GENERALIZATION AND ROBUSTNESS IN SNNS THROUGH SIGNED RATE ENCODING AND SPARSE ENCODING ATTACKS**

Bhaskar Mukhoty, Hilal AlQuabeh, Bin Gu


## Requirements

```bash
conda create --name hypergeom python=3.10 -y
conda activate hypergeom
# install torch/torchvision/torchaudio from the section below
pip install -r requirements.txt
```

### PyTorch install

Install PyTorch separately before `pip install -r requirements.txt` so the correct CUDA build is selected for your machine.

For RTX 5090 / Blackwell GPUs, use PyTorch 2.8+ with a CUDA 12.8+ build. Current stable example:

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

For older NVIDIA GPUs, you can use any compatible PyTorch build from the official selector:

https://pytorch.org/get-started/locally/

## Usage 
Signed Rate Encoding (--signed) 

#### Training CIFAR-10/100
python main_train.py --dataset cifar10  --signed

python main_train.py --dataset cifar10 --signed --attack gn --eps 8

python main_train.py --dataset cifar10 --signed --attack fgsm --eps 8 

python main_train.py --dataset cifar10 --signed --attack pgd --eps 8 --alpha 2 --steps 4

python main_train.py --dataset cifar10 --signed --attack sea --eps 5



#### Training SVHN
python main_train.py --dataset svhn --signed

python main_train.py --dataset svhn --signed --attack gn --eps 2 

python main_train.py --dataset svhn --signed --attack fgsm --eps 2

python main_train.py --dataset svhn --signed --attack pgd --eps 2 --alpha 1 --steps 2

python main_train.py --dataset svhn --signed --attack sea --eps 5      # Rate encoding does not converge with --eps 5, use --eps 2


#### Training Imagenet-100
python main_train.py --dataset imagenet100 --signed --model sewresnet 

#### Training Imagenet-1000
python main_train.py --dataset imagenet1000 --signed --model sewresnet

#### Download helpers
python scripts/download_imagenet100.py

python scripts/download_imagenet1000.py --token $HF_TOKEN

`ILSVRC/imagenet-1k` on Hugging Face is gated, so you need to accept the dataset terms and authenticate first.

#### Testing
The testing usage with hyper-parameters are supplied in run_test_{dataset}.sh

#### Log Files
Log files for models reported in paper can be found in logs directory.
