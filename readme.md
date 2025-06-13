# GitHub Repository for Diffusion Classifier Guidance for Non-robust Classifiers

## Conda environment
```
conda env create -f environment.yml
conda activate nrcg
```

## Training models
```
python3 classifier_training.py --dataset {celeba/sportballs/celebahq} --robustclassifier {true/false}
python3 diffusion_training.py --dataset {celeba/sportballs/celebahq}
```

## Comparison of classifier accuracy and robustness
```
python3 classifier_acc_comparison.py --dataset {celeba/sportballs/celebahq}
python3 classifier_robustness_comparison.py --dataset {celeba/sportballs/celebahq}
```

## FID calculation for data sets
### CelebA (unconditional FID, conditional FID)
```
python3 fid_comparison_torcheval.py --dataset celeba --image_size 64
python3 fid_comparison_torcheval.py --dataset celeba --image_size 64 --class_idx 20
```
### SportBalls (unconditional FID, conditional FID)
```
python3 fid_comparison_torcheval.py --dataset sportballs --image_size 64
python3 fid_comparison_torcheval.py --dataset sportballs --image_size 64 --class_idx 0
```
### CelebA-HQ (unconditional FID, conditional FID)
```
python3 fid_comparison_torcheval.py --dataset celebahq --image_size 256 
python3 fid_comparison_torcheval.py --dataset celebahq --image_size 256  --class_idx 20
```

## Sampling from the models
```
python3 sampling.py --dataset {celeba/sportballs/celebahq} --guidance_scale {...} --guidance_stabilization {none/adam/ema09/ema099}
optional:
--robust_classifier (changes to robust classifier)
--xzeroprediction (adds xzero-prediction before classifier guidance)
```
