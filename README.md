# PE-amplfi

This repository contains scripts for distillation, pruning, and quantization of neural network models, specifically using ResNet architectures. The primary focus is to accelerate multi-messenger parameter estimation using Likelihood-free Inference (LFI).

## Environment Information

The following environment setup is used for running the scripts:

- **Framework**: PyTorch 2.1.0
- **Python Version**: Python 3.10 (Ubuntu 22.04)
- **CUDA Version**: Cuda 12.1
- **GPU**: A800-80GB (80GB) x 1
- **CPU**: 14 vCPU Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz

## Scripts Overview

### dis_train.py
This script distills an embedding network (ResNet) using a fully trained flow model.

- **Customization**:
  - You can change the layer configuration of the distilled ResNet as shown below:
    ```python
    dis_resnet = ResNet(
        (n_ifos, strain_dim),
        context_dim=resnet_context_dim,
        layers=[1, 1],  # Specify the layers you want
        norm_groups=resnet_norm_groups,
    )
    ```
- **Output**:
  - All distilled models will be saved to the `distillation` directory.

### distillation.py
This script distills an embedding network (ResNet) using a trained ResNet model.

- **Customization**:
  - You can change the layer configuration of the distilled ResNet as shown below:
    ```python
    student_resnet = ResNet(
        (n_ifos, strain_dim),
        context_dim=resnet_context_dim,
        layers=[3, 3],  # Specify the layers you want
        norm_groups=resnet_norm_groups,
    )
    ```
- **Output**:
  - All distilled models will be saved to the `distillation` directory.

### tesing_dis.py
This script tests a distilled model, currently tailored for a model distilled using a fully trained flow model.

- **Customization**:
  - Change the path of the distilled model in the script:
    ```python
    ckpt_path = "./distillation/lightning_logs/version_17/checkpoints/epoch=170-step=34200.ckpt"
    ```
- **Note**: Adjust the code as needed or write a new testing script if required.

### pruning.py
This script prunes a trained model.

- **Customization**:
  - You can change the percentage of weights to be pruned:
    ```python
    flow_obj.embedding_net = prune_model(flow_obj.embedding_net, amount=0.1717)  # Change the amount here
    ```
  - The `amount` represents the total percentage of weights to be pruned based on their absolute values.

### ptdq.py
This script performs Post Training Dynamic Quantization on the model.

- **Output**:
  - Quantized models are saved to the base directory.

### ptsq.py
This script performs Post Training Static Quantization on the model.

- **Output**:
  - Quantized models are saved to the base directory.

### qat.py
This script performs Quantization Aware Training on the model.

- **Output**:
  - Quantized models are saved to the base directory.

### test_quantized_model.py
This script tests a quantized model.

## Usage
1. **Distillation**:
   - Use `dis_train.py` or `distillation.py` to distill your ResNet models. Adjust the layer configuration as needed.
   - Distilled models will be saved in the `distillation` directory.

2. **Pruning**:
   - Run `pruning.py` to prune the weights of your trained model. Adjust the pruning percentage in the script.

3. **Quantization**:
   The Quantization is still in experiment, the result may be misleading.
   - Use `ptdq.py` for Post Training Dynamic Quantization.
   - Use `ptsq.py` for Post Training Static Quantization.
   - Use `qat.py` for Quantization Aware Training.
   - Quantized models are saved in the base directory.

4. **Testing**:
   - Use `tesing_dis.py` to test distilled models. Adjust the checkpoint path in the script.
   - Use `test_quantized_model.py` to test quantized models.

## Environment Variable
To use the provided scripts, you need to set the environment variable `DATA_DIR` to the base directory containing your data. For example:
```python
background_path = os.getenv('DATA_DIR') + "/background.h5"
ckpt_path = os.getenv("BASE_DIR") + "/pl-logdir/phenomd-60-transforms-4-4-resnet-wider-dl/version_51/checkpoints/epoch=323-step=64800.ckpt"
```
Alternatively, you can hard code the path directly in the script to ignore the environment variable:
```python
background_path = "/path/to/your/base/directory/background.h5"
ckpt_path = "/path/to/your/base/directory/epoch=323-step=64800.ckpt"
```
Ensure to adjust paths and parameters within the scripts as needed for your specific use case.
