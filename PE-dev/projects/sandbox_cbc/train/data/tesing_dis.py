import os

import torch
from data import SignalDataSet
from lightning.pytorch import Trainer, callbacks, loggers
from lightning.pytorch.cli import LightningCLI
from lightning import pytorch as pl
from ml4gw.waveforms import IMRPhenomD, TaylorF2
from mlpe.architectures.embeddings import ResNet, QuantizedResNet
from mlpe.architectures.embeddings import DenseNet1D
from ml4gw.nn.resnet import ResNet1D
from mlpe.architectures.flows import MaskedAutoRegressiveFlow
from mlpe.injection.priors import nonspin_bbh_component_mass, nonspin_bbh_chirp_mass_q, nonspin_bbh_component_mass_parameter_sampler, nonspin_bbh_chirp_mass_q_parameter_sampler
from mlpe.logging import configure_logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import time

def prune_model(model, amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

class DistillatedResNet(pl.LightningModule):
    def __init__(self, student_embedding_net, teacher_embedding_net, optimizer, scheduler):
        super(DistillatedResNet, self).__init__()
        self.student_embedding_net = student_embedding_net
        self.teacher_embedding_net = teacher_embedding_net
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        for param in self.teacher_embedding_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.student_embedding_net(x)

    def training_step(self, batch, batch_idx):
        strain, _ = batch
        
        student_embedding = self.student_embedding_net(strain)
        with torch.no_grad():
            teacher_embedding = self.teacher_embedding_net(strain)

        # Calculate distillation loss (e.g., MSE loss)
        distillation_loss = F.mse_loss(student_embedding, teacher_embedding)
        
        self.log("train_loss", distillation_loss, on_step=True, prog_bar=True, sync_dist=False)
        return distillation_loss
        
    def validation_step(self, batch, batch_idx):
        strain, _ = batch
        
        student_embedding = self.student_embedding_net(strain)
        with torch.no_grad():
            teacher_embedding = self.teacher_embedding_net(strain)

        # Calculate distillation loss (e.g., MSE loss)
        distillation_loss = F.mse_loss(student_embedding, teacher_embedding)
        
        self.log("valid_loss", distillation_loss, on_step=True, prog_bar=True, sync_dist=False)

    def configure_optimizers(self):
        opt = self.optimizer(self.parameters(), lr= 5e-5, weight_decay=0.001)
        sched = self.scheduler(opt)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "valid_loss"}}

def main():
    
    background_path = os.getenv('DATA_DIR') + "/background.h5"
    ifos = ["H1", "L1"]
    batch_size = 500
    batches_per_epoch = 200
    sample_rate = 2048
    time_duration = 4
    f_max = 200
    f_min = 20
    f_ref = 40
    highpass = 25
    valid_frac = 0.2
    learning_rate = 1e-3

    densenet_context_dim = 100
    densenet_growth_rate = 32
    densenet_block_config = [6, 6]
    densenet_drop_rate = 0
    densenet_norm_groups = 8

    resnet_context_dim = 100
    resnet_layers = [4, 4]
    resnet_norm_groups = 8

    inference_params = [
        "chirp_mass",
        "mass_ratio",
        "luminosity_distance",
        "phase",
        "theta_jn",
        "dec",
        "psi",
        "phi",
    ]
    num_transforms = 60
    num_blocks = 5
    hidden_features = 120

    optimizer = torch.optim.AdamW
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    
    param_dim, n_ifos, strain_dim = (
        len(inference_params),
        len(ifos),
        int(sample_rate * time_duration),
    )

    embedding = ResNet(
        (n_ifos, strain_dim),
        context_dim=resnet_context_dim,
        layers=[2, 1],
        norm_groups=resnet_norm_groups,
    )

    #prior_func = nonspin_bbh_component_mass_parameter_sampler
    prior_func = nonspin_bbh_chirp_mass_q_parameter_sampler

    flow_obj = MaskedAutoRegressiveFlow(
        (param_dim, n_ifos, strain_dim),
        embedding,
        optimizer,
        scheduler,
        inference_params,
        prior_func,
        num_transforms=num_transforms,
        num_blocks=num_blocks,
        hidden_features=hidden_features
    )

    # ckpt_path = os.getenv("BASE_DIR") + "/pl-logdir/phenomd-60-transforms-4-4-resnet-wider-dl/version_51/checkpoints/epoch=323-step=64800.ckpt"
    # ckpt_path = os.getenv("BASE_DIR") + "/pl-logdir/phenomd-60-transforms-4-4-resnet-wider-dl/version_838/checkpoints/epoch=249-step=50000.ckpt"
    # ckpt_path = "./distillation/lightning_logs/version_13/checkpoints/epoch=143-step=28800.ckpt"
    # ckpt_path = "./distillation/lightning_logs/version_15/checkpoints/epoch=214-step=43000.ckpt"
    ckpt_path = "./distillation/lightning_logs/version_17/checkpoints/epoch=170-step=34200.ckpt"
    # ckpt_path = "./distillation/lightning_logs/version_28/checkpoints/epoch=223-step=44800.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    unwanted_keys = [key for key in checkpoint['state_dict'].keys() if key.startswith('teacher_flow.')]
    for key in unwanted_keys:
        del checkpoint['state_dict'][key]
    flow_obj.load_state_dict(checkpoint['state_dict'])
    
    # data
    sig_dat = SignalDataSet(
        background_path,
        ifos,
        valid_frac,
        batch_size,
        batches_per_epoch,
        sample_rate,
        time_duration,
        f_min,
        f_max,
        f_ref,
        prior_func=prior_func,
        approximant=IMRPhenomD,
    )
    
    print("##### Initialized data loader, calling setup ####")
    sig_dat.setup(None)

    print("##### Dataloader initialized #####")
    torch.set_float32_matmul_precision("high")
    early_stop_cb = callbacks.EarlyStopping(
        "valid_loss", patience=20, check_finite=True, verbose=True
    )
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="epoch")
    logger = loggers.CSVLogger(save_dir="distillation")
    #logger = loggers.CSVLogger(save_dir=outdir + "/pl-logdir", name="phenomd-50-transforms-2-2-resnet")
    # print("##### Initializing trainer #####")
    trainer = Trainer(
        max_epochs=1000,
        log_every_n_steps=100,
        callbacks=[early_stop_cb, lr_monitor],
        logger=logger,
        gradient_clip_val=10.0,
    )

    # trainer.fit(model=flow_obj, datamodule=sig_dat)
    # flow_obj.embedding_net = prune_model(flow_obj.embedding_net, amount=0.03)
    trainer.test(model=flow_obj, datamodule=sig_dat, ckpt_path=None)



    ### speed test ###

    # data_loader = sig_dat.test_dataloader()
    # embedding = flow_obj.embedding_net
    # print(embedding)
    # start = time.time()
    # with torch.no_grad():
    #     for batch in data_loader:
    #         inputs, _ = batch
    #         inputs = inputs.to('cpu')
    #         embedding(inputs)  
    # print(f"time taken is {time.time() - start}")
if __name__ == '__main__':
    main()

