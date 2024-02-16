import torch
torch.set_float32_matmul_precision('high')

def base_test(trainer, dm, lit_mod, ckpt_path):
    trainer.test(lit_mod, datamodule=dm, ckpt_path=ckpt_path)
