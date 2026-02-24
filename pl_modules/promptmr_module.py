import torch
from data import transforms
from pl_modules import MriModule
from typing import List
import copy 
from mri_utils import SSIMLoss
import torch.nn.functional as F
import importlib

def get_model_class(module_name, class_name="PromptMR"):
    """
    Dynamically imports the specified module and retrieves the class.

    Args:
        module_name (str): The module to import (e.g., 'model.m1', 'model.m2').
        class_name (str): The class to retrieve from the module (default: 'PromptMR').

    Returns:
        type: The imported class.
    """
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class

class PromptMrModule(MriModule):

    def __init__(
        self,
        num_cascades: int = 12,
        num_adj_slices: int = 5,
        n_feat0: int = 48,
        feature_dim: List[int] = [72,96,120],
        prompt_dim: List[int] = [24,48,72],
        sens_n_feat0: int = 24,
        sens_feature_dim: List[int] = [36,48,60],
        sens_prompt_dim: List[int] = [12,24,36],
        len_prompt: List[int] = [5,5,5],
        prompt_size: List[int] = [64,32,16],
        n_enc_cab: List[int] = [2,3,3],
        n_dec_cab: List[int] = [2,2,3],
        n_skip_cab: List[int] = [1,1,1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        learnable_prompt: bool = False,
        adaptive_input: bool = True,
        n_buffer: int = 4,
        n_history: int = 0,
        use_sens_adj: bool = True,
        model_version: str = "promptmr_v2",
        lr: float = 0.0002,
        lr_step_size: int = 11,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.01,
        use_checkpoint: bool = False,
        compute_sens_per_coil: bool = False,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational network.
            num_adj_slices: Number of adjacent slices.
            n_feat0: Number of top-level feature channels for PromptUnet.
            feature_dim: feature dim for each level in PromptUnet.
            prompt_dim: prompt dim for each level in PromptUnet.
            sens_n_feat0: Number of top-level feature channels for sense map
                estimation PromptUnet in PromptMR.
            sens_feature_dim: feature dim for each level in PromptUnet for
                sensitivity map estimation (SME) network.
            sens_prompt_dim: prompt dim for each level in PromptUnet in
                sensitivity map estimation (SME) network.
            len_prompt: number of prompt component in each level.
            prompt_size: prompt spatial size.
            n_enc_cab: number of CABs (channel attention Blocks) in DownBlock.
            n_dec_cab: number of CABs (channel attention Blocks) in UpBlock.
            n_skip_cab: number of CABs (channel attention Blocks) in SkipBlock.
            n_bottleneck_cab: number of CABs (channel attention Blocks) in BottleneckBlock.
            no_use_ca: not using channel attention.
            learnable_prompt: whether to set the prompt as learnable parameters.
            adaptive_input: whether to use adaptive input.
            n_buffer: number of buffer in adaptive input.
            n_history: number of historical feature aggregation, should be less than num_cascades.
            use_sens_adj: whether to use adjacent sensitivity map estimation.
            model_version: model version. Default is "promptmr_v2".
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            use_checkpoint: Whether to use checkpointing to trade compute for GPU memory.
            compute_sens_per_coil: (bool) whether to compute sensitivity maps per coil for memory saving
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.num_adj_slices = num_adj_slices

        self.n_feat0 = n_feat0
        self.feature_dim = feature_dim
        self.prompt_dim = prompt_dim

        self.sens_n_feat0 = sens_n_feat0
        self.sens_feature_dim = sens_feature_dim
        self.sens_prompt_dim = sens_prompt_dim

        self.len_prompt = len_prompt
        self.prompt_size = prompt_size
        self.n_enc_cab = n_enc_cab
        self.n_dec_cab = n_dec_cab
        self.n_skip_cab = n_skip_cab
        self.n_bottleneck_cab = n_bottleneck_cab

        self.no_use_ca = no_use_ca

        self.learnable_prompt = learnable_prompt
        self.adaptive_input = adaptive_input
        self.n_buffer = n_buffer
        self.n_history = n_history
        self.use_sens_adj = use_sens_adj
        # two flags for reducing memory usage
        self.use_checkpoint = use_checkpoint
        self.compute_sens_per_coil = compute_sens_per_coil
        
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.model_version = model_version
        PromptMR = get_model_class(f"models.{model_version}")  # Dynamically get the model class
        
        self.promptmr = PromptMR(
            num_cascades=self.num_cascades,
            num_adj_slices=self.num_adj_slices,
            n_feat0=self.n_feat0,
            feature_dim = self.feature_dim,
            prompt_dim = self.prompt_dim,
            sens_n_feat0=self.sens_n_feat0,
            sens_feature_dim = self.sens_feature_dim,
            sens_prompt_dim = self.sens_prompt_dim,
            len_prompt = self.len_prompt,
            prompt_size = self.prompt_size,
            n_enc_cab = self.n_enc_cab,
            n_dec_cab = self.n_dec_cab,
            n_skip_cab = self.n_skip_cab,
            n_bottleneck_cab = self.n_bottleneck_cab,
            no_use_ca=self.no_use_ca,
            learnable_prompt = learnable_prompt,
            n_history = self.n_history,
            n_buffer = self.n_buffer,
            adaptive_input = self.adaptive_input,
            use_sens_adj = self.use_sens_adj,
        )

        self.loss = SSIMLoss()

    def configure_optimizers(self):

        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # step lr scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )
        return [optim], [scheduler]
    
    def forward(self, masked_kspace, mask, num_low_frequencies, mask_type="cartesian",
                use_checkpoint=False, compute_sens_per_coil=False, precomputed_sens_maps=None):
        return self.promptmr(masked_kspace, mask, num_low_frequencies, mask_type,
                             use_checkpoint=use_checkpoint, compute_sens_per_coil=compute_sens_per_coil,
                             precomputed_sens_maps=precomputed_sens_maps)

    def _get_precomputed_sens_maps(self, batch):
        """Extract pre-computed sensitivity maps from batch if available."""
        if hasattr(batch, 'sens_maps') and batch.sens_maps is not None:
            # Check it's not a zero-dim placeholder
            if batch.sens_maps.dim() > 1:
                return batch.sens_maps
        return None

    def training_step(self, batch, batch_idx):
        # --- DEBUG: Print batch info for first few steps ---
        if batch_idx < 3:
            print(f"\n[DEBUG train_step] batch_idx={batch_idx}")
            print(f"  masked_kspace: shape={batch.masked_kspace.shape}, "
                  f"has_nan={torch.isnan(batch.masked_kspace).any()}, "
                  f"abs_max={batch.masked_kspace.abs().max():.6e}")
            print(f"  mask: shape={batch.mask.shape}, dtype={batch.mask.dtype}")
            print(f"  target: shape={batch.target.shape}, "
                  f"has_nan={torch.isnan(batch.target).any()}, "
                  f"range=[{batch.target.min():.6e}, {batch.target.max():.6e}]")
            print(f"  max_value: {batch.max_value}")
            if hasattr(batch, 'sens_maps') and batch.sens_maps is not None and batch.sens_maps.dim() > 1:
                print(f"  sens_maps: shape={batch.sens_maps.shape}, "
                      f"has_nan={torch.isnan(batch.sens_maps).any()}, "
                      f"abs_max={batch.sens_maps.abs().max():.6e}")
            print(f"  fname={batch.fname}, slice_num={batch.slice_num}")

        precomputed_sens_maps = self._get_precomputed_sens_maps(batch)
        output_dict = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.mask_type,
                           use_checkpoint=self.use_checkpoint, compute_sens_per_coil=self.compute_sens_per_coil,
                           precomputed_sens_maps=precomputed_sens_maps)
        output = output_dict['img_pred']

        # --- DEBUG: Print output info for first few steps ---
        if batch_idx < 3:
            print(f"  output: shape={output.shape}, "
                  f"has_nan={torch.isnan(output).any()}, "
                  f"range=[{output.min():.6e}, {output.max():.6e}]")

        target, output = transforms.center_crop_to_smallest(
            batch.target, output)

        loss = self.loss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
        )

        # --- DEBUG: Print loss info for first few steps ---
        if batch_idx < 3:
            print(f"  loss={loss.item():.6e}")

        self.log("train_loss", loss, prog_bar=True)

        ##! raise error if loss is nan
        if torch.isnan(loss):
            print(f"\n[DEBUG NaN DETECTED] batch_idx={batch_idx}")
            print(f"  fname={batch.fname}, slice_num={batch.slice_num}")
            print(f"  target: has_nan={torch.isnan(batch.target).any()}, range=[{batch.target.min():.6e}, {batch.target.max():.6e}]")
            print(f"  output: has_nan={torch.isnan(output).any()}, range=[{output.min():.6e}, {output.max():.6e}]")
            print(f"  max_value={batch.max_value}")
            raise ValueError(f'nan loss on {batch.fname} of slice {batch.slice_num}')
        return loss

    def on_after_backward(self):
        if self.global_step % self.trainer.log_every_n_steps ==0:
            grad_norm = torch.nn.utils.get_total_norm(
                [p.grad for p in self.promptmr.parameters() if p.grad is not None]
            )
            self.log("grad_norm", grad_norm)

    def validation_step(self, batch, batch_idx):
        precomputed_sens_maps = self._get_precomputed_sens_maps(batch)
        output_dict = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.mask_type,
                           compute_sens_per_coil=self.compute_sens_per_coil,
                           precomputed_sens_maps=precomputed_sens_maps)
        output = output_dict['img_pred']
        img_zf = output_dict['img_zf']
        target, output = transforms.center_crop_to_smallest(
            batch.target, output)
        _, img_zf = transforms.center_crop_to_smallest(
            batch.target, img_zf)
        val_loss = self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
            )
        cc = batch.masked_kspace.shape[1]
        centered_coil_ksp_visual = torch.log10(1e-10+torch.view_as_complex(batch.masked_kspace[:,cc//2]).abs())
        centered_sens_maps_visual = output_dict['sens_maps'][:,cc//self.num_adj_slices//2].abs()
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "img_zf":   img_zf,
            "mask": centered_coil_ksp_visual,
            "sens_maps": centered_sens_maps_visual,
            "output": output,
            "target": target,
            "loss": val_loss,
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        precomputed_sens_maps = self._get_precomputed_sens_maps(batch)
        output_dict = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.mask_type,
                           compute_sens_per_coil=self.compute_sens_per_coil,
                           precomputed_sens_maps=precomputed_sens_maps)
        output = output_dict['img_pred']

        crop_size = batch.crop_size
        crop_size = [crop_size[0][0], crop_size[1][0]] # if batch_size>1
        # detect FLAIR 203
        if output.shape[-1] < crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        output = transforms.center_crop(output, crop_size)

        num_slc = batch.num_slc
        return {
            'output': output.cpu(),
            'slice_num': batch.slice_num,
            'fname': batch.fname,
            'num_slc':  num_slc
        }
        