"""
Description: This script is the main entry point for the LightningCLI.
"""

import os
import sys
from itertools import chain
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict

import yaml
import torch
import numpy as np
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import BasePredictionWriter, Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from mri_utils import save_reconstructions, save_reconstructions_npy

@rank_zero_only
def print_on_rank0(*args, **kwargs):
    print(*args, **kwargs)

def preprocess_save_dir():
    """Ensure `save_dir` exists, handling both command-line arguments and YAML configuration."""
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, nargs="*",
                        help="Path(s) to YAML config file(s)")
    parser.add_argument("--trainer.logger.save_dir",
                        type=str, help="Logger save directory")
    args, _ = parser.parse_known_args(sys.argv[1:])

    save_dir = None  # Default to None

    if args.config:
        for config_path in args.config:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding='utf-8') as f:
                    try:
                        config = yaml.safe_load(f)
                        if config is not None:
                            # Safely navigate to trainer.logger.save_dir
                            trainer = config.get("trainer", {})
                            logger = trainer.get("logger", {})
                            if isinstance(logger, dict) :  # Ensure logger is a dictionary
                                yaml_save_dir = logger.get(
                                    "init_args", {}).get("save_dir")
                                if yaml_save_dir:
                                    save_dir = yaml_save_dir  # Use the first valid save_dir found
                                    break
                    except yaml.YAMLError as e:
                        print(f"Error parsing YAML file {config_path}: {e}")

    for i, arg in enumerate(sys.argv):
        if arg == "--trainer.logger.save_dir":
            save_dir = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
            break

    if not save_dir:
        print("Logger save_dir is None. No action taken.")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Pre-created logger save_dir: {save_dir}")

class ChangeLRStepSizeCallback(Callback):
    """
    Change the step size and gamma in lr_schedulers during training.
    """
    def __init__(self, step_size=None, lr_gamma=None):
        self.step_size = step_size
        self.lr_gamma = lr_gamma

    def on_train_start(self, trainer, pl_module):
        if self.step_size is not None:
            lr_before = pl_module.lr_schedulers().get_last_lr()
            step_size_before = pl_module.lr_schedulers().step_size
            lr_gamma_before = pl_module.lr_schedulers().gamma

            pl_module.lr_schedulers().step_size = self.step_size
            if self.lr_gamma is not None:
                pl_module.lr_schedulers().gamma = self.lr_gamma

            pl_module.lr_schedulers().last_epoch = pl_module.current_epoch-1
            pl_module.lr_schedulers().step()

            lr_after = pl_module.lr_schedulers().get_last_lr()
            step_size_after = pl_module.lr_schedulers().step_size
            lr_gamma_after = pl_module.lr_schedulers().gamma

            print_on_rank0(f'ChangeLRStepSizeCallback: step_size before: {step_size_before}, step_size after: {step_size_after}')
            print_on_rank0(f'ChangeLRStepSizeCallback: lr_gamma before: {lr_gamma_before}, lr_gamma after: {lr_gamma_after}')
            print_on_rank0(f'ChangeLRStepSizeCallback: lr before: {lr_before}, lr after: {lr_after}')

class SkipCurrentEpochOnResume(Callback):
    """
    If resuming mid-epoch, skip all remaining batches so we start the NEXT epoch.
    """
    def __init__(self, only_if_resumed=True, trigger_if_batch_ge=0, verbose=True):
        self.only_if_resumed = only_if_resumed
        self.trigger_if_batch_ge = trigger_if_batch_ge
        self.verbose = verbose
        self._loaded_from_ckpt = False
        self._done = False

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        # Called when actually resuming from a checkpoint
        self._loaded_from_ckpt = True

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._done:
            return
        if self.only_if_resumed and not self._loaded_from_ckpt:
            return
        if batch_idx >= self.trigger_if_batch_ge:
            self._done = True
            if self.verbose:
                print(f"[SKIP] Ending epoch {trainer.current_epoch} at batch {batch_idx} -> starting next epoch.")
            # Force exit from current epoch's batch loop
            raise StopIteration
        
class CustomSaveConfigCallback(SaveConfigCallback):
    '''save the config file to the logger's run directory, merge tags from different configs'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merged_tags = self._collect_tags_from_configs()

    def _collect_tags_from_configs(self):
        config_files = []
        merged_tags = set()

        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                config_files.append(sys.argv[i + 1])

        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                        if isinstance(config_data, dict):
                            logger = config_data.get('trainer', {}).get(
                                'logger', {})
                            if logger and isinstance(logger, dict):
                                tags = logger.get('init_args', {}).get('tags', [])
                                if isinstance(tags, list):
                                    merged_tags.update(tags)
                except (yaml.YAMLError, IOError) as e:
                    print(f"Warning: Error reading {config_file}: {str(e)}")
        return merged_tags

    def setup(self, trainer, pl_module, stage):
        if hasattr(self.config, 'trainer') and hasattr(self.config.trainer, 'logger'):
            logger_config = self.config.trainer.logger
            if hasattr(logger_config, 'init_args'):
                logger_config.init_args['tags'] = list(self.merged_tags)
                if hasattr(trainer, 'logger') and trainer.logger is not None:
                    trainer.logger.experiment.tags = list(self.merged_tags)

        super().setup(trainer, pl_module, stage)

    def save_config(self, trainer, pl_module, stage) -> None:
        """Save the configuration file under the logger's run directory."""
        if stage == "predict":
            print("Skipping saving configuration in predict mode.")
            return  
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            project_name = trainer.logger.experiment.project_name()
            run_id = trainer.logger.experiment.id
            save_dir = trainer.logger.save_dir
            run_dir = os.path.join(save_dir, project_name, run_id)
            
            os.makedirs(run_dir, exist_ok=True)
            config_path = os.path.join(run_dir, "config.yaml")
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
            print(f"Configuration saved to {config_path}")


class CustomWriter(BasePredictionWriter):
    """
    A custom prediction writer to save reconstructions to disk.
    """

    def __init__(self, output_dir: Path, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.outputs = defaultdict(list)
        
    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        """
        Collect predictions batch by batch and organize them by volume.
        Assumes `predictions` contains a dictionary with 'volume_id' and 'slice_prediction'.
        """
        pass
        
    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):

        gathered = [None] * torch.distributed.get_world_size()
        gathered_indices = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, predictions)
        torch.distributed.all_gather_object(gathered_indices, batch_indices)
        torch.distributed.barrier()
        if not trainer.is_global_zero:
            return
        predictions = sum(gathered, [])
        batch_indices = sum(gathered_indices, [])
        batch_indices = list(chain.from_iterable(batch_indices))
        outputs = defaultdict(list)
        outputs_complex = defaultdict(list)
        num_slc_dict = {} # for reshape
        # Iterate through batches
        for batch_predictions in predictions:
            for i in range(len(batch_predictions["fname"])):
                fname = batch_predictions["fname"][i]
                slice_num = int(batch_predictions["slice_num"][i])
                output = batch_predictions["output"][i:i+1]
                outputs[fname].append((slice_num, output))
                output_complex = batch_predictions["output_complex"][i:i+1]
                outputs_complex[fname].append((slice_num, output_complex))
                # if num_slc_list[fname] exist, assign
                num_slc = batch_predictions["num_slc"][i].numpy()
                if fname not in num_slc_dict and num_slc!=-1:
                    num_slc_dict[fname] = batch_predictions["num_slc"][i]

        # Sort slices and stack them into volumes
        for fname in outputs:
            outputs[fname] = np.concatenate(
                [out.cpu() for _, out in sorted(outputs[fname])])
            outputs_complex[fname] = np.concatenate(
                [out.cpu() for _, out in sorted(outputs_complex[fname])])

        # Save the reconstructions
        save_reconstructions(outputs, num_slc_dict, self.output_dir / "reconstructions")
        save_reconstructions_npy(outputs_complex, self.output_dir / "reconstructions_complex")
        print(f"Done! Reconstructions saved to {self.output_dir / 'reconstructions'}")
        print(f"Complex predictions saved to {self.output_dir / 'reconstructions_complex'}")


class NpyCustomWriter(BasePredictionWriter):
    """
    A custom prediction writer to save reconstructions as .npy files.
    Used for cine MRI data with precomputed masks and sensitivity maps.
    """

    def __init__(self, output_dir: Path, write_interval):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        pass

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        gathered = [None] * torch.distributed.get_world_size()
        gathered_indices = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, predictions)
        torch.distributed.all_gather_object(gathered_indices, batch_indices)
        torch.distributed.barrier()
        if not trainer.is_global_zero:
            return
        predictions = sum(gathered, [])
        batch_indices = sum(gathered_indices, [])
        batch_indices = list(chain.from_iterable(batch_indices))
        outputs = defaultdict(list)
        outputs_complex = defaultdict(list)
        # Iterate through batches
        for batch_predictions in predictions:
            for i in range(len(batch_predictions["fname"])):
                fname = batch_predictions["fname"][i]
                slice_num = int(batch_predictions["slice_num"][i])
                output = batch_predictions["output"][i:i+1]
                outputs[fname].append((slice_num, output))
                output_complex = batch_predictions["output_complex"][i:i+1]
                outputs_complex[fname].append((slice_num, output_complex))

        # Sort slices and stack them into volumes (sorted by temporal frame)
        for fname in outputs:
            outputs[fname] = np.concatenate(
                [out.cpu() for _, out in sorted(outputs[fname])])
            outputs_complex[fname] = np.concatenate(
                [out.cpu() for _, out in sorted(outputs_complex[fname])])

        # Save as .npy
        save_reconstructions_npy(outputs, self.output_dir / "reconstructions")
        save_reconstructions_npy(outputs_complex, self.output_dir / "reconstructions_complex")
        print(f"Done! Reconstructions saved to {self.output_dir / 'reconstructions'}")
        print(f"Complex predictions saved to {self.output_dir / 'reconstructions_complex'}")


def get_model_init_arg_overrides():
    """Returns a dict of model.init_args overrides set on the CLI only."""
    overrides = {}
    prefix = "--model.init_args."
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg.startswith(prefix):
            key = arg[len(prefix):]
            if i + 1 < len(argv):
                value = argv[i + 1]
                # Basic type conversion
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                overrides[key] = value
    return overrides

class CustomLightningCLI(LightningCLI):

    def instantiate_classes(self):
        super().instantiate_classes()
        if self.config_init.subcommand == 'predict':
            model_class = self.config_init.predict.model.__class__
            init_args = get_model_init_arg_overrides()
            self.model = model_class.load_from_checkpoint(self.config_init.predict.ckpt_path, **init_args)


def run_cli():

    preprocess_save_dir()

    cli = CustomLightningCLI(
        save_config_callback=CustomSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
    )

if __name__ == "__main__":
    run_cli()
