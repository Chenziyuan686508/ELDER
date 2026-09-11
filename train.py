# Adapted from Tevatron code
import json
import logging
import os
import os.path
import sys

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensures logs appear in stdout
)
logger = logging.getLogger(__name__)

import sys
import torch
import wandb
import yaml
from transformers import HfArgumentParser
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.train_collator import MultimodalDataCollator
from src.data.loader.mixed_dataset import init_mixed_dataset
from src.model.model import MMEBModel
from src.trainer import GradCacheLateProcessTrainer
from src.utils.basic_utils import print_rank, print_master, find_latest_checkpoint
from src.utils.config_utils import resolve_dataset_paths
from src.model.processor import load_processor, get_backbone_name
from src.elder.stage1 import validate_stage1_v2_dora_configuration


def main():
    # a hack for torch.distributed.launch: https://github.com/huggingface/transformers/issues/22171
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments


    # DEBUG PRINTS for Distributed Setup
    print("Distributed init debug info:")
    print(f"RANK: {os.environ.get('RANK')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")

    if torch.distributed.is_available():
        print(f"torch.distributed.is_initialized: {torch.distributed.is_initialized()}")
        if torch.distributed.is_initialized():
            print(f"torch.distributed.get_rank(): {torch.distributed.get_rank()}")
            print(f"torch.distributed.get_world_size(): {torch.distributed.get_world_size()}")


    # Check for existing checkpoints
    if training_args.resume_from == 'auto':
        resume_checkpoint_dir = find_latest_checkpoint(training_args.output_dir)
        if resume_checkpoint_dir:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    elif training_args.resume_from.isdigit():
        resume_checkpoint_dir = os.path.join(training_args.output_dir, f'checkpoint-{training_args.resume_from}')
        if os.path.exists(resume_checkpoint_dir):
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    else:
        resume_checkpoint_dir = None
        logger.info("No checkpoint found. Starting fresh training.")

    # Initialize WandB if enabled
    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):
            print_rank('init wandb')
            wandb.init(project=training_args.project_name, name=training_args.run_name, mode="online")
            wandb.config.update(model_args)
            wandb.config.update(data_args)
            wandb.config.update(training_args)

    model = MMEBModel.build(model_args)
    if model_args.strict_stage1_dora:
        audit_report = validate_stage1_v2_dora_configuration(model_args, model)
        is_world_process_zero = (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        if is_world_process_zero:
            os.makedirs(training_args.output_dir, exist_ok=True)
            audit_path = os.path.join(
                training_args.output_dir, "stage1_trainable_parameters.json"
            )
            with open(audit_path, "w", encoding="utf-8") as handle:
                json.dump(audit_report, handle, indent=2, sort_keys=True)
                handle.write("\n")
            logger.info(
                "Strict Stage 1 V2 DoRA audit passed: trainable=%s; report=%s",
                f"{audit_report['parameters']['trainable_parameters']:,}",
                audit_path,
            )
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')
    processor = load_processor(model_args, data_args)
    setattr(model, 'processor', processor)

    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_config = yaml.safe_load(yaml_file)
        resolve_dataset_paths(dataset_config, data_basedir=data_args.data_basedir)
        train_dataset = init_mixed_dataset(dataset_config, model_args, data_args, training_args)
    train_collator = MultimodalDataCollator(processor, model_args, data_args, training_args)

    trainer_cls = GradCacheLateProcessTrainer
    trainer = trainer_cls(
        model=model,
        processing_class=processor,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        data_collator=train_collator,
        max_length=data_args.max_len,
    )
    train_dataset.trainer = trainer

    trainer.train(resume_from_checkpoint=resume_checkpoint_dir)

    skip_final_save = os.environ.get("ELDER_SKIP_FINAL_SAVE", "0").lower() in {"1", "true", "yes"}
    if skip_final_save:
        if trainer.is_world_process_zero():
            logger.info("ELDER_SKIP_FINAL_SAVE is enabled; skipping final model export.")
    else:
        trainer.save_model(training_args.output_dir)
        if trainer.is_world_process_zero():
            processor.save_pretrained(training_args.output_dir)


    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
