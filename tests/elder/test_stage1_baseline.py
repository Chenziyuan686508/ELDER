import math
from types import MethodType, SimpleNamespace

import torch

from src.data.collator.train_collator import MultimodalDataCollator
from src.loss import SimpleContrastiveLoss
from src.model.model import MMEBModel
from src.model.processor import process_vlm_inputs_fns
from src.trainer import (
    HYBRID_CHECKPOINT_METADATA,
    GradCacheLateProcessTrainer,
    _save_nested_peft_adapter,
)
from src.utils.config_utils import resolve_dataset_paths


def test_dataset_path_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("ELDER_DATA_ROOT", "colpali")
    config = {
        "debug": {
            "dataset_parser": "vidore",
            "dataset_path": "${ELDER_DATA_ROOT}/train.parquet",
            "weight": 1,
        }
    }

    resolved = resolve_dataset_paths(config, data_basedir=str(tmp_path))

    assert resolved["debug"]["dataset_path"] == str(
        tmp_path / "colpali" / "train.parquet"
    )


def test_contrastive_loss_backpropagates():
    query = (torch.eye(4) * 0.1).requires_grad_()
    positive = torch.eye(4)

    loss = SimpleContrastiveLoss(temperature=0.02)(query, positive)
    loss.backward()

    assert math.isfinite(loss.item())
    assert query.grad is not None
    assert query.grad.norm().item() > 0


def test_last_pooling_uses_last_non_padding_token_and_normalizes():
    encoder = torch.nn.Linear(2, 2, bias=False)
    encoder.config = type("Config", (), {"hidden_size": 2})()
    model = MMEBModel(
        encoder=encoder,
        pooling="last",
        normalize=True,
        temperature=0.02,
    )
    hidden_states = torch.tensor(
        [
            [[3.0, 4.0], [0.0, 2.0], [9.0, 9.0]],
            [[8.0, 8.0], [5.0, 12.0], [0.0, 0.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0], [0, 1, 0]])

    pooled = model._pooling(hidden_states, attention_mask)

    expected = torch.tensor([[0.0, 1.0], [5.0 / 13.0, 12.0 / 13.0]])
    torch.testing.assert_close(pooled, expected)


def test_training_collator_only_processes_returned_pair_views(monkeypatch):
    collator = MultimodalDataCollator(
        processor=None,
        model_args=SimpleNamespace(),
        data_args=SimpleNamespace(max_len=None),
        training_args=SimpleNamespace(model_backbone="elder_test"),
    )
    calls = []

    def fake_get_batch_inputs(self, examples, text_keyname, image_keyname):
        calls.append((text_keyname, image_keyname))
        return {"text": ["example"]}

    def fake_process(inputs, processor, max_length):
        return {
            "input_ids": torch.ones(1, 1, dtype=torch.long),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
        }

    monkeypatch.setitem(process_vlm_inputs_fns, "elder_test", fake_process)
    collator._get_batch_inputs = MethodType(fake_get_batch_inputs, collator)

    query, positive = collator(
        [
            {
                "query_text": "query",
                "pos_text": "positive",
                "global_dataset_name": "elder/test",
            }
        ]
    )

    assert query["text"] == ["query"]
    assert positive["text"] == ["positive"]
    assert calls == [
        ("query_text", "query_image"),
        ("pos_text", "pos_image"),
    ]


def test_grad_cache_is_used_on_single_device_when_enabled():
    class DummyGradCache:
        def __init__(self):
            self.models = None
            self.call = None

        def __call__(self, query, target, no_sync_except_last):
            self.call = (query, target, no_sync_except_last)
            return torch.tensor(2.0)

    model = torch.nn.Linear(2, 2)
    trainer = object.__new__(GradCacheLateProcessTrainer)
    trainer.args = SimpleNamespace(grad_cache=True)
    trainer.gc = DummyGradCache()
    trainer._dist_loss_scale_factor = 1
    query = {"input_ids": torch.ones(1, 2)}
    target = {"input_ids": torch.ones(1, 2)}

    loss = trainer.training_step(model, (query, target))

    assert loss.item() == 2.0
    assert trainer.gc.models == [model, model]
    assert trainer.gc.call is not None
    assert trainer.gc.call[2] is False


def test_nested_peft_checkpoint_writes_resume_metadata(tmp_path):
    class DummyPeftModel:
        peft_config = {"default": object()}

        def save_pretrained(self, output_dir, safe_serialization):
            assert safe_serialization is True

    encoder = SimpleNamespace(model=DummyPeftModel())

    _save_nested_peft_adapter(encoder, str(tmp_path))

    metadata_path = tmp_path / HYBRID_CHECKPOINT_METADATA
    assert metadata_path.is_file()
    assert '"adapter_scope": "encoder.model"' in metadata_path.read_text()
