import math
from types import MethodType, SimpleNamespace

import torch

from src.data.collator.train_collator import MultimodalDataCollator
from src.elder.stage1 import (
    audit_stage1_v2_trainable_parameters,
    compare_stage1_reports,
    compute_retrieval_metrics,
    validate_stage1_v2_dora_configuration,
    validate_qwen2_vl_2b_config,
)
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


def test_fixed_qwen2_vl_2b_contract():
    contract = validate_qwen2_vl_2b_config(
        {
            "model_type": "qwen2_vl",
            "architectures": ["Qwen2VLForConditionalGeneration"],
            "hidden_size": 1536,
            "num_hidden_layers": 28,
        }
    )

    assert contract["model_id"] == "Qwen/Qwen2-VL-2B-Instruct"


def test_fixed_qwen2_vl_2b_contract_rejects_other_size():
    try:
        validate_qwen2_vl_2b_config(
            {
                "model_type": "qwen2_vl",
                "architectures": ["Qwen2VLForConditionalGeneration"],
                "hidden_size": 3584,
                "num_hidden_layers": 28,
            }
        )
    except ValueError as error:
        assert "Qwen/Qwen2-VL-2B-Instruct" in str(error)
    else:
        raise AssertionError("A non-2B config should be rejected.")


class _FakeDoraLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.ParameterDict(
            {"default": torch.nn.Parameter(torch.ones(1))}
        )
        self.lora_B = torch.nn.ParameterDict(
            {"default": torch.nn.Parameter(torch.ones(1))}
        )
        self.lora_magnitude_vector = torch.nn.ParameterDict(
            {"default": torch.nn.Parameter(torch.ones(1))}
        )


class _FakeDoraModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Linear(1, 1)
        self.visual.requires_grad_(False)
        block = torch.nn.Module()
        for module_type in ("q_proj", "k_proj", "v_proj", "o_proj", "down_proj"):
            setattr(block, module_type, _FakeDoraLinear())
        language_model = torch.nn.Module()
        language_model.layers = torch.nn.ModuleList([block])
        self.model = language_model


def test_stage1_v2_trainable_parameter_audit_accepts_only_original_dora_targets():
    model = _FakeDoraModel()
    report = audit_stage1_v2_trainable_parameters(
        model, expected_trainable_parameters=None
    )

    assert report["status"] == "passed"
    assert report["checks"]["visual_tower_frozen"]
    assert report["observed_module_types"] == [
        "down_proj",
        "k_proj",
        "o_proj",
        "q_proj",
        "v_proj",
    ]


def test_stage1_v2_configuration_rejects_elder_alpha_32():
    model_args = SimpleNamespace(
        lora=True,
        lora_adapter_scope="full_model",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj,down_proj",
    )

    try:
        validate_stage1_v2_dora_configuration(model_args, _FakeDoraModel())
    except ValueError as error:
        assert "alpha_is_64" in str(error)
    else:
        raise AssertionError("Strict V2 audit should reject alpha=32.")


def test_stage1_retrieval_metrics_and_acceptance():
    embeddings = torch.eye(4)
    after_metrics = compute_retrieval_metrics(embeddings, embeddings)
    before_metrics = compute_retrieval_metrics(
        embeddings,
        embeddings.flip(0),
    )

    assert after_metrics["query_to_candidate"]["recall_at"]["1"] == 1.0
    result = compare_stage1_reports(
        {"metrics": before_metrics},
        {"metrics": after_metrics},
    )
    assert result["status"] == "passed"
    assert all(result["checks"].values())


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

    encoder = SimpleNamespace(
        model=DummyPeftModel(),
        config=SimpleNamespace(
            model_type="qwen2_vl",
            architectures=["Qwen2VLForConditionalGeneration"],
            hidden_size=1536,
            num_hidden_layers=28,
        ),
    )

    _save_nested_peft_adapter(encoder, str(tmp_path))

    metadata_path = tmp_path / HYBRID_CHECKPOINT_METADATA
    assert metadata_path.is_file()
    metadata = metadata_path.read_text()
    assert '"adapter_scope": "encoder.model"' in metadata
    assert '"model_type": "qwen2_vl"' in metadata
    assert '"hidden_size": 1536' in metadata


def test_full_model_peft_checkpoint_does_not_write_hybrid_metadata(tmp_path):
    class InnerModel:
        peft_config = {"default": object()}

        def save_pretrained(self, *args, **kwargs):
            raise AssertionError("Nested saver must not run for full-model PEFT.")

    encoder = SimpleNamespace(
        peft_config={"default": object()},
        model=InnerModel(),
    )

    _save_nested_peft_adapter(encoder, str(tmp_path))

    assert not (tmp_path / HYBRID_CHECKPOINT_METADATA).exists()


def test_strict_stage1_resume_reaudits_loaded_checkpoint(monkeypatch, tmp_path):
    import src.trainer as trainer_module

    loaded_model = object()
    calls = []
    monkeypatch.setattr(
        trainer_module.MMEBModel,
        "load",
        lambda model_args: loaded_model,
    )

    def fake_validate(model_args, model):
        calls.append((model_args, model))
        return {"parameters": {"trainable_parameters": 9_203_712}}

    monkeypatch.setattr(
        trainer_module, "validate_stage1_v2_dora_configuration", fake_validate
    )
    trainer = object.__new__(trainer_module.MMEBTrainer)
    trainer.model_args = SimpleNamespace(
        checkpoint_path=None,
        strict_stage1_dora=True,
    )
