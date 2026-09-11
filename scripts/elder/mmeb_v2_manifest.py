"""Canonical MMEB-V2 task and local metadata manifest.

The task lists are pinned to the 78 tasks reported by
``experiments/public/all_scores/v2.0.0/VLM2Vec-V2.0-Qwen2VL-2B.json``.
They are shared by the metadata preparation and readiness-check scripts so
that download, validation, and evaluation use exactly the same benchmark
scope.
"""

from __future__ import annotations


IMAGE_TASKS = (
    "ImageNet-1K",
    "N24News",
    "HatefulMemes",
    "VOC2007",
    "SUN397",
    "Place365",
    "ImageNet-A",
    "ImageNet-R",
    "ObjectNet",
    "Country211",
    "OK-VQA",
    "A-OKVQA",
    "DocVQA",
    "InfographicsVQA",
    "ChartQA",
    "Visual7W",
    "ScienceQA",
    "VizWiz",
    "GQA",
    "TextVQA",
    "VisDial",
    "CIRR",
    "VisualNews_t2i",
    "VisualNews_i2t",
    "MSCOCO_t2i",
    "MSCOCO_i2t",
    "NIGHTS",
    "WebQA",
    "FashionIQ",
    "Wiki-SS-NQ",
    "OVEN",
    "EDIS",
    "MSCOCO",
    "RefCOCO",
    "RefCOCO-Matching",
    "Visual7W-Pointing",
)


VIDEO_TASKS = (
    "K700",
    "SmthSmthV2",
    "HMDB51",
    "UCF101",
    "Breakfast",
    "MVBench",
    "Video-MME",
    "NExTQA",
    "EgoSchema",
    "ActivityNetQA",
    "DiDeMo",
    "MSR-VTT",
    "MSVD",
    "VATEX",
    "YouCook2",
    "QVHighlight",
    "Charades-STA",
    "MomentSeeker",
)


VISDOC_TASKS = (
    "ViDoRe_arxivqa",
    "ViDoRe_docvqa",
    "ViDoRe_infovqa",
    "ViDoRe_tabfquad",
    "ViDoRe_tatdqa",
    "ViDoRe_shiftproject",
    "ViDoRe_syntheticDocQA_artificial_intelligence",
    "ViDoRe_syntheticDocQA_energy",
    "ViDoRe_syntheticDocQA_government_reports",
    "ViDoRe_syntheticDocQA_healthcare_industry",
    "ViDoRe_esg_reports_human_labeled_v2",
    "ViDoRe_biomedical_lectures_v2_multilingual",
    "ViDoRe_economics_reports_v2_multilingual",
    "ViDoRe_esg_reports_v2_multilingual",
    "VisRAG_ArxivQA",
    "VisRAG_ChartQA",
    "VisRAG_MP-DocVQA",
    "VisRAG_SlideVQA",
    "VisRAG_InfoVQA",
    "VisRAG_PlotQA",
    "ViDoSeek-page",
    "ViDoSeek-doc",
    "MMLongBench-page",
    "MMLongBench-doc",
)


TASKS_BY_MODALITY = {
    "image": IMAGE_TASKS,
    "video": VIDEO_TASKS,
    "visdoc": VISDOC_TASKS,
}


MVBench_SUBSETS = (
    "episodic_reasoning",
    "action_sequence",
    "action_prediction",
    "action_antonym",
    "fine_grained_action",
    "unexpected_action",
    "object_existence",
    "object_interaction",
    "object_shuffle",
    "moving_direction",
    "action_localization",
    "scene_transition",
    "action_count",
    "moving_count",
    "moving_attribute",
    "state_change",
    "fine_grained_pose",
    "character_order",
    "egocentric_navigation",
    "counterfactual_inference",
)


# repo, subset, split, local relative path, required columns
VIDEO_METADATA = {
    "K700": (
        "VLM2Vec/Kinetics-700",
        None,
        "test",
        "video-tasks/data/k700.jsonl",
        ("video_id", "pos_text"),
    ),
    "SmthSmthV2": (
        "VLM2Vec/SmthSmthV2",
        None,
        "test",
        "video-tasks/data/ssv2.jsonl",
        ("video_id", "pos_text", "neg_text"),
    ),
    "HMDB51": (
        "VLM2Vec/HMDB51",
        None,
        "test",
        "video-tasks/data/hmdb51.jsonl",
        ("video_id", "pos_text"),
    ),
    "UCF101": (
        "VLM2Vec/UCF101",
        None,
        "test",
        "video-tasks/data/ucf101.jsonl",
        ("video_id", "pos_text"),
    ),
    "Breakfast": (
        "VLM2Vec/Breakfast",
        None,
        "test",
        "video-tasks/data/breakfast.jsonl",
        ("video_id", "pos_text"),
    ),
    "MVBench": (
        "VLM2Vec/MVBench",
        None,
        "train",
        "video-tasks/data/mvbench",
        ("video", "question", "answer", "candidates"),
    ),
    "Video-MME": (
        "VLM2Vec/Video-MME",
        None,
        "test",
        "video-tasks/data/video-mme.jsonl",
        ("videoID", "question", "options", "answer"),
    ),
    "NExTQA": (
        "VLM2Vec/NExTQA",
        "MC",
        "test",
        "video-tasks/data/nextqa.jsonl",
        ("video", "question", "answer"),
    ),
    "EgoSchema": (
        "VLM2Vec/EgoSchema",
        "Subset",
        "test",
        "video-tasks/data/egoschema.jsonl",
        ("video_idx", "question", "answer", "option"),
    ),
    "ActivityNetQA": (
        "VLM2Vec/ActivityNetQA",
        None,
        "test",
        "video-tasks/data/activitynetqa.jsonl",
        ("video_name", "question", "answer"),
    ),
    "DiDeMo": (
        "VLM2Vec/DiDeMo",
        None,
        "test",
        "video-tasks/data/didemo.jsonl",
        ("video", "caption"),
    ),
    "MSR-VTT": (
        "VLM2Vec/MSR-VTT",
        "test_1k",
        "test",
        "video-tasks/data/msr-vtt.jsonl",
        ("video_id", "video", "caption"),
    ),
    "MSVD": (
        "VLM2Vec/MSVD",
        None,
        "test",
        "video-tasks/data/msvd.jsonl",
        ("video_id", "video", "caption"),
    ),
    "VATEX": (
        "VLM2Vec/VATEX",
        None,
        "test",
        "video-tasks/data/vatex.jsonl",
        ("videoID", "enCap"),
    ),
    "YouCook2": (
        "lmms-lab/YouCook2",
        None,
        "val",
        "video-tasks/data/youcook2-val.jsonl",
        ("id", "video_path", "sentence"),
    ),
    "QVHighlight": (
        "VLM2Vec/QVHighlight",
        None,
        "test",
        "video-tasks/data/qvhighlight.jsonl",
        ("query", "video_path"),
    ),
    "Charades-STA": (
        "VLM2Vec/Charades-STA",
        None,
        "test",
        "video-tasks/data/charades_sta.jsonl",
        ("query", "video_path"),
    ),
    "MomentSeeker": (
        "VLM2Vec/MomentSeeker",
        None,
        "test",
        "video-tasks/data/momentseeker_1k6.jsonl",
        ("query", "positive_frames", "negative_frames", "input_frames"),
    ),
}


VIDEO_FRAME_ROOTS = {
    "K700": "video-tasks/frames/video_cls/K700",
    "SmthSmthV2": "video-tasks/frames/video_cls/SSv2",
    "HMDB51": "video-tasks/frames/video_cls/HMDB51",
    "UCF101": "video-tasks/frames/video_cls/UCF101",
    "Breakfast": "video-tasks/frames/video_cls/Breakfast",
    "MVBench": "video-tasks/frames/video_qa/MVBench",
    "Video-MME": "video-tasks/frames/video_qa/Video-MME",
    "NExTQA": "video-tasks/frames/video_qa/NExTQA",
    "EgoSchema": "video-tasks/frames/video_qa/egoschema",
    "ActivityNetQA": "video-tasks/frames/video_qa/ActivityNetQA",
    "DiDeMo": "video-tasks/frames/video_ret/data/ziyan/video_retrieval/DiDeMo/frames",
    "MSR-VTT": "video-tasks/frames/video_ret/data/ziyan/video_retrieval/MSR-VTT/frames",
    "MSVD": "video-tasks/frames/video_ret/data/ziyan/video_retrieval/MSVD/frames",
    "VATEX": "video-tasks/frames/video_ret/data/ziyan/video_retrieval/VATEX/frames",
    "YouCook2": "video-tasks/frames/video_ret/data/ziyan/video_retrieval/YouCook2/frames",
    "QVHighlight": "video-tasks/frames/video_mret/QVHighlight",
    "Charades-STA": "video-tasks/frames/video_mret/Charades-STA",
    "MomentSeeker": "video-tasks/frames/video_mret/MomentSeeker",
}


# data directory, image directory, expected split prefix
VISDOC_LAYOUT = {
    "ViDoRe_arxivqa": ("arxivqa_test_subsampled_beir", "ViDoRe_arxivqa", "test"),
    "ViDoRe_docvqa": ("docvqa_test_subsampled_beir", "ViDoRe_docvqa", "test"),
    "ViDoRe_infovqa": ("infovqa_test_subsampled_beir", "ViDoRe_infovqa", "test"),
    "ViDoRe_tabfquad": ("tabfquad_test_subsampled_beir", "ViDoRe_tabfquad", "test"),
    "ViDoRe_tatdqa": ("tatdqa_test_beir", "ViDoRe_tatdqa", "test"),
    "ViDoRe_shiftproject": ("shiftproject_test_beir", "ViDoRe_shiftproject", "test"),
    "ViDoRe_syntheticDocQA_artificial_intelligence": (
        "syntheticDocQA_artificial_intelligence_test_beir",
        "ViDoRe_syntheticDocQA_artificial_intelligence",
        "test",
    ),
    "ViDoRe_syntheticDocQA_energy": (
        "syntheticDocQA_energy_test_beir",
        "ViDoRe_syntheticDocQA_energy",
        "test",
    ),
    "ViDoRe_syntheticDocQA_government_reports": (
        "syntheticDocQA_government_reports_test_beir",
        "ViDoRe_syntheticDocQA_government_reports",
        "test",
    ),
    "ViDoRe_syntheticDocQA_healthcare_industry": (
        "syntheticDocQA_healthcare_industry_test_beir",
        "ViDoRe_syntheticDocQA_healthcare_industry",
        "test",
    ),
    "ViDoRe_esg_reports_human_labeled_v2": (
        "esg_reports_human_labeled_v2",
        "esg_reports_human_labeled_v2",
        "test",
    ),
    "ViDoRe_biomedical_lectures_v2_multilingual": (
        "biomedical_lectures_v2",
        "biomedical_lectures_v2_multilingual",
        "test",
    ),
    "ViDoRe_economics_reports_v2_multilingual": (
        "economics_reports_v2",
        "economics_reports_v2_multilingual",
        "test",
    ),
    "ViDoRe_esg_reports_v2_multilingual": (
        "esg_reports_v2",
        "esg_reports_v2_multilingual",
        "test",
    ),
    "VisRAG_ArxivQA": ("VisRAG-Ret-Test-ArxivQA", "VisRAG_ArxivQA", "train"),
    "VisRAG_ChartQA": ("VisRAG-Ret-Test-ChartQA", "VisRAG_ChartQA", "train"),
    "VisRAG_MP-DocVQA": ("VisRAG-Ret-Test-MP-DocVQA", "VisRAG_MP-DocVQA", "train"),
    "VisRAG_SlideVQA": ("VisRAG-Ret-Test-SlideVQA", "VisRAG_SlideVQA", "train"),
    "VisRAG_InfoVQA": ("VisRAG-Ret-Test-InfoVQA", "VisRAG_InfoVQA", "train"),
    "VisRAG_PlotQA": ("VisRAG-Ret-Test-PlotQA", "VisRAG_PlotQA", "train"),
    "ViDoSeek-page": ("ViDoSeek-page", "ViDoSeek-page", "test"),
    # The document-level corpus uses the same rendered images as page-level ViDoSeek.
    "ViDoSeek-doc": ("ViDoSeek", "ViDoSeek-page", "test"),
    "MMLongBench-page": ("MMLongBench", "MMLongBench-page", "test"),
    "MMLongBench-doc": ("MMLongBench-doc", "MMLongBench-doc", "test"),
}


def all_task_names() -> tuple[str, ...]:
    return IMAGE_TASKS + VIDEO_TASKS + VISDOC_TASKS


def modality_for_task(task_name: str) -> str:
    for modality, task_names in TASKS_BY_MODALITY.items():
        if task_name in task_names:
            return modality
    raise KeyError(task_name)

