# (repo, subset, split)
import os

BASE_RAW_DATA_DIR = os.environ.get(
    "MMEB_V2_DATA_DIR",
    os.environ.get("MMEB_V3_DATA_DIR", "data/MMEB-V3"),
)
IMAGE_QUERY_DATA_DIR = os.environ.get(
    "MMEB_V2_IMAGE_QUERY_DIR",
    os.path.join(BASE_RAW_DATA_DIR, "image-query"),
)

EVAL_DATASET_HF_PATH = {
    # Video-RET
    "MSR-VTT": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "msr-vtt.jsonl"), None, "test"),
    "MSVD": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "msvd.jsonl"), None, "test"),
    "DiDeMo": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "didemo.jsonl"), None, "test"),
    "YouCook2": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "youcook2-val.jsonl"), None, "val"),
    "VATEX": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "vatex.jsonl"), None, "test"),

    # Video-CLS
    "HMDB51": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "hmdb51.jsonl"), None, "test"),
    "UCF101": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "ucf101.jsonl"), None, "test"),
    "Breakfast": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "breakfast.jsonl"), None, "test"),
    "Kinetics-700": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "k700.jsonl"), None, "test"),
    "SmthSmthV2": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "ssv2.jsonl"), None, "test"),

    # Video-MRET
    "QVHighlight": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "qvhighlight.jsonl"), None, "test"),
    "Charades-STA": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "charades_sta.jsonl"), None, "test"),
    "MomentSeeker": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "momentseeker_1k6.jsonl"), None, "test"),
    "MomentSeeker_1k8": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "momentseeker_1k8.jsonl"), None, "test"),

    # Video-QA
    "NExTQA": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "nextqa.jsonl"), "MC", "test"),
    "EgoSchema": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "egoschema.jsonl"), "Subset", "test"),
    "MVBench": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "mvbench"), None, "train"),
    "Video-MME": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "video-mme.jsonl"), None, "test"),
    "ActivityNetQA": (os.path.join(BASE_RAW_DATA_DIR, "video-tasks", "data", "activitynetqa.jsonl"), None, "test"),

    # Visdoc-ViDoRe
    "ViDoRe_arxivqa": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "arxivqa_test_subsampled_beir"), None, "test"),
    "ViDoRe_docvqa": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "docvqa_test_subsampled_beir"), None, "test"),
    "ViDoRe_infovqa": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "infovqa_test_subsampled_beir"), None, "test"),
    "ViDoRe_tabfquad": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "tabfquad_test_subsampled_beir"), None, "test"),
    "ViDoRe_tatdqa": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "tatdqa_test_beir"), None, "test"),
    "ViDoRe_shiftproject": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "shiftproject_test_beir"), None, "test"),
    "ViDoRe_syntheticDocQA_artificial_intelligence": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "syntheticDocQA_artificial_intelligence_test_beir"), None, "test"),
    "ViDoRe_syntheticDocQA_energy": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "syntheticDocQA_energy_test_beir"), None, "test"),
    "ViDoRe_syntheticDocQA_government_reports": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "syntheticDocQA_government_reports_test_beir"), None, "test"),
    "ViDoRe_syntheticDocQA_healthcare_industry": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "syntheticDocQA_healthcare_industry_test_beir"), None, "test"),

    # Visdoc-VisRAG
    "VisRAG_ArxivQA": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "VisRAG-Ret-Test-ArxivQA"), None, "train"),
    "VisRAG_ChartQA": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "VisRAG-Ret-Test-ChartQA"), None, "train"),
    "VisRAG_MP-DocVQA": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "VisRAG-Ret-Test-MP-DocVQA"), None, "train"),
    "VisRAG_SlideVQA": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "VisRAG-Ret-Test-SlideVQA"), None, "train"),
    "VisRAG_InfoVQA": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "VisRAG-Ret-Test-InfoVQA"), None, "train"),
    "VisRAG_PlotQA": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "VisRAG-Ret-Test-PlotQA"), None, "train"),

    # Visdoc-ViDoSeek
    "ViDoSeek-doc": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "ViDoSeek"), None, "test"),
    "ViDoSeek-page": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "ViDoSeek-page"), None, "test"),
    "MMLongBench-doc": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "MMLongBench-doc"), None, "test"),
    "MMLongBench-page": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "MMLongBench"), None, "test"),

    # Visdoc-ViDoRe_v2
    "ViDoRe_esg_reports_human_labeled_v2": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "esg_reports_human_labeled_v2"), None, "test"),
    "ViDoRe_biomedical_lectures_v2": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "biomedical_lectures_v2"), "english", "test"),
    "ViDoRe_biomedical_lectures_v2_multilingual": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "biomedical_lectures_v2"), None, "test"),
    "ViDoRe_economics_reports_v2": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "economics_reports_v2"), "english", "test"),
    "ViDoRe_economics_reports_v2_multilingual": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "economics_reports_v2"), None, "test"),
    "ViDoRe_esg_reports_v2": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "esg_reports_v2"), "english", "test"),
    "ViDoRe_esg_reports_v2_multilingual": (os.path.join(BASE_RAW_DATA_DIR, "visdoc-tasks", "data", "esg_reports_v2"), None, "test"),

    # NOTE: Comment translated to English.
    "ImageNet-1K": (IMAGE_QUERY_DATA_DIR, "ImageNet-1K", "test"),
    "N24News": (IMAGE_QUERY_DATA_DIR, "N24News", "test"),
    "HatefulMemes": (IMAGE_QUERY_DATA_DIR, "HatefulMemes", "test"),
    "VOC2007": (IMAGE_QUERY_DATA_DIR, "VOC2007", "test"),
    "SUN397": (IMAGE_QUERY_DATA_DIR, "SUN397", "test"),
    "Place365": (IMAGE_QUERY_DATA_DIR, "Place365", "test"),
    "ImageNet-A": (IMAGE_QUERY_DATA_DIR, "ImageNet-A", "test"),
    "ImageNet-R": (IMAGE_QUERY_DATA_DIR, "ImageNet-R", "test"),
    "ObjectNet": (IMAGE_QUERY_DATA_DIR, "ObjectNet", "test"),
    "Country211": (IMAGE_QUERY_DATA_DIR, "Country211", "test"),
    # Image-QA
    "OK-VQA": (IMAGE_QUERY_DATA_DIR, "OK-VQA", "test"),
    "A-OKVQA": (IMAGE_QUERY_DATA_DIR, "A-OKVQA", "test"),
    "DocVQA": (IMAGE_QUERY_DATA_DIR, "DocVQA", "test"),
    "InfographicsVQA": (IMAGE_QUERY_DATA_DIR, "InfographicsVQA", "test"),
    "ChartQA": (IMAGE_QUERY_DATA_DIR, "ChartQA", "test"),
    "Visual7W": (IMAGE_QUERY_DATA_DIR, "Visual7W", "test"),
    "ScienceQA": (IMAGE_QUERY_DATA_DIR, "ScienceQA", "test"),
    "VizWiz": (IMAGE_QUERY_DATA_DIR, "VizWiz", "test"),
    "GQA": (IMAGE_QUERY_DATA_DIR, "GQA", "test"),
    "TextVQA": (IMAGE_QUERY_DATA_DIR, "TextVQA", "test"),
    # Image-RET
    "VisDial": (IMAGE_QUERY_DATA_DIR, "VisDial", "test"),
    "CIRR": (IMAGE_QUERY_DATA_DIR, "CIRR", "test"),
    "VisualNews_t2i": (IMAGE_QUERY_DATA_DIR, "VisualNews_t2i", "test"),
    "VisualNews_i2t": (IMAGE_QUERY_DATA_DIR, "VisualNews_i2t", "test"),
    "MSCOCO_t2i": (IMAGE_QUERY_DATA_DIR, "MSCOCO_t2i", "test"),
    "MSCOCO_i2t": (IMAGE_QUERY_DATA_DIR, "MSCOCO_i2t", "test"),
    "NIGHTS": (IMAGE_QUERY_DATA_DIR, "NIGHTS", "test"),
    "WebQA": (IMAGE_QUERY_DATA_DIR, "WebQA", "test"),
    "FashionIQ": (IMAGE_QUERY_DATA_DIR, "FashionIQ", "test"),
    "Wiki-SS-NQ": (IMAGE_QUERY_DATA_DIR, "Wiki-SS-NQ", "test"),
    "OVEN": (IMAGE_QUERY_DATA_DIR, "OVEN", "test"),
    "EDIS": (IMAGE_QUERY_DATA_DIR, "EDIS", "test"),
    # Image-VG
    "MSCOCO": (IMAGE_QUERY_DATA_DIR, "MSCOCO", "test"),
    "RefCOCO": (IMAGE_QUERY_DATA_DIR, "RefCOCO", "test"),
    "RefCOCO-Matching": (IMAGE_QUERY_DATA_DIR, "RefCOCO-Matching", "test"),
    "Visual7W-Pointing": (IMAGE_QUERY_DATA_DIR, "Visual7W-Pointing", "test"),

    # ToolDe
    "ToolDe-Queries-web": (os.path.join(BASE_RAW_DATA_DIR, "tool-tasks", "ToolDe-Queries", "web"), "ToolDe-Queries-web", "test"),
    "ToolDe-Queries-code": (os.path.join(BASE_RAW_DATA_DIR, "tool-tasks", "ToolDe-Queries", "code"), "ToolDe-Queries-code", "test"),
    "ToolDe-Queries-customized": (os.path.join(BASE_RAW_DATA_DIR, "tool-tasks", "ToolDe-Queries", "customized"), "ToolDe-Queries-customized", "test"),
    
    # Audio-CLS
    "NSynth": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "nsynth-1k"), "NSynth", "test"),
    "ESC-50": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "esc50"), "ESC-50", "test"),
    "UrbanSound8K": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "urbansound8k"), "UrbanSound8K", "test"),
    "SpeechCommands": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "speechcommand-1k"), "SpeechCommands", "test"),
    "CREMA-D": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "creamD"), "CREMA-D", "test"),
    # Audio-RET
    "SoundDescs": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "sounddescs-1k"), "SoundDescs", "test"),
    "Clotho": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "clotho"), "Clotho", "test"),
    "AVE": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "AVE", "AVE_Dataset"), "AVE", "test"),
    "SpeechCOCO": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "speechcoco-1k"), "SpeechCOCO", "validation"),
    # Audio-GND
    "TUTSound": (os.path.join(BASE_RAW_DATA_DIR, "audio-tasks", "tutsound"), "TUTSound", "test"),


    # Memory-RET
    "KnowMeBench": (os.path.join(BASE_RAW_DATA_DIR, "memory-tasks", "Episodic", "KnowMeBench"), None, "test"),
    "REALTALK": (os.path.join(BASE_RAW_DATA_DIR, "memory-tasks", "Dialogue", "REALTALK"), None, "test"),
    "PeerQA": (os.path.join(BASE_RAW_DATA_DIR, "memory-tasks", "Semantic", "PeerQA"), None, "test"),
    "DeepPlanning": (os.path.join(BASE_RAW_DATA_DIR, "memory-tasks", "Procedural", "DeepPlanning"), None, "test"),

    # GAE-GUIAct
    "GAE-GUIAct_q2t": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-GUIAct"), None, "q2t"),
    "GAE-GUIAct_q2s": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-GUIAct"), None, "q2s"),
    "GAE-GUIAct_s2s": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-GUIAct"), None, "s2s"),
    "GAE-GUIAct_t2s": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-GUIAct"), None, "t2s"),
    # GAE-Mind2Web
    "GAE-Mind2Web_q2t": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-Mind2Web"), None, "q2t"),
    "GAE-Mind2Web_q2s": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-Mind2Web"), None, "q2s"),
    "GAE-Mind2Web_s2s": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-Mind2Web"), None, "s2s"),
    "GAE-Mind2Web_t2s": (os.path.join(BASE_RAW_DATA_DIR, "gui-tasks", "GAE-Mind2Web"), None, "t2s"),

}
