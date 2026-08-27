import os
import json
from datasets import GeneratorBasedBuilder, DatasetInfo, BuilderConfig, SplitGenerator, Split, Features, Value, Sequence

class AssetOpsBenchConfig(BuilderConfig):
    def __init__(self, filename, folder, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.folder = folder

class AssetOpsBench(GeneratorBasedBuilder):
    DEFAULT_CONFIG_NAME = "scenarios"
    BUILDER_CONFIGS = [
        AssetOpsBenchConfig(
            name="scenarios",
            folder="scenarios",
            filename="all_utterance.jsonl",
            description="Scenario definitions for AssetOpsBench"
        ),
        AssetOpsBenchConfig(
            name="failuresensoriq_all",
            folder="failuresensoriq_standard",
            filename="all.jsonl",
            description="Complete FailureSensorIQ dataset"
        ),
        AssetOpsBenchConfig(
            name="failuresensoriq_10_options",
            folder="failuresensoriq_standard",
            filename="all_10_options.jsonl",
            description="10-option multiple choice version"
        ),
        AssetOpsBenchConfig(
            name="failuresensoriq_perturbed_complex",
            folder="failuresensoriq_perturbed",
            filename="all_10_options_perturbed_complex.jsonl",
            description="Adversarially perturbed (complex)"
        ),
        AssetOpsBenchConfig(
            name="failuresensoriq_perturbed_simple",
            folder="failuresensoriq_perturbed",
            filename="all_10_options_all_perturbed_simple.jsonl",
            description="Simpler perturbed version"
        ),
        AssetOpsBenchConfig(
            name="failuresensoriq_multi_answers",
            folder="failuresensoriq_standard",
            filename="all_multi_answers.jsonl",
            description="Multiple-answer QA variant"
        ),
        AssetOpsBenchConfig(
            name="failuresensoriq_sample_50",
            folder="failuresensoriq_standard",
            filename="sample_50_questions.jsonl",
            description="Sampled 50-question subset"
        ),
    ]

    def _info(self):
        if self.config.name == "scenarios":
            features = Features({
                "id": Value("string"),
                "utterance": Value("string"),
                "tags": Sequence(Value("string")),
            })
        elif self.config.name.startswith("failuresensoriq_perturbed"):
            features = Features({
                "subject": Value("string"),
                "id": Value("int64"),
                "question": Value("string"),
                "options": Sequence(Value("string")),
                "option_ids": Sequence(Value("string")),
                "question_first": Value("bool"),
                "correct": Sequence(Value("bool")),
                "text_type": Value("string"),
                "trigger_statement": Value("string"),
                "context": Value("string"),
            })
        else:
            features = Features({
                "subject": Value("string"),
                "id": Value("int64"),
                "question": Value("string"),
                "options": Sequence(Value("string")),
                "option_ids": Sequence(Value("string")),
                "question_first": Value("bool"),
                "correct": Sequence(Value("bool")),
                "text_type": Value("string"),
                "asset_name": Value("string"),
                "relevancy": Value("string"),
                "question_type": Value("string"),
            })

        return DatasetInfo(
            description="AssetOpsBench benchmark dataset for industrial diagnostic agents. Includes scenarios and failure mode QA.",
            features=features,
            supervised_keys=None,
            homepage="https://github.com/ibm-research/AssetOpsBench",
            citation="Add your citation here",
        )

    def _split_generators(self, dl_manager):
        base_dir = os.path.join(os.path.dirname(__file__), "data")
        filepath = os.path.join(base_dir, self.config.folder, self.config.filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"filepath": filepath})]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    yield idx, json.loads(line)
