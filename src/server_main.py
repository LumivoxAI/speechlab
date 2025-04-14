import sys
from typing import Any
from pathlib import Path

import click


class FishSpeech:
    def get_config(self) -> Any:
        from speechlab.tts.fish_speech.config import DeviceOption, ModelVersion, FishSpeechConfig

        return FishSpeechConfig(
            device=DeviceOption.CUDA,
            version=ModelVersion.V1_5,
            half_precision=True,
            compile=True,
        )

    def run(self, model_dir: Path, reference_dir: Path) -> None:
        from speechlab.tts.server import TTSServer
        from speechlab.tts.fish_speech.model import FishSpeechModel
        from speechlab.tts.fish_speech.config import FishSpeechConfig
        from speechlab.tts.fish_speech.loader import FishSpeechModelLoader
        from speechlab.tts.fish_speech.handler import FishSpeechHandler

        config: FishSpeechConfig = self.get_config()
        model: FishSpeechModel = FishSpeechModelLoader(model_dir, reference_dir).get_model(config)
        handler = FishSpeechHandler(model)
        server = TTSServer(handler, "tcp://*:5501", "FishSpeech")
        server.run()


class GigaAM:
    def get_config(self) -> Any:
        from speechlab.stt.gigaam.config import ModelOption, DeviceOption, GigaAMConfig

        return GigaAMConfig(
            device=DeviceOption.CUDA,
            model=ModelOption.RNNT_V2,
            half_encoder=True,
            compile=False,
        )

    def run(self, model_dir: Path) -> None:
        from speechlab.stt.server import STTServer
        from speechlab.stt.gigaam.model import GigaAMModel
        from speechlab.stt.gigaam.config import GigaAMConfig
        from speechlab.stt.gigaam.loader import GigaAMModelLoader
        from speechlab.stt.gigaam.handler import STTGigaAMHandler

        config: GigaAMConfig = self.get_config()
        model: GigaAMModel = GigaAMModelLoader(model_dir).get_model(config)
        handler = STTGigaAMHandler(model, inactivity_timeout_ms=10 * 1000)
        server = STTServer(handler, "tcp://*:5502", "GigaAM")
        server.run()


class RuNorm:
    def get_config(self) -> Any:
        from speechlab.preprocess.runorm.config import ModelSize, DeviceOption, RuNormConfig

        return RuNormConfig(
            device=DeviceOption.CUDA,
            model_size=ModelSize.MEDIUM,
        )

    def run(self, model_dir: Path, reference_dir: Path) -> None:
        from speechlab.preprocess.server import PreprocessServer
        from speechlab.preprocess.runorm.model import RuNormModel
        from speechlab.preprocess.runorm.config import RuNormConfig
        from speechlab.preprocess.runorm.loader import RuNormModelLoader
        from speechlab.preprocess.runorm.handler import RuNormHandler

        config: RuNormConfig = self.get_config()
        model: RuNormModel = RuNormModelLoader(model_dir).get_model(config)
        handler = RuNormHandler(model, "RuNorm")
        server = PreprocessServer(handler, "tcp://*:5503", "RuNorm")
        server.run()


@click.command()
@click.option("--docker", default=False, help="Run in Docker mode")
@click.option("--model_dir", default="/model", help="Path to the model directory")
@click.option("--reference_dir", default="/reference", help="Path to the reference directory")
def main(docker: bool, model_dir: str, reference_dir: str) -> None:
    if docker:
        model_dir = Path(model_dir) / "audio"
        reference_dir = Path(reference_dir)
    else:
        project_root = Path(__file__).resolve().parent
        model_dir = project_root.parent.parent / "model" / "audio"
        reference_dir = project_root.parent.parent / "reference"

        src_path = str(project_root)
        if src_path not in sys.path:
            sys.path.append(src_path)

    from speechlab.runner import Runner

    runner = Runner()
    runner.start(
        "FishSpeech",
        FishSpeech().run,
        model_dir=model_dir,
        reference_dir=reference_dir,
    )
    runner.start(
        "GigaAM",
        GigaAM().run,
        model_dir=model_dir,
        reference_dir=reference_dir,
    )
    runner.start(
        "RuNorm",
        RuNorm().run,
        model_dir=model_dir,
        reference_dir=reference_dir,
    )
    runner.wait()


if __name__ == "__main__":
    main()
