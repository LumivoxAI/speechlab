import sys
from venv import logger
from typing import Any
from pathlib import Path

import click

from speechlab.transport.loader import BaseLoader


class EntryPoint:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def _load_model(self, loader: BaseLoader, cfg_name: str) -> Any:
        if cfg_name == "default":
            return loader.from_default(cfg_name)

        return loader.from_file(cfg_name)

    def fish_speech(self, cfg_name: str) -> None:
        from speechlab.tts.server import TTSServer
        from speechlab.tts.fish_speech.model import FishSpeechModel
        from speechlab.tts.fish_speech.loader import FishSpeechModelLoader
        from speechlab.tts.fish_speech.handler import FishSpeechHandler

        loader = FishSpeechModelLoader(self._data_dir)
        model: FishSpeechModel = self._load_model(loader, cfg_name)
        handler = FishSpeechHandler(model)
        server = TTSServer(handler, "tcp://*:5501", "FishSpeech")
        server.run()

    def xtts(self, cfg_name: str) -> None:
        from speechlab.tts.server import TTSServer
        from speechlab.tts.xtts.model import XTTSModel
        from speechlab.tts.xtts.loader import XTTSModelLoader
        from speechlab.tts.xtts.handler import XTTSHandler

        loader = XTTSModelLoader(self._data_dir)
        model: XTTSModel = self._load_model(loader, cfg_name)
        handler = XTTSHandler(model)
        server = TTSServer(handler, "tcp://*:5502", "XTTS")
        server.run()

    def giga_am(self, cfg_name: str) -> None:
        from speechlab.stt.server import STTServer
        from speechlab.stt.gigaam.model import GigaAMModel
        from speechlab.stt.gigaam.loader import GigaAMModelLoader
        from speechlab.stt.gigaam.handler import STTGigaAMHandler

        loader = GigaAMModelLoader(self._data_dir)
        model: GigaAMModel = self._load_model(loader, cfg_name)
        handler = STTGigaAMHandler(model, inactivity_timeout_ms=10 * 1000)
        server = STTServer(handler, "tcp://*:5510", "GigaAM")
        server.run()

    def ru_norm(self, cfg_name: str) -> None:
        from speechlab.preprocess.server import PreprocessServer
        from speechlab.preprocess.runorm.model import RuNormModel
        from speechlab.preprocess.runorm.loader import RuNormModelLoader
        from speechlab.preprocess.runorm.handler import RuNormHandler

        loader = RuNormModelLoader(self._data_dir)
        model: RuNormModel = self._load_model(loader, cfg_name)
        handler = RuNormHandler(model, "RuNorm")
        server = PreprocessServer(handler, "tcp://*:5520", "RuNorm")
        server.run()


@click.command()
@click.option("--docker", default=True, help="Run in Docker mode")
@click.option("--data_dir", default="/data", help="Path to the data directory")
@click.option("--fish_speech", default="none", help="Fish speech config")
@click.option("--xtts", default="none", help="XTTS config")
@click.option("--giga_am", default="none", help="GigaAM config")
@click.option("--ru_norm", default="none", help="RuNorm config")
def main(
    docker: bool,
    data_dir: str,
    fish_speech: str,
    xtts: str,
    giga_am: str,
    ru_norm: str,
) -> None:
    if docker:
        data_dir = Path(data_dir)
    else:
        project_root = Path(__file__).resolve().parent
        data_dir = project_root.parent.parent / "data"

        src_path = str(project_root)
        if src_path not in sys.path:
            sys.path.append(src_path)

    from speechlab.runner import Runner

    ep = EntryPoint(data_dir)
    runner = Runner()

    if fish_speech != "none":
        runner.start(
            "FishSpeech",
            ep.fish_speech,
            cfg_name=fish_speech,
        )
    if xtts != "none":
        runner.start(
            "XTTS",
            ep.xtts,
            cfg_name=xtts,
        )
    if giga_am != "none":
        runner.start(
            "GigaAM",
            ep.giga_am,
            cfg_name=giga_am,
        )
    if ru_norm != "none":
        runner.start(
            "RuNorm",
            ep.ru_norm,
            cfg_name=ru_norm,
        )
    runner.wait()


if __name__ == "__main__":
    main()
