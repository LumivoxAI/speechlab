from .elements import (
    AppSink,
    AudioQueue,
    CapsFilter,
    PipeWireSrc,
    AudioConvert,
    AudioResample,
    AudioQueueDropPolicy,
)
from .base_pipeline import SamplesHandler, BaseCapturePipeline


class CapturePipeline(BaseCapturePipeline):
    """
    pipewiresrc -> audioconvert -> audioresample -> capsfilter -> queue -> appsink
    """

    def __init__(
        self,
        handler: SamplesHandler,
        target_object: str,
        samplerate: int,
        channels: int,
        client_name: str | None = None,
        resample_quality: int = 4,
        queue_size_ms: int = 200,
    ) -> None:
        super().__init__("capture-pipeline", handler)

        self._target_object = target_object
        self._samplerate = samplerate
        self._channels = channels
        self._client_name = client_name
        self._resample_quality = resample_quality
        self._queue_size_ms = queue_size_ms

        self._appsink: AppSink | None = None

    def _build(self) -> None:
        src = PipeWireSrc(
            target_object=self._target_object,
            client_name=self._client_name,
            name="capture-src",
        )
        convert = AudioConvert(name="capture-convert")
        resample = AudioResample(quality=self._resample_quality, name="capture-resample")
        capsfilter = CapsFilter(
            samplerate=self._samplerate,
            channels=self._channels,
            name="capture-caps",
        )
        q = AudioQueue(
            max_size_time_ms=self._queue_size_ms,
            drop_policy=AudioQueueDropPolicy.DROP_NEW,
            name="capture-queue",
        )
        self._appsink = AppSink("capture-appsink")

        self._add_and_link([src, convert, resample, capsfilter, q, self._appsink])

    def _get_appsink(self) -> AppSink | None:
        return self._appsink
