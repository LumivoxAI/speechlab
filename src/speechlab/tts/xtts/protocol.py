from pydantic import Field
from typing_extensions import Annotated

from ...transport.message import BaseMessage


class XTTSRequest(BaseMessage):
    session_id: Annotated[
        int,
        Field(description="Служебное поле для идентификации и различения разных сессий/запросов."),
    ]

    text: Annotated[str, Field(description="Текст, который требуется озвучить.")]

    speaker_id: Annotated[
        str,
        Field(description="Идентификатор голоса, которым будет озвучен текст."),
    ]

    language: Annotated[
        str, Field(description="Язык текста, который требуется озвучить. Например: 'ru', 'en'.")
    ] = "ru"

    temperature: Annotated[
        float,
        Field(
            ge=0.1,
            le=2.0,
            description="Температура softmax для авторегрессионной модели. Меньше — более детерминированный результат, больше — больше вариативности.",
        ),
    ] = 0.75

    length_penalty: Annotated[
        float,
        Field(
            ge=0.0,
            le=2.0,
            description="Штраф за длину для авторегрессионного декодера. Чем выше значение, тем короче (жестче) будет результат.",
        ),
    ] = 1.0

    repetition_penalty: Annotated[
        float,
        Field(
            ge=1.0,
            le=10.0,
            description="Штраф за повторения для декодера, помогает уменьшить длинные паузы или 'ээээ'.",
        ),
    ] = 5.0

    top_k: Annotated[
        int,
        Field(
            ge=0,
            le=100,
            description="Сужает выбор следующего токена до top-K наиболее вероятных. Меньшее значение — более предсказуемый (скучный) результат.",
        ),
    ] = 50

    top_p: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Сужает выбор следующего токена до вероятностной массы top-P. Меньшее значение — более предсказуемый (скучный) результат.",
        ),
    ] = 0.85

    speed: Annotated[
        float,
        Field(
            ge=0.5,
            le=2.0,
            description="Скорость воспроизведения сгенерированного аудио. Далеко от 1.0 могут появляться артефакты.",
        ),
    ] = 1.0

    enable_text_splitting: Annotated[
        bool,
        Field(
            description="Разбивать ли текст на предложения и генерировать аудио для каждого предложения отдельно. Позволяет обрабатывать длинные тексты, но может теряться контекст между предложениями."
        ),
    ] = False


class XTTSResponse(BaseMessage):
    session_id: int
    samplerate: int
    is_final: bool
    error: str | None = None
