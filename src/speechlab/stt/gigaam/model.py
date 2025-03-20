from torch import Tensor, nn, full, device, float16, autocast, inference_mode


class GigaAMModel(nn.Module):
    def __init__(
        self,
        preprocessor: nn.Module,
        encoder: nn.Module,
        head: nn.Module,
        decoding: nn.Module,
        device: device,
    ) -> None:
        super().__init__()

        self.preprocessor = preprocessor
        self.encoder = encoder
        self.head = head
        self.decoding = decoding
        self._device = device
        self._autocast = self._device.type != "cpu"

    @inference_mode()
    def stt(self, data: Tensor) -> str:
        tdata = data.to(device=self._device).unsqueeze(0)
        tlength = full([1], tdata.shape[-1], device=self._device)

        features, feature_lengths = self.preprocessor(tdata, tlength)
        with autocast(
            device_type=self._device.type,
            dtype=float16,
            enabled=self._autocast,
        ):
            encoded, encoded_len = self.encoder(features, feature_lengths)

        return self.decoding.decode(self.head, encoded, encoded_len)

    def clear(self) -> None:
        self.to("cpu")
