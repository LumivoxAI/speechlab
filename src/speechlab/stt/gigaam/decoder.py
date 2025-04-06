import torch
from torch import Tensor, nn


class RNNTJoint(nn.Module):
    def __init__(
        self,
        enc_hidden: int = 768,
        pred_hidden: int = 320,
        joint_hidden: int = 320,
        num_classes: int = 34,
    ) -> None:
        super().__init__()
        self.pred = nn.Linear(pred_hidden, joint_hidden)
        self.enc = nn.Linear(enc_hidden, joint_hidden)
        self.joint_net = nn.Sequential(nn.ReLU(), nn.Linear(joint_hidden, num_classes))


class RNNTDecoder(nn.Module):
    def __init__(
        self,
        pred_hidden: int = 320,
        pred_rnn_layers: int = 1,
        num_classes: int = 34,
    ) -> None:
        super().__init__()
        self.blank_id = num_classes - 1
        self.pred_hidden = pred_hidden
        self.embed = nn.Embedding(num_classes, pred_hidden, padding_idx=self.blank_id)
        self.lstm = nn.LSTM(pred_hidden, pred_hidden, pred_rnn_layers)
        self.pred = None  # RNNTJoint.pred = nn.Linear(pred_hidden, joint_hidden)
        self.joint_net = None  # RNNTJoint.joint_net = nn.Sequential(nn.ReLU(), nn.Linear(joint_hidden, num_classes))

    def forward(
        self,
        x: Tensor,
        token: Tensor,
        h: Tensor,
        c: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        emb: Tensor = self.embed(token)
        g, (h, c) = self.lstm(emb.unsqueeze(0), (h, c))
        pred_proj = self.pred(g[0])
        joint_out = self.joint_net(x + pred_proj)
        # origin: joint_out.argmax(dim=-1), h, c
        # change for tensor rt
        return joint_out.unsqueeze(0).argmax(dim=-1).squeeze(), h, c

    def input_example(self) -> tuple[Tensor]:
        device = next(self.parameters()).device
        x = torch.zeros(self.pred_hidden).to(device)
        token = torch.tensor(0).to(device)
        hidden_h = torch.zeros(1, self.pred_hidden).to(device)
        hidden_c = torch.zeros(1, self.pred_hidden).to(device)
        return x, token, hidden_h, hidden_c

    def input_names(self) -> list[str]:
        return ["x", "token_in", "h_in", "c_in"]

    def output_names(self) -> list[str]:
        return ["token_out", "h_out", "c_out"]


class RNNTHead(nn.Module):
    def __init__(self, decoder: dict[str, int], joint: dict[str, int]) -> None:
        super().__init__()
        self.decoder = RNNTDecoder(**decoder)
        self.joint = RNNTJoint(**joint)
        self.blank_id = self.decoder.blank_id
        self.max_symbols = 10  # from RNNTGreedyDecoding

    def init(self) -> None:
        self.decoder.pred = self.joint.pred
        self.decoder.joint_net = self.joint.joint_net
        del self.joint

    def forward(self, enc_proj: Tensor) -> Tensor:
        device = enc_proj.device
        dtype = torch.float32
        tsize = enc_proj.shape[0]
        hyp = torch.zeros(tsize * self.max_symbols, dtype=torch.long, device=device)
        hyp_idx = 0

        pred_hidden = 320  # RNNTDecoder.pred_hidden
        h = torch.zeros(
            1,
            pred_hidden,
            dtype=dtype,
            device=device,
        )
        c = torch.zeros(
            1,
            pred_hidden,
            dtype=dtype,
            device=device,
        )

        blank_id = torch.tensor(self.blank_id, device=device)
        token = blank_id
        decoder = self.decoder

        for t in range(tsize):
            x = enc_proj[t]  # [pred_hidden]
            for _ in range(self.max_symbols):
                new_token, new_h, new_c = decoder(x, token, h, c)
                if new_token == blank_id:
                    break

                hyp[hyp_idx] = new_token
                hyp_idx += 1
                h = new_h
                c = new_c
                token = new_token

        return hyp[:hyp_idx]
