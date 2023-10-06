import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, decoder_cell, r, dim_mel, stop_threshold=-2.0):
        super().__init__()
        self.decoder_cell = decoder_cell
        self.stop_threshold = stop_threshold
        self.r = r
        self.dim_mel = dim_mel

        self.fc_mel = nn.Linear(decoder_cell.dim_output, self.r * self.dim_mel)
        self.fc_stop = nn.Linear(decoder_cell.dim_output, self.r)

    def forward(self, memory, mmask, x=None, max_steps: int = 0, p_no_forcing: float = None):
        # memory: B x L x D_enc
        # x:      B x T x D_mel
        B = memory.shape[0]  # batch size
        L = memory.shape[1]  # text length
        dtype = memory.dtype
        device = memory.device

        # print(
        #     f"|fc_mel| = {torch.norm(self.fc_mel.weight):.2f} |fc_stop| = {torch.norm(self.fc_stop.weight):.2f} "
        #     f"|prenet.layer1| = {torch.norm(self.decoder_cell.pre_net.layers[0].weight):.2f} "
        #     f"|prenet.layer2| = {torch.norm(self.decoder_cell.pre_net.layers[1].weight):.2f}"
        #     f"|att.query| = {torch.norm(self.decoder_cell.attention_module.query_layer.weight):.2f}"
        # )
        state_t = self.decoder_cell.initial_state(B, L, dtype, device)

        # GO frame B x r x D_mel
        y_t = torch.zeros((B, self.r, self.dim_mel), dtype=dtype, device=device)

        # Split the teacher input into non-overlapping sequences
        if x is None:
            x_split = None
        else:
            T = (x.shape[1] // self.r) * self.r
            x_split = x[:, :T, :].split(self.r, dim=1)
            # x_avg = torch.mean(x, dim=1).unsqueeze(1).expand_as(y_t)

        y, s, w = [], [], []
        step = 0
        while True:
            y_t = y_t[:, -1, :].unsqueeze(1)
            d_t, c_t, state_t = self.decoder_cell(y_t, state_t, memory, mmask)

            w_t = state_t[0]  # B x L
            s_t = self.fc_stop(d_t).unsqueeze(2)  # B x r x 1
            y_t = nn.functional.leaky_relu(self.fc_mel(d_t), 0.01)
            y_t = y_t.view(-1, self.r, self.dim_mel)  # B x r x D_mel

            y.append(y_t)
            s.append(s_t)
            w.append(w_t)

            step += 1
            if x_split is not None:
                if step >= len(x_split):
                    break
                # Force teacher inputs
                if not p_no_forcing or torch.rand(1) > p_no_forcing:
                    y_t = x_split[step - 1]  # B x r x D_mel
            else:
                if torch.any(s_t < self.stop_threshold) or (max_steps and (step > max_steps)):
                    break

        y = torch.cat(y, dim=1)  # B x T x D_mel
        s = torch.cat(s, dim=1)  # B x T x 1
        w = torch.stack(w, dim=1)  # B x T x L

        return y, s, w
