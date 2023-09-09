import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, decoder_cell, r, dim_mel, prenet=None, stop_threshold=-2):
        super().__init__()
        self.decoder_cell = decoder_cell
        self.prenet = prenet
        self.stop_threshold = stop_threshold
        self.r = r
        self.dim_mel = dim_mel

        self.fc_mel = nn.Linear(decoder_cell.dim_output, self.r * self.dim_mel)
        self.fc_stop = nn.Linear(decoder_cell.dim_output, self.r)
        self.p_no_forcing = 0.1

    def forward(self, memory, mmask, x=None, max_steps: int = 0):
        # memory: B x L x D_enc
        # x:      B x T x D_mel
        B = memory.shape[0]  # batch size
        L = memory.shape[1]  # text length
        dtype = memory.dtype
        device = memory.device

        state_t = self.decoder_cell.initial_state(B, L, dtype, device)

        # GO frame B x r x D_mel
        y_t = torch.zeros((B, self.r, self.dim_mel), dtype=dtype, device=device)

        # Split the teacher input into non-overlapping sequences
        if x is None:
            x_split = None
        else:
            T = (x.shape[1] // self.r) * self.r
            x_split = x[:, :T, :].split(self.r, dim=1)
            # drop_frame = torch.mean(x, dim=1).unsqueeze(1).tile(1, self.r, 1)

        y, s, w = [], [], []
        step = 0
        while True:
            y_t = y_t[:, -1, :].unsqueeze(1)
            d_t, c_t, state_t = self.decoder_cell(y_t, state_t, memory, mmask)
            # d_t = d_t.detach()

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
                if not self.p_no_forcing or torch.rand(1) > self.p_no_forcing:
                    y_t = x_split[step - 1]  # B x r x D_mel
            else:
                if torch.any(s_t < self.stop_threshold) or (max_steps and (step > max_steps)):
                    break

        y = torch.cat(y, dim=1)  # B x T x D_mel
        s = torch.cat(s, dim=1)  # B x T x 1
        w = torch.stack(w, dim=1)  # B x T x L

        return y, s, w
