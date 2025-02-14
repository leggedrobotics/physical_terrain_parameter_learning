#
# Copyright (c) 2024, ETH Zurich, Jiaqi Chen.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
#
import torch.nn as nn
import numpy as np
import torch
from torch import Tensor
import math
import torch.nn.functional as F
from . import modules as rslgym_module
import pytorch_lightning as pl
from torch.nn import TransformerDecoder, TransformerDecoderLayer

""" Add an option to use only the output from the middle position of the sequence for prediction and loss calculation. """


class BinAccuracy:
    def __init__(self, ord_num, model_type, discretization="SID", shift_SID=0.01):
        self.ord_num = ord_num
        self.model_type = model_type
        self.discretization = discretization
        self.shift = shift_SID
        if self.model_type == "fric":
            self.alpha, self.beta = 0, 1
        elif self.model_type == "stiff":
            self.alpha, self.beta = 1, 10
        else:
            raise ValueError("Unknown model type")

    def _get_label(self, gt):
        if self.discretization == "SID":
            shift = torch.tensor(self.shift).to(gt.device)
            alpha = torch.tensor(self.alpha + self.shift).to(gt.device)
            beta = torch.tensor(self.beta + self.shift).to(gt.device)
            gt = gt + shift
            label = torch.log(gt / alpha) / torch.log(beta / alpha) * self.ord_num
        elif self.discretization == "UD":
            label = (gt - self.alpha) / (self.beta - self.alpha) * self.ord_num
        else:
            raise ValueError(f"Unknown discretization method: {self.discretization}")
        label = label.clamp(max=self.ord_num - 1, min=0)
        label = label.long()
        return label

    def calculate_accuracy(self, pred_or_prob, gt, greater_eq_probs=None):
        pred_label = self._get_label(pred_or_prob)
        gt_label = self._get_label(gt)
        # Initialize a dictionary to store probability vectors of false assignments
        assignments_probs = {
            i: {j: [] for j in range(self.ord_num)} for i in range(self.ord_num)
        }

        accuracies = []
        # Initialize confusion matrix
        confusion_matrix = torch.zeros(self.ord_num, self.ord_num, dtype=torch.int)
        for i in range(self.ord_num):
            if self.discretization == "UD":
                bin_start = self.alpha + i * (self.beta - self.alpha) / self.ord_num
                bin_end = self.alpha + (i + 1) * (self.beta - self.alpha) / self.ord_num
            else:  # SID
                shift = torch.tensor(self.shift).to(gt.device)
                shifted_alpha = torch.tensor(self.alpha + self.shift).to(gt.device)
                shifted_beta = torch.tensor(self.beta + self.shift).to(gt.device)
                bin_start = (
                    torch.exp(
                        torch.log(shifted_alpha)
                        + torch.log(shifted_beta / shifted_alpha) * i / self.ord_num
                    )
                    - shift
                )
                bin_end = (
                    torch.exp(
                        torch.log(shifted_alpha)
                        + torch.log(shifted_beta / shifted_alpha)
                        * (i + 1)
                        / self.ord_num
                    )
                    - shift
                )

            correct = (pred_label == i) & (gt_label == i)
            total = gt_label == i

            # Handle empty bins
            if total.float().sum() == 0:
                accuracies.append(float("nan"))
            else:
                accuracy = correct.float().sum() / total.float().sum()
                accuracies.append(accuracy.item())
            # Extract the predictions corresponding to the current gt bin
            mask = gt_label == i
            binned_preds_for_current_gt_bin = pred_label[mask]

            # Update the confusion matrix
            for j in range(self.ord_num):
                confusion_matrix[i, j] = (binned_preds_for_current_gt_bin == j).sum()
            # Check for false assignments and extract their probability vectors
            if greater_eq_probs is not None:
                for j in range(self.ord_num):
                    assignments_mask = (gt_label == i) & (pred_label == j)
                    assignments_indices = torch.nonzero(assignments_mask).squeeze()
                    AA = greater_eq_probs[
                        assignments_indices[:, 0],
                        assignments_indices[:, 1],
                        assignments_indices[:, 2],
                    ]
                    assignments_probs[i][j].extend(AA.tolist())
            print(
                f"Bin {i}: Range [{bin_start:.2f}, {bin_end:.2f}] - Accuracy: {accuracies[-1]*100:.2f}%"
            )
        # Display the confusion matrix
        print("Confusion Matrix:")
        print(confusion_matrix)
        if greater_eq_probs is not None:
            # Calculate and display the average probability vectors for false assignments
            print("\nAverage Probability vectors for assignments:")
            for i in range(self.ord_num):
                for j in range(self.ord_num):
                    # Before the problematic line
                    SS = assignments_probs[i][j]
                    avg_probs = [
                        sum(ch) / len(ch) for ch in zip(*assignments_probs[i][j])
                    ]
                    print(f"From bin {i} to bin {j}: {avg_probs}")
        return accuracies, confusion_matrix


class OrdinalRegressionLayer(nn.Module):
    def __init__(self, ord_num):
        super(OrdinalRegressionLayer, self).__init__()
        self.ord_num = ord_num

    def forward(self, x, training=True):
        N, seq_len, priv_dim = x.size()
        assert (
            priv_dim == 4 * self.ord_num * 2
        ), "Mismatch between priv_dim and expected dimensions"

        # Reshape such that each foot value gets ord_num * 2 bins
        x = x.view(N, seq_len, 4, self.ord_num, 2)
        if training:
            # Softmax over the last dimension
            ord_prob = F.log_softmax(x, dim=-1)
            return ord_prob
        else:
            ord_prob = F.softmax(x, dim=-1)[:, :, :, :, 0]
            ord_label = torch.sum((ord_prob > 0.5), dim=-1) - 1
            ord_label = ord_label.clamp(max=self.ord_num - 1, min=0)
            return ord_prob, ord_label


class OrdinalRegressionLoss(object):
    def __init__(self, ord_num, model_type, discretization="SID", shift_SID=0.01):
        self.ord_num = ord_num
        self.model_type = model_type
        if self.model_type == "fric":
            self.alpha, self.beta = 0, 1
        elif self.model_type == "stiff":
            self.alpha, self.beta = 1, 10
        else:
            raise ValueError("Unknown model type")
        self.discretization = discretization
        self.shift = shift_SID

    def _create_ord_label(self, gt):
        N, seq_len, priv_dim = gt.shape

        ord_c0 = torch.ones(N, seq_len, priv_dim, self.ord_num).to(gt.device)
        if self.discretization == "SID":
            shift = torch.tensor(self.shift).to(gt.device)
            alpha = torch.tensor(self.alpha + self.shift).to(gt.device)
            beta = torch.tensor(self.beta + self.shift).to(gt.device)
            gt = gt + shift
            label = torch.log(gt / alpha) / torch.log(beta / alpha) * self.ord_num
        elif self.discretization == "UD":
            label = (gt - self.alpha) / (self.beta - self.alpha) * self.ord_num
        else:
            raise ValueError(f"Unknown discretization method: {self.discretization}")
        label = label.clamp(max=self.ord_num - 1, min=0)
        label = label.long()
        label = label.unsqueeze(-1)
        mask = (
            torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False)
            .view(1, 1, 1, self.ord_num)
            .to(gt.device)
        )
        mask = mask.repeat(N, seq_len, priv_dim, 1).contiguous().long()
        mask = mask > label
        ord_c0[mask] = 0  # 1 Represents P(true value≥threshold of bin i)
        ord_c1 = 1 - ord_c0  # 1 Represents P(true value<threshold of bin i)
        ord_label = torch.stack((ord_c0, ord_c1), dim=-1)
        return ord_label, mask

    def __call__(self, prob, gt):
        # gt = torch.unsqueeze(gt, dim=3)
        ord_label, mask = self._create_ord_label(gt)
        entropy = -prob * ord_label

        loss = torch.sum(entropy, dim=(2, 3, 4))
        return loss.mean()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if d_model % 2 == 1:
            ori_d_model = d_model
            d_model = d_model + 1
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if ori_d_model % 2 == 1:
            pe = pe[:, :, :-1]
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# temporal-wise attention
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim**0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


class BeliefDecoderRNN(nn.Module):
    def __init__(
        self,
        sens_size: int,
        inner_state_size: int,
        priv_size=4,
        seq_length=10,
        hidden_to_pred_shape=[128, 64],
        hidden_to_gate_shape=[128, 64],
        gate_to_pred_shape=[64, 32],
        hidden_size=200,
        num_rnn_layers=2,
        l1_alpha=0.2,
        activation_fn=nn.LeakyReLU,
        device="cpu",
        mode="base",
        is_training=True,
        model_type="fric",
        paradigm=None,
    ):
        super(BeliefDecoderRNN, self).__init__()
        self.model_type = model_type
        self.mode = mode
        self.hidden_size = hidden_size
        self.l1_alpha = l1_alpha
        self.priv_size = priv_size
        self.inner_state_size = inner_state_size
        self.device = device
        self.lambda_temporal = 0.01
        self.paradigm = paradigm  # "ordinal" # or None
        self.middle_only = False
        self.exact_middle_step = None
        self.num_rnn_layers = num_rnn_layers
        encoder_input_dim = sens_size + inner_state_size
        if model_type == "fric":
            self.discretization = "SID"
            self.ord_num = 5
            self.shift = 0.2
        else:
            self.discretization = "SID"
            self.ord_num = 5
            self.shift = 1.0
        self.bin_accuracy_cal = BinAccuracy(
            self.ord_num,
            model_type=self.model_type,
            discretization=self.discretization,
            shift_SID=self.shift,
        )
        if self.paradigm == "ordinal":
            self.ordinal_regression_layer = OrdinalRegressionLayer(ord_num=self.ord_num)
            self.ordinal_regression_loss = OrdinalRegressionLoss(
                ord_num=self.ord_num,
                model_type=model_type,
                discretization=self.discretization,
                shift_SID=self.shift,
            )
            priv_size = priv_size * self.ord_num * 2
            self.priv_size = priv_size
            # hidden_to_pred_shape=[100]
            if hidden_to_pred_shape[-1] < priv_size or self.hidden_size < priv_size:
                raise ValueError(
                    "The final dimension of hidden_to_pred_shape or the hidden_size should be greater than or equal to priv_size."
                )

        if self.mode == "base":
            self.hidden_to_pred = rslgym_module.MLP(
                hidden_to_pred_shape,
                activation_fn,
                hidden_size,
                priv_size,
                init_scale=1.0 / np.sqrt(2),
            )
        elif self.mode == "bi":
            self.hidden_to_pred = rslgym_module.MLP(
                hidden_to_pred_shape,
                activation_fn,
                hidden_size * 2,
                priv_size,
                init_scale=1.0 / np.sqrt(2),
            )

        elif self.mode == "full":
            self.hidden_to_pred = rslgym_module.MLP(
                hidden_to_pred_shape,
                activation_fn,
                hidden_size,
                priv_size,
                init_scale=1.0 / np.sqrt(2),
            )
            self.hidden_to_gate = rslgym_module.MLP(
                hidden_to_gate_shape,
                activation_fn,
                hidden_size,
                inner_state_size,
                init_scale=1.0 / np.sqrt(2),
            )
            self.gate_to_pred = rslgym_module.MLP(
                gate_to_pred_shape,
                activation_fn,
                inner_state_size,
                priv_size,
                init_scale=1.0 / np.sqrt(2),
            )
        elif self.mode == "transformer":
            encoder_layer = TransformerDecoderLayer(d_model=encoder_input_dim, nhead=11)
            self.transformer_decoder = TransformerDecoder(
                encoder_layer, num_layers=num_rnn_layers
            )
            self.pos_encoder = PositionalEncoding(encoder_input_dim)
            self.output_layer = rslgym_module.MLP(
                hidden_to_pred_shape,
                activation_fn,
                encoder_input_dim,
                priv_size,
                init_scale=1.0 / np.sqrt(2),
            )
            pass

        # add a mode "parallel", using an independent GRU for exteroceptive input and add a gated attention layer to combine the two outputs
        elif self.mode == "parallel":
            self.alpha = None
            # detect whether sens_size is 341. If not, raise error
            if sens_size != 341:
                raise ValueError(
                    "The sens_size should be 341 (pro+exte) for parallel mode."
                )
            self.rnn_pro = nn.GRU(
                133,
                self.hidden_size,
                self.num_rnn_layers,
                bidirectional=True if self.mode == "bi" else False,
            )
            self.hidden_to_pred = rslgym_module.MLP(
                hidden_to_pred_shape,
                activation_fn,
                hidden_size,
                priv_size,
                init_scale=1.0 / np.sqrt(2),
            )
            self.hidden_to_gate = rslgym_module.MLP(
                hidden_to_gate_shape,
                activation_fn,
                hidden_size,
                priv_size,
                init_scale=1.0 / np.sqrt(2),
            )
            self.hidden_to_pred_exte = rslgym_module.MLP(
                hidden_to_pred_shape,
                activation_fn,
                hidden_size,
                priv_size,
                init_scale=1.0 / np.sqrt(2),
            )

        if self.mode != "transformer":
            self.rnn = nn.GRU(
                encoder_input_dim,
                self.hidden_size,
                self.num_rnn_layers,
                bidirectional=True if self.mode == "bi" else False,
            )
            self.attention = SelfAttention(
                2 * self.hidden_size if self.mode == "bi" else self.hidden_size
            )
            if self.mode == "parallel":
                self.attention_exte = SelfAttention(
                    2 * self.hidden_size if self.mode == "bi" else self.hidden_size
                )

        self.ranges_weight = None
        self.interval = None
        # in middle_only mode, priv_normalizer init, seq_length should be manually set to 1
        if self.paradigm != "ordinal":
            self.priv_normalizer = rslgym_module.EmpiricalNormalization(
                [seq_length if not self.middle_only else 1, priv_size],
                until=1e8,
                clip_threshold=10,
                is_training=is_training,
            )
        # self.priv_normalizer_input = rslgym_module.EmpiricalNormalization(
        #     [seq_length,encoder_input_dim], until=1e8, clip_threshold=10, is_training=is_training
        # )
        self.seq_length = seq_length
        self.reset_parameters()

    def forward(self, latent, hidden=None):
        if latent.shape[-1] == 441 or latent.shape[-1] == 233:
            sens = latent[:, :, :-100]
            inner = latent[:, :, -100:]
        else:
            sens = latent  # This assumes that in other cases, the entire latent is sens
            inner = torch.empty(
                (sens.shape[0], sens.shape[1], 0), device=sens.device, dtype=sens.dtype
            )  # Empty tensor with zero size in the last dimension

        x = self.convert_rnn_input(sens, inner)

        if self.mode != "transformer":
            hidden = hidden.detach()

            if self.mode == "parallel":
                # extract sens [:,:,-208:] as exteroceptive input and pass through the encoder. THen concantenate the embedding with sens[:,:,:133]
                exte_input = x[:, :, -208:]
                pro_input = x[:, :, :133]
                self.output, next_hidden = self.forward_rnn(x, hidden)
                self.output_weighted = self.attention(self.output)
                pro_output, _ = self.rnn_pro(pro_input, hidden)
                pro_output = pro_output.permute(1, 0, 2)
                pro_output_weighted = self.attention_exte(pro_output)
                pro_pred = self.hidden_to_pred_exte(pro_output_weighted)
                self.alpha = F.sigmoid(self.hidden_to_gate(self.output_weighted))
                filtered_pro_output = self.alpha * pro_pred
                pred = filtered_pro_output + self.hidden_to_pred(self.output_weighted)
            else:
                self.output, next_hidden = self.forward_rnn(x, hidden)
                self.output_weighted = self.attention(self.output)
        if self.mode == "base" or self.mode == "bi":
            pred = self.hidden_to_pred(self.output_weighted)
        elif self.mode == "full":
            latent = self.hidden_to_pred(self.output_weighted)

            inner_alpha = F.sigmoid(self.hidden_to_gate(self.output_weighted))
            filtered_inner = inner * inner_alpha
            # latent_alpha = F.pad(input=filtered_inner, pad=(0, self.priv_size - self.inner_state_size), mode="constant", value=0)
            latent_alpha = self.gate_to_pred(filtered_inner)
            self.latent = latent + latent_alpha
            # pred = torch.cat([sens, self.latent], 1)
            pred = self.latent
        elif self.mode == "transformer":
            # x=self.pos_encoder(x)
            transformer_output = self.transformer_decoder(x, memory=x)
            output = transformer_output.permute(1, 0, 2)
            # Pass the Transformer's output through the final layer
            pred = self.output_layer(output)
            next_hidden = hidden
            pass
        elif self.mode == "parallel":
            pass
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        """ if self.middle_only: extract the middle position pred into a single dimension (keep sequence formatt but=1) """
        if self.middle_only:
            if self.exact_middle_step is None:
                self.exact_middle_step = self.seq_length // 2
            pred = pred[:, self.exact_middle_step, :].unsqueeze(1)
        return pred, next_hidden

    def get_rnn_output(self):
        return self.output

    def get_latent(self):
        return self.latent

    def convert_rnn_input(self, sens, inner):
        # # Check if the dimensions of sens and inner are as expected
        # if len(sens.shape) != 3 or len(inner.shape) != 3:
        #     raise ValueError("Expected sens and inner to have 3 dimensions: [batch_size, sequence_length, feature_dim]")

        # Assuming sens and inner have shapes [batch_size, sequence_length, feature_dim]
        x = torch.cat([sens, inner], dim=-1)  # concatenate along the feature dimension
        # x=self.priv_normalizer_input(x)
        x = x.permute(
            1, 0, 2
        )  # Permute to get [sequence_length, batch_size, feature_size]
        return x

    def forward_rnn(self, x, hidden):
        output, next_hidden = self.rnn(x, hidden)
        output = output.permute(1, 0, 2)
        return output, next_hidden

    def init_hidden(self, batch_size):
        """
        Return new latent state.
        batch_size is the number of parallel environments being used.
        """
        h0 = torch.zeros(
            self.num_rnn_layers if self.mode != "bi" else 2 * self.num_rnn_layers,
            batch_size,
            self.hidden_size,
        ).to(self.device)
        return h0

    def reset_parameters(self):
        kaimin_normal = torch.nn.init.kaiming_normal_
        xavier_normal = torch.nn.init.xavier_normal_
        gain = nn.init.calculate_gain("relu")
        orthogonal = torch.nn.init.orthogonal_

        def weights_init():
            def fn(m):
                classname = m.__class__.__name__
                if classname.find("Conv") != -1 or classname.find("Linear") != -1:
                    kaimin_normal(m.weight.data)
                    try:
                        m.bias.data.fill_(0)
                    except:
                        pass

                if classname.find("PFGRU") != -1:
                    xavier_normal(m.fc_z.weight.data)
                    xavier_normal(m.fc_r.weight.data)
                    kaimin_normal(m.fc_n.weight.data, nonlinearity="relu")
                    orthogonal(m.fc_obs.weight.data, gain=gain)
                    m.fc_z.bias.data.fill_(0)
                    m.fc_r.bias.data.fill_(0)
                    m.fc_n.bias.data.fill_(0)
                    m.fc_obs.bias.data.fill_(0)

            return fn

        self.apply(weights_init())

    def calc_prediction_loss(self, output, priv):
        batch_size = priv.shape[0]
        # check whether sequence length dimension of output and priv is the same since we may set middle_only=True somewehere,if middle_only=True and dim not the same, extract the middle of priv then.
        if output.shape[1] != priv.shape[1]:
            if self.middle_only and output.shape[1] == 1:
                priv = priv[:, self.exact_middle_step, :].unsqueeze(1)
            else:
                raise ValueError(
                    "Expected output and priv to have the same sequence length dimension"
                )
        # For privileged info, not using alpha
        priv_recon = output
        if self.paradigm != "ordinal":
            normalized_priv = self.priv_normalizer(priv)
            loss_w = None
            if self.ranges_weight is not None and self.interval is not None:
                indices = (priv / self.interval).long()
                indices = torch.clamp(
                    indices, 0, len(self.ranges_weight) - 1
                )  # ensure indices are within range
                loss_w = self.ranges_weight[indices]
            self.priv_loss = self.recon_loss(priv_recon, normalized_priv, loss_w)
            # priv_recon_not_scaled=self.priv_normalizer.inverse(priv_recon)
            # self.priv_loss_not_scaled = self.recon_loss(priv_recon_not_scaled, priv,loss_w)
            self.loss = self.priv_loss

        else:
            prob = self.ordinal_regression_layer(output)
            self.loss = self.ordinal_regression_loss(prob, priv)

        return self.loss

    def recon_loss(self, output, true_state, loss_w_in=None):
        batch_size = true_state.shape[0]

        if loss_w_in is not None:
            # weights based on data balance
            loss_w = loss_w_in
        else:
            loss_w = torch.ones(
                batch_size, true_state.shape[1], true_state.shape[2]
            ).to(self.device)
        loss_l2 = self.weighted_mse(output, true_state, loss_w)
        loss_l1 = self.weighted_l1(output, true_state, loss_w)
        loss = loss_l2 + self.l1_alpha * loss_l1
        # Computing individual MSE and L1 for each foot
        # mse_errors = (output - true_state) ** 2
        # l1_errors = torch.abs(output - true_state)

        # # Weighted errors
        # weighted_mse_errors = loss_w * mse_errors
        # weighted_l1_errors = loss_w * l1_errors

        # # Taking mean over the sequence dimension (assuming it's the second dimension)
        # mean_mse_errors = weighted_mse_errors.mean(dim=1)
        # mean_l1_errors = weighted_l1_errors.mean(dim=1)

        # # Taking the maximum error among the feet
        # max_mse_error = mean_mse_errors.max(dim=1)[0]
        # max_l1_error = mean_l1_errors.max(dim=1)[0]

        # # Combining L2 and L1 losses
        # loss = max_mse_error.mean() + self.l1_alpha * max_l1_error.mean()

        return loss

    def get_unnormalized_recon(self, latent, hidden):
        if self.paradigm != "ordinal":
            priv_recon, next_hidden = self.forward(latent, hidden)
            # print(self.priv_normalizer._mean)
            # return self.priv_normalizer.inverse(priv_recon), self.exte_normalizer.inverse(exte_recon)
            return self.priv_normalizer.inverse(priv_recon), next_hidden
            # return priv_recon, exte_recon
        else:
            priv_recon, next_hidden = self.forward(latent, hidden)
            prob, label = self.ordinal_regression_layer(priv_recon, training=False)
            self.prob = prob
            if self.model_type == "fric":
                alpha, beta = 0, 1
            elif self.model_type == "stiff":
                alpha, beta = 1, 10
            else:
                raise ValueError("Unknown model type")
            if self.discretization == "SID":
                alpha = alpha + self.shift
                beta = beta + self.shift
                t0 = (
                    torch.exp(
                        np.log(alpha)
                        + np.log(beta / alpha) * label.float() / self.ord_num
                    )
                    - self.shift
                )
                t1 = (
                    torch.exp(
                        np.log(alpha)
                        + np.log(beta / alpha) * (label.float() + 1) / self.ord_num
                    )
                    - self.shift
                )
            else:
                t0 = alpha + (beta - alpha) * label.float() / self.ord_num
                t1 = alpha + (beta - alpha) * (label.float() + 1) / self.ord_num
            value = (t0 + t1) / 2
            return value, next_hidden

    def weighted_mse(self, a, b, w):
        # w = w.reshape(a.shape[0], 1)
        return (w * (a - b) ** 2).mean()

    def weighted_l1(self, a, b, w):
        # w = w.reshape(a.shape[0], 1)
        return (w * torch.abs(a - b)).mean()

    def get_loss(self):
        return self.loss

    def set_range_weight(self, data, use_weight):
        if self.model_type == "fric":
            interval = 0.1
            max_range = 1
            min_range = 0
        elif self.model_type == "stiff":
            interval = 1
            max_range = 10
            min_range = 1
        else:
            raise ValueError("Unknown model type")
        # Convert tuple of tensors to a single numpy array
        data = torch.cat(data, dim=0).cpu().numpy()
        # Adjust range if needed
        bins = np.arange(min_range, max_range + interval, interval)
        hist, _ = np.histogram(data, bins=bins)
        self.interval = interval
        # Compute weights based on the inverse of frequency
        if use_weight:
            weights = 1.0 / (hist + 1e-8)  # Adding epsilon to avoid division by zero
        else:
            weights = 0.0 * hist + 1.0
        weights /= weights.sum()

        self.ranges_weight = torch.tensor(weights).to(self.device)

    def generate_mask(self, seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask.to(self.device)


class BeliefDecoderMLP(nn.Module):
    def __init__(
        self,
        priv_size=4,
        priv_decoder_shape=[64, 32],
        latent_size=100,
        l1_alpha=0.2,
        activation_fn=nn.LeakyReLU,
        device="cpu",
        mode="base",
        is_training=True,
        model_type="fric",
    ):
        super(BeliefDecoderMLP, self).__init__()

        self.mode = mode
        self.latent_size = latent_size
        self.priv_size = priv_size
        self.l1_alpha = l1_alpha
        self.device = device
        self.model_type = model_type
        self.ranges_weight = None
        self.interval = None
        self.priv_normalizer = rslgym_module.EmpiricalNormalization(
            [priv_size], until=1e8, clip_threshold=10, is_training=is_training
        )

        self.priv_decoder = rslgym_module.MLP(
            priv_decoder_shape,
            activation_fn,
            self.latent_size,
            priv_size,
            init_scale=1.0 / np.sqrt(2),
        )

        if self.mode == "separate":
            self.separate_layers = nn.ModuleList(
                [nn.Linear(priv_size, priv_size // 4) for _ in range(4)]
            )

    def forward(self, latent):
        """
        Reconstruct priv states from prop, exte and latent of encoder
        Note that exte can be the raw input
        """
        priv_recon = self.priv_decoder(latent)

        if self.mode == "separate":
            priv_recon_separated = []
            for i in range(4):
                priv_recon_separated.append(
                    self.separate_layers[i](priv_recon[:, i : i + 1])
                )
            priv_recon = torch.cat(priv_recon_separated, dim=1)

        return priv_recon

    def calc_prediction_loss(self, latent, priv):
        batch_size = priv.shape[0]
        # For privileged info, not using alpha
        priv_recon = self.forward(latent)
        normalized_priv = self.priv_normalizer(priv)
        loss_w = None
        if self.ranges_weight is not None and self.interval is not None:
            indices = (priv / self.interval).long()
            indices = torch.clamp(
                indices, 0, len(self.ranges_weight) - 1
            )  # ensure indices are within range
            loss_w = self.ranges_weight[indices].reshape(batch_size, priv.shape[1])
        self.priv_loss = self.recon_loss(priv_recon, normalized_priv, loss_w)

        # self.priv_loss = self.recon_loss(priv_recon, priv)
        self.loss = self.priv_loss
        return self.loss

    def recon_loss(self, output, true_state, loss_w_in=None):
        batch_size = true_state.shape[0]
        if loss_w_in is not None:
            loss_w = loss_w_in
        else:
            loss_w = torch.ones(batch_size, true_state.shape[1]).to(self.device)
        loss_l2 = self.weighted_mse(output, true_state, loss_w)
        loss_l1 = self.weighted_l1(output, true_state, loss_w)
        loss = loss_l2 + self.l1_alpha * loss_l1
        return loss

    def get_unnormalized_recon(self, latent):
        priv_recon = self.forward(latent)
        # print(self.priv_normalizer._mean)
        # return self.priv_normalizer.inverse(priv_recon), self.exte_normalizer.inverse(exte_recon)
        return self.priv_normalizer.inverse(priv_recon)
        # return priv_recon, exte_recon

    def weighted_mse(self, a, b, w):
        # w = w.reshape(a.shape[0], 1)
        return (w * (a - b) ** 2).mean()

    def weighted_l1(self, a, b, w):
        # w = w.reshape(a.shape[0], 1)
        return (w * torch.abs(a - b)).mean()

    def get_loss(self):
        return self.loss

    def set_range_weight(self, data, use_weight):
        if self.model_type == "fric":
            interval = 0.1
            max_range = 1
            min_range = 0
        elif self.model_type == "stiff":
            interval = 1
            max_range = 10
            min_range = 1
        else:
            raise ValueError("Unknown model type")
        # Convert tuple of tensors to a single numpy array
        data = torch.cat(data, dim=0).cpu().numpy()
        # Adjust range if needed
        bins = np.arange(min_range, max_range + interval, interval)
        hist, _ = np.histogram(data, bins=bins)
        self.interval = interval
        # Compute weights based on the inverse of frequency
        if use_weight:
            weights = 1.0 / (hist + 1e-8)  # Adding epsilon to avoid division by zero
        else:
            weights = 0.0 * hist + 1.0
        weights /= weights.sum()

        self.ranges_weight = torch.tensor(weights).to(self.device)


class BeliefDecoderLightning(pl.LightningModule):
    def __init__(
        self, args, model_name="model", use_rnn=False, reset_hidden_each_epoch=True
    ):
        super(BeliefDecoderLightning, self).__init__()
        # Initialize BeliefDecoderMLP components
        args["model_type"] = model_name
        self.use_rnn = use_rnn
        self.reset_hidden_each_epoch = reset_hidden_each_epoch

        if self.use_rnn:
            self.decoder = BeliefDecoderRNN(**args)
            self.hidden_state = None
        else:
            self.decoder = BeliefDecoderMLP(**args)

        self.model_name = model_name
        # def count_parameters(model):
        #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
        # # Print the number of parameters
        # print(f'The model has {count_parameters(self.decoder):,} trainable parameters')

    def forward(self, exte, latent):
        if self.use_rnn:
            return self.decoder(exte, latent)[0]
        else:
            return self.decoder(latent)

    def training_step(self, batch, batch_idx):
        latent, priv = batch
        batch_size = latent.shape[0]
        if self.decoder.paradigm != "ordinal":
            self.decoder.priv_normalizer.set_training_mode()
        # self.decoder.priv_normalizer_input.set_training_mode()

        if self.use_rnn:
            # Reset hidden state at the beginning of each epoch if the flag is set
            if (self.reset_hidden_each_epoch) or self.hidden_state is None:
                self.hidden_state = self.decoder.init_hidden(batch_size)
                # print(f"Reset hidden state for RNN:{self.model_name}")

            output, self.hidden_state = self.decoder(latent, self.hidden_state)
            loss = self.decoder.calc_prediction_loss(output, priv)
        else:
            loss = self.decoder.calc_prediction_loss(latent, priv)

        self.log(f"{self.model_name}_train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        latent, priv = batch
        batch_size = latent.shape[0]
        if self.decoder.paradigm != "ordinal":
            self.decoder.priv_normalizer.set_validation_mode()
        # self.decoder.priv_normalizer_input.set_validation_mode()
        if self.use_rnn:
            # Reset hidden state at the beginning of each epoch if the flag is set
            if (self.reset_hidden_each_epoch) or self.hidden_state is None:
                self.hidden_state = self.decoder.init_hidden(batch_size)
                # print(f"Reset hidden state for RNN:{self.model_name}")
            output, self.hidden_state = self.decoder(
                latent, self.hidden_state
            )  # Don't update the hidden state during validation
            loss = self.decoder.calc_prediction_loss(output, priv)
        else:
            loss = self.decoder.calc_prediction_loss(latent, priv)

        self.log(f"{self.model_name}_val_loss", loss)

    def configure_optimizers(self):
        if self.use_rnn:
            lr = 0.001
            weight_decay = 0.00001
        else:
            lr = 0.001
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        return optimizer
