# src/model.py
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_size, config):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=config.N_HIDDEN_ENCODER, num_layers=1)
        self.attn_linear = nn.Sequential(nn.Linear(in_features=2 * config.N_HIDDEN_ENCODER, out_features=config.N_HIDDEN_ENCODER), nn.Tanh(), nn.Linear(config.N_HIDDEN_ENCODER, input_size))
    def forward(self, X_input, device):
        h_prev = torch.zeros(1, X_input.size(0), self.lstm.hidden_size, device=device); s_prev = torch.zeros(1, X_input.size(0), self.lstm.hidden_size, device=device); encoder_hidden_states = torch.zeros(X_input.size(0), X_input.size(1), self.lstm.hidden_size, device=device); all_alpha_t = torch.zeros(X_input.size(0), X_input.size(1), X_input.size(2), device=device)
        for t in range(X_input.size(1)):
            h_s_t_minus_1 = torch.cat((h_prev.squeeze(0), s_prev.squeeze(0)), dim=1); alpha_t = torch.softmax(self.attn_linear(h_s_t_minus_1), dim=1); all_alpha_t[:, t, :] = alpha_t; weighted_input = torch.mul(alpha_t, X_input[:, t, :]); self.lstm.flatten_parameters(); _, (h_prev, s_prev) = self.lstm(weighted_input.unsqueeze(0), (h_prev, s_prev)); encoder_hidden_states[:, t, :] = h_prev.squeeze(0)
        return encoder_hidden_states, all_alpha_t

class DecoderBody(nn.Module):
    def __init__(self, config):
        super(DecoderBody, self).__init__()
        self.attn_layer = nn.Sequential(nn.Linear(config.N_HIDDEN_DECODER + config.N_HIDDEN_ENCODER, config.N_HIDDEN_ENCODER), nn.Tanh(), nn.Linear(config.N_HIDDEN_ENCODER, 1))
        self.lstm = nn.LSTM(input_size=1, hidden_size=config.N_HIDDEN_DECODER)
    def forward(self, encoder_hidden_states, y_history, device):
        d_prev = torch.zeros(1, y_history.size(0), self.lstm.hidden_size, device=device); s_prev = torch.zeros(1, y_history.size(0), self.lstm.hidden_size, device=device)
        for t in range(y_history.size(1) - 1):
            h_prev_decoder_repeated = d_prev.squeeze(0).unsqueeze(1).repeat(1, encoder_hidden_states.size(1), 1); x = torch.cat((h_prev_decoder_repeated, encoder_hidden_states), dim=2); beta = torch.softmax(self.attn_layer(x).squeeze(2), dim=1).unsqueeze(1); context_vector = torch.bmm(beta, encoder_hidden_states).squeeze(1); lstm_input_t = y_history[:, t].unsqueeze(0); self.lstm.flatten_parameters(); _, (d_prev, s_prev) = self.lstm(lstm_input_t, (d_prev, s_prev))
        h_prev_decoder_repeated = d_prev.squeeze(0).unsqueeze(1).repeat(1, encoder_hidden_states.size(1), 1); x = torch.cat((h_prev_decoder_repeated, encoder_hidden_states), dim=2); beta_final = torch.softmax(self.attn_layer(x).squeeze(2), dim=1).unsqueeze(1); context_vector = torch.bmm(beta_final, encoder_hidden_states).squeeze(1)
        return torch.cat((d_prev.squeeze(0), context_vector), dim=1), beta_final.squeeze(1)

class DA_RNN_MultiTask(nn.Module):
    def __init__(self, config, feature_list):
        super(DA_RNN_MultiTask, self).__init__()
        self.config = config
        self.feature_list = feature_list
        self.input_size = len(feature_list)
        self.encoder = Encoder(self.input_size, config).to(config.DEVICE)
        self.decoder_body = DecoderBody(config).to(config.DEVICE)
        self.shared_layer = nn.Sequential(nn.LayerNorm(config.N_HIDDEN_DECODER + config.N_HIDDEN_ENCODER), nn.Dropout(config.DROPOUT_RATE)).to(config.DEVICE)
        self.regression_head = nn.Linear(config.N_HIDDEN_DECODER + config.N_HIDDEN_ENCODER, 1).to(config.DEVICE)
        self.classification_head = nn.Linear(config.N_HIDDEN_DECODER + config.N_HIDDEN_ENCODER, config.NUM_CLASSES).to(config.DEVICE)

    def forward(self, X, return_attentions=False):
        close_pct_idx = self.feature_list.index('close_pct_change')
        y_history = X[:, :, [close_pct_idx]]
        encoder_hidden_states, input_attentions = self.encoder(X, self.config.DEVICE)
        decoder_output, temporal_attentions = self.decoder_body(encoder_hidden_states, y_history, self.config.DEVICE)
        shared_output = self.shared_layer(decoder_output)
        pct_change_pred = self.regression_head(shared_output)
        trend_logits = self.classification_head(shared_output)
        if return_attentions:
            return pct_change_pred, trend_logits, input_attentions, temporal_attentions
        return pct_change_pred, trend_logits