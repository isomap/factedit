from typing import Dict, Tuple, List, Union, Any
from overrides import overrides
import numpy
import torch
import torch.nn as nn

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.nn import InitializerApplicator, util
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, FeedForward
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.attention import AdditiveAttention
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import BLEU
from allennlp.commands.train import train_model

from reader import EditReader


@Model.register("editor")
class Editor(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embed: TextFieldEmbedder,
                 encoder_size: int,
                 decoder_size: int,
                 num_layers: int,
                 beam_size: int,
                 max_decoding_steps: int,
                 use_bleu: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super().__init__(vocab)

        self.START, self.END = self.vocab.get_token_index(START_SYMBOL), self.vocab.get_token_index(END_SYMBOL)
        self.OOV = self.vocab.get_token_index(self.vocab._oov_token)  # pylint: disable=protected-access
        self.PAD = self.vocab.get_token_index(self.vocab._padding_token)  # pylint: disable=protected-access
        self.COPY = self.vocab.get_token_index("@@COPY@@")
        self.KEEP = self.vocab.get_token_index("@@KEEP@@")
        self.DROP = self.vocab.get_token_index("@@DROP@@")

        self.SYMBOL = (self.START, self.END, self.PAD, self.KEEP, self.DROP)
        self.vocab_size = vocab.get_vocab_size()
        self.EMB = embed

        self.emb_size = self.EMB.token_embedder_tokens.output_dim
        self.encoder_size, self.decoder_size = encoder_size, decoder_size
        self.FACT_ENCODER = FeedForward(3 * self.emb_size, 1, encoder_size, nn.Tanh())
        self.ATTN = AdditiveAttention(encoder_size + decoder_size, encoder_size)
        self.COPY_ATTN = AdditiveAttention(decoder_size, encoder_size)
        module = nn.LSTM(self.emb_size, encoder_size // 2, num_layers, bidirectional=True, batch_first=True)
        self.BUFFER = PytorchSeq2SeqWrapper(module)  # BiLSTM to encode draft text
        self.STREAM = nn.LSTMCell(2 * encoder_size, decoder_size)  # Store revised text

        self.BEAM = BeamSearch(self.END, max_steps=max_decoding_steps, beam_size=beam_size)

        self.U = nn.Sequential(nn.Linear(2 * encoder_size, decoder_size), nn.Tanh())
        self.ADD = nn.Sequential(nn.Linear(self.emb_size, encoder_size), nn.Tanh())

        self.P = nn.Sequential(nn.Linear(encoder_size + decoder_size, decoder_size), nn.Tanh())
        self.W = nn.Linear(decoder_size, self.vocab_size)
        self.G = nn.Sequential(nn.Linear(decoder_size, 1), nn.Sigmoid())

        initializer(self)
        self._bleu = BLEU(exclude_indices=set(self.SYMBOL)) if use_bleu else None

    @overrides
    def forward(self,  # type: ignore
                metadata: List[Dict[str, Any]],
                triple_tokens: Dict[str, torch.LongTensor],
                triple_token_ids: torch.Tensor,
                predicate_tokens: Dict[str, torch.Tensor],
                draft_tokens: Dict[str, torch.LongTensor],
                action_tokens: Dict[str, torch.LongTensor] = None,
                revised_tokens: Dict[str, torch.LongTensor] = None,
                action_token_ids: torch.Tensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        state = self._init_state(triple_tokens, predicate_tokens, draft_tokens, triple_token_ids)
        if action_tokens:
            # Initialize Decoder
            state = self._decoder_init(state)
            output_dict = self._forward_loss(action_tokens, action_token_ids, state, **kwargs)
        else:
            output_dict = {}
        output_dict["metadata"] = metadata

        if not self.training:
            # Re-initialize decoder
            state = self._decoder_init(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)

            if revised_tokens and self._bleu:
                top_k_predictions = output_dict["predictions"]
                best_actions = top_k_predictions[:, 0]
                best_predictions = self._action_to_token(best_actions, draft_tokens["tokens"])
                gold_tokens = self._extend_gold_tokens(
                    revised_tokens["tokens"], action_tokens["tokens"], triple_token_ids, action_token_ids)
                self._bleu(best_predictions, gold_tokens)

        return output_dict

    def _extend_gold_tokens(self,
                            revised_tokens: torch.Tensor,
                            action_tokens: torch.Tensor,
                            triple_token_ids: torch.Tensor,
                            action_token_ids: torch.Tensor
                            ):
        batch_size, action_length = action_tokens.size()
        triple_size = triple_token_ids.size(1)
        expanded_triple_ids = triple_token_ids.unsqueeze(1).expand(batch_size, action_length, triple_size)
        expanded_revised_ids = action_token_ids.unsqueeze(-1).expand(batch_size, action_length, triple_size)
        match = expanded_triple_ids == expanded_revised_ids
        copied = match.sum(-1) > 0
        oov = action_tokens == self.OOV
        mask = (oov & copied).long()

        first_match = ((match.cumsum(-1) == 1) * match).byte().argmax(-1)
        new_action_tokens = action_tokens * (1 - mask) + (first_match.long() + self.vocab_size) * mask

        increment_mask = ~(new_action_tokens == self.DROP)
        pointer = revised_tokens.new_zeros((revised_tokens.size(0),))
        end_point = ((revised_tokens != 0).sum(dim=1) - 1)

        for i in range(action_length):
            act_step, mask_step = new_action_tokens[:, i], mask[:, i].bool()
            revised_tokens[mask_step.nonzero().squeeze(1), pointer[mask_step]] = act_step[mask_step]
            pointer[increment_mask[:, i]] += 1
            pointer = torch.min(pointer, end_point)
        return revised_tokens

    def _action_to_token(self,
                         action_tokens: torch.LongTensor,
                         draft_tokens: torch.LongTensor) -> torch.LongTensor:
        predicted_pointer = action_tokens.new_zeros((draft_tokens.size(0), 1))
        draft_pointer = draft_tokens.new_ones((draft_tokens.size(0), 1))

        predicted_tokens = action_tokens.new_full((action_tokens.size()), self.END)

        for act_step in action_tokens.t():
            # KEEP, DELETE, COPY, ADD (other)
            keep_mask = act_step == self.KEEP
            drop_mask = act_step == self.DROP
            add_mask = ~(keep_mask | drop_mask)

            predicted_tokens.scatter_(
                1,
                predicted_pointer,
                draft_tokens.gather(1, draft_pointer)
            )
            predicted_tokens[add_mask] = predicted_tokens[add_mask].scatter(
                1,
                predicted_pointer[add_mask],
                act_step[add_mask].unsqueeze(1)
            )

            draft_pointer[keep_mask | drop_mask] += 1
            predicted_pointer[~drop_mask] += 1
        return predicted_tokens

    def _decoder_init(self, state):
        mean_draft = util.masked_mean(state["encoded_draft"], state["draft_mask"].unsqueeze(-1), 1)
        mean_triple = util.masked_mean(state["encoded_triple"], state["triple_mask"].unsqueeze(-1), 1)
        concatenated = torch.cat((mean_draft, mean_triple), dim=-1)
        batch_size = state["draft_mask"].size(0)

        zeros = mean_draft.new_zeros((batch_size, self.decoder_size))
        state["stream_hidden"], state["stream_context"] = self.U(concatenated), zeros
        state["draft_pointer"] = state["draft_mask"].new_ones((batch_size,))

        action_mask = mean_draft.new_ones((batch_size, self.vocab_size))
        action_mask[:, self.PAD] = 0
        action_mask[:, self.END] = 0

        state["action_mask"] = action_mask

        return state

    def _init_state(self,
                    triples: Dict[str, torch.LongTensor],
                    predicate: Dict[str, torch.LongTensor],
                    draft: Dict[str, torch.LongTensor],
                    triple_ids: torch.LongTensor) -> Dict[str, torch.Tensor]:
        emb_pred = util.masked_mean(self.EMB(predicate),
                                    util.get_text_field_mask(predicate, num_wrapping_dims=1,).unsqueeze(-1), 2)
        emb_triple = self.EMB(triples)
        triple_mask = util.get_text_field_mask(triples)
        flat_triples = torch.cat((emb_triple.flatten(2, 3), emb_pred), dim=-1)

        encoded_triples = self.FACT_ENCODER(flat_triples)

        emb_draft = self.EMB(draft)
        draft_mask = util.get_text_field_mask(draft)
        end_point = (draft_mask.sum(dim=1) - 1)
        encoded_draft = self.BUFFER(emb_draft, draft_mask)

        return {"draft_mask": draft_mask, "triple_mask": triple_mask, "end_point": end_point,
                "encoded_triple": encoded_triples, "encoded_draft": encoded_draft,
                "triple_tokens": triples["tokens"][:, :, -1], "triple_token_ids": triple_ids}

    def _forward_loss(self,
                      target_actions: Dict[str, torch.LongTensor],
                      target_token_ids: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, target_sequence_length = target_actions["tokens"].size()
        num_decoding_steps = target_sequence_length - 1

        target_to_triple = state["triple_mask"].new_zeros(state["triple_mask"].size()).bool()
        copy_input_choice = state["triple_mask"].new_full((batch_size,), self.COPY)

        step_log_likelihoods = []
        for t in range(num_decoding_steps):
            input_actions = target_actions["tokens"][:, t]
            if t < num_decoding_steps - 1:
                copied = (target_to_triple.sum(dim=-1) > 0) & (input_actions == self.OOV)
                target_to_triple = state["triple_token_ids"] == target_token_ids[:, t + 1].unsqueeze(-1)
                input_actions = copied.long() * (copy_input_choice - input_actions) + input_actions

            state = self._decoder_step(input_actions, state)
            step_target_actions = target_actions["tokens"][:, t + 1]
            step_log_likelihoods.append(
                self._get_log_likelihood(state, step_target_actions, target_to_triple)
            )

        log_likelihoods = torch.stack(step_log_likelihoods, dim=-1)
        target_mask = util.get_text_field_mask(target_actions)
        target_mask = target_mask[:, 1:].float()

        log_likelihood = (log_likelihoods * target_mask).sum(dim=-1)
        loss = - log_likelihood.sum()
        loss /= batch_size

        return {"loss": loss}

    @staticmethod
    def _get_query(state: Dict[str, torch.Tensor]):
        batch_size = state["encoded_draft"].size(0)
        buffer_head = state["encoded_draft"][torch.arange(batch_size), state["draft_pointer"]]

        query = torch.cat([buffer_head, state["stream_hidden"]], dim=1)
        return query

    def _get_log_likelihood(self,
                            state: Dict[str, torch.Tensor],
                            target_actions: torch.Tensor,
                            target_to_source: torch.Tensor) -> torch.Tensor:
        hidden = self.P(self._get_query(state))
        gate_prob = self.G(hidden).squeeze(1)

        gen_prob = util.masked_softmax(self.W(hidden), state["action_mask"], memory_efficient=True)\
            .gather(1, target_actions.unsqueeze(1)).squeeze(1)
        gen_mask = (target_actions != self.OOV) | (target_to_source.sum(dim=-1) == 0)
        gen_prob = gen_prob.min(gen_mask.float())

        copy_prob = self.COPY_ATTN(hidden, state["encoded_triple"], state["triple_mask"])\
            .masked_fill(~target_to_source, 0.).sum(dim=-1)

        step_prob = gen_prob * gate_prob + copy_prob * (- gate_prob + 1)
        step_log_likelihood = step_prob.clamp(1e-30).log()

        return step_log_likelihood

    def _decoder_step(self,
                      last_actions: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embed_actions = self.EMB({"tokens": last_actions})
        batch_size = embed_actions.size(0)
        # Update stack given draft pointer information

        draft_head = state["encoded_draft"][torch.arange(batch_size), state["draft_pointer"]]
        query = torch.cat([state["stream_hidden"], draft_head], dim=1)
        attend = self.ATTN(query, state["encoded_triple"], state["triple_mask"])
        attended_triple = util.weighted_sum(state["encoded_triple"], attend)

        is_added = torch.stack([last_actions != tok for tok in self.SYMBOL]).all(dim=0)
        draft_head[is_added] = self.ADD(embed_actions[is_added])

        hs, cs = self.STREAM(torch.cat((draft_head, attended_triple), dim=-1),
                             (state["stream_hidden"], state["stream_context"]))
        drop_mask = (last_actions != self.DROP).unsqueeze(1).float()
        hx = drop_mask * hs + (- drop_mask + 1) * state["stream_hidden"]
        cx = drop_mask * cs + (- drop_mask + 1) * state["stream_context"]
        state["stream_hidden"], state["stream_context"] = hx, cx

        # Update Pointer
        move_forward = ((last_actions == self.KEEP) | (last_actions == self.DROP)).long()

        state["draft_pointer"] = state["draft_pointer"] + move_forward
        # Simple masking for pointer
        state["draft_pointer"] = torch.min(state["draft_pointer"], state["end_point"])

        is_ended = state["end_point"] == state["draft_pointer"]
        state["action_mask"][is_ended, self.KEEP] = 0
        state["action_mask"][is_ended, self.DROP] = 0
        state["action_mask"][is_ended, self.END] = 1

        return state

    def take_search_step(self,
                         last_predictions: torch.Tensor,
                         state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        input_choices = self._get_input(last_predictions)
        state = self._decoder_step(input_choices, state)
        final_prob = self._make_prob(state)

        return final_prob.clamp(1e-30).log(), state

    def _get_input(self,
                   last_predictions: torch.Tensor,
                   ) -> torch.Tensor:
        group_size, = last_predictions.size()
        only_copy_mask = (last_predictions >= self.vocab_size).long()
        copy_input_choices = only_copy_mask.new_full((group_size,), self.COPY)
        input_choices = (copy_input_choices - last_predictions) * only_copy_mask + last_predictions
        return input_choices

    def _make_prob(self,
                   state: Dict[str, torch.Tensor]) -> torch.Tensor:

        triple_token_ids = state["triple_token_ids"]
        batch_size, triple_length = triple_token_ids.size()

        hidden = self.P(self._get_query(state))

        gate_prob = self.G(hidden)
        gen_prob = util.masked_softmax(self.W(hidden), state["action_mask"], memory_efficient=True) * gate_prob

        copy_prob = self.COPY_ATTN(hidden, state["encoded_triple"], state["triple_mask"]) * (- gate_prob + 1)
        modified_prob_list: List[torch.Tensor] = []
        for i in range(triple_length):
            copy_prob_slice = copy_prob[:, i]
            token_slice = state["triple_tokens"][:, i]
            copy_to_add_mask = token_slice != self.OOV
            copy_to_add = copy_prob_slice.min(copy_to_add_mask.float()).unsqueeze(-1)
            gen_prob = gen_prob.scatter_add(-1, token_slice.unsqueeze(1), copy_to_add)

            if i < (triple_length - 1):
                future_occurrences = ((triple_token_ids[:, i + 1:]) == triple_token_ids[:, i].unsqueeze(-1)).float()
                future_copy_prob = copy_prob[:, i + 1:].min(future_occurrences)
                copy_prob_slice += future_copy_prob.sum(-1)

            if i > 0:
                prev_occurrences = triple_token_ids[:, :i] == triple_token_ids[:, i].unsqueeze(-1)
                duplicate_mask = (prev_occurrences.sum(-1) == 0).float()
                copy_prob_slice = copy_prob_slice.min(duplicate_mask)

            left_over_copy_prob = copy_prob_slice.min((~copy_to_add_mask).float())
            modified_prob_list.append(left_over_copy_prob.unsqueeze(-1))

        modified_prob_list.insert(0, gen_prob)
        modified_prob = torch.cat(modified_prob_list, dim=-1)
        return modified_prob

    def _forward_beam_search(self,
                             state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size = state["draft_mask"].size(0)
        start_predictions = state["draft_mask"].new_full((batch_size,), self.START)
        all_top_k_predictions, log_probabilities = self.BEAM.search(start_predictions, state, self.take_search_step)
        return {
            "predicted_log_probs": log_probabilities,
            "predictions": all_top_k_predictions
        }

    def _get_predicted_tokens(self,
                              predicted_indices: Union[torch.Tensor, numpy.ndarray],
                              batch_metadata,
                              n_best: int = None):
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        predicted_tokens = []

        for top_k_predictions, metadata in zip(predicted_indices, batch_metadata):
            batch_predicted_tokens = []
            draft, triple = metadata['draft'], metadata["triple"]
            for indices in top_k_predictions[:n_best]:
                pointer, tokens = 0, []
                indices = list(indices)
                if self.END in indices:
                    indices = indices[:indices.index(self.END)]
                for index in indices:
                    if index == self.KEEP:
                        tokens.append(draft[pointer])
                        pointer += 1
                    elif index == self.DROP:
                        pointer += 1
                    elif index >= self.vocab_size:
                        adjusted_index = index - self.vocab_size
                        tokens.append(triple[adjusted_index])
                    else:
                        tokens.append(str(self.vocab.get_token_from_index(index)))
                batch_predicted_tokens.append(tokens)
            if n_best == 1:
                predicted_tokens.append(batch_predicted_tokens[0])
            else:
                predicted_tokens.append(batch_predicted_tokens)
        return predicted_tokens

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        predicted_tokens = self._get_predicted_tokens(output_dict["predictions"],
                                                      output_dict["metadata"])
        output_dict["predicted_tokens"] = predicted_tokens
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics
