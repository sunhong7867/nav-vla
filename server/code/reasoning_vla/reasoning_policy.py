"""SmolVLAPolicy subclass that co-trains a reasoning-text head.

Design (see docs/ver/20260831_2148_reasoning-label-pipeline.md for the label
side):

* The ACTION path is byte-identical to stock SmolVLA. Reasoning is computed
  in a SECOND trunk pass over ``[image | lang | state | reasoning tokens]``
  that shares the (SigLIP-dominated) ``embed_prefix`` output with the action
  pass. The action expert never sees reasoning tokens — no mask surgery, no
  inference-latency change, and a checkpoint from this class still serves
  through the untouched ``predict_action_chunk``.
* Reasoning tokens get att-flag 1 each, which under ``make_att_2d_masks``
  is exactly per-token causal attention on top of the bidirectional
  image/lang prefix block.
* The head is a fresh ``nn.Linear`` initialized from the checkpoint's
  never-used-but-alive ``lm_head`` (SmolVLM2 kept it; smolvla froze it).
  Layer-15-vs-31 mismatch and the trunk's 60k-step drift are exactly what
  the CE fine-tune corrects.
* Loss: ``flow_matching + ce_weight * CE``. Both are logged separately.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy, make_att_2d_masks,
)

try:  # lerobot 0.4.4 keys
    from lerobot.constants import (
        ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE,
    )
except ImportError:  # pragma: no cover - layout moved in other versions
    from lerobot.policies.smolvla.modeling_smolvla import (  # type: ignore
        ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE,
    )

REASONING_KEY = "reasoning"


class ReasoningSmolVLAPolicy(SmolVLAPolicy):
    """SmolVLA + causal reasoning head over the truncated VLM trunk."""

    def __init__(self, config, reasoning_max_length: int = 64,
                 ce_weight: float = 0.1, **kwargs):
        super().__init__(config, **kwargs)
        self.reasoning_max_length = reasoning_max_length
        self.ce_weight = ce_weight

        vlm = self.model.vlm_with_expert.vlm
        text_cfg = vlm.config.text_config
        self.reasoning_head = nn.Linear(
            text_cfg.hidden_size, text_cfg.vocab_size, bias=False)
        # Init from the (frozen, unused) pretrained lm_head so the full
        # vocabulary prior comes for free; a fresh module keeps
        # save/load_pretrained self-contained.
        with torch.no_grad():
            self.reasoning_head.weight.copy_(vlm.lm_head.weight)
        self.reasoning_head.weight.requires_grad = True
        self._tokenizer = None

    # ------------------------------------------------------------- helpers

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.vlm_model_name)
        return self._tokenizer

    def _tokenize_reasoning(self, texts, device):
        """[bos] + tokens + [eos], padded. Returns (ids, attn) LongTensors."""
        tok = self.tokenizer
        bos = tok.bos_token_id if tok.bos_token_id is not None \
            else tok.eos_token_id
        eos = tok.eos_token_id
        rows = []
        for t in texts:
            ids = tok(t or "", add_special_tokens=False)["input_ids"]
            ids = [bos] + ids[: self.reasoning_max_length - 2] + [eos]
            rows.append(ids)
        maxlen = max(len(r) for r in rows)
        ids = torch.full((len(rows), maxlen), eos, dtype=torch.long)
        attn = torch.zeros((len(rows), maxlen), dtype=torch.bool)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            attn[i, : len(r)] = True
        return ids.to(device), attn.to(device)

    def _reasoning_ce(self, prefix_embs, prefix_pad, prefix_att,
                      texts) -> Tensor:
        """Trunk-only causal pass over [prefix | reasoning]; next-token CE."""
        m = self.model
        device = prefix_embs.device
        ids, attn = self._tokenize_reasoning(texts, device)

        r_emb = m.vlm_with_expert.embed_language_tokens(ids)
        # match embed_prefix's language-embedding scaling
        r_emb = r_emb * math.sqrt(r_emb.shape[-1])
        r_emb = r_emb.to(prefix_embs.dtype)

        embs = torch.cat([prefix_embs, r_emb], dim=1)
        pad = torch.cat([prefix_pad, attn], dim=1)
        # flag 1 per reasoning token == causal among themselves, full view
        # of the image/lang/state prefix (make_att_2d_masks semantics)
        att = torch.cat(
            [prefix_att,
             torch.ones_like(attn, dtype=prefix_att.dtype)], dim=1)

        att_2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        # fill_kv_cache=True forces every layer down forward_attn_layer,
        # which is the only path that accepts a None expert stream (the
        # cross-attn path dereferences inputs_embeds[1]); the returned
        # cache is discarded. Same trick sample_actions' prefill relies on.
        (vlm_out, _), _ = m.vlm_with_expert.forward(
            attention_mask=att_2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=True, fill_kv_cache=True)

        hid = vlm_out[:, -ids.shape[1]:].to(torch.float32)
        logits = self.reasoning_head(hid[:, :-1])
        targets = ids[:, 1:].clone()
        targets[~attn[:, 1:]] = -100
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
            ignore_index=-100)

    # -------------------------------------------------------------- train

    def forward(self, batch: dict, noise=None, time=None,
                reduction: str = "mean"):
        """Stock flow-matching loss + reasoning CE, sharing embed_prefix.

        Mirrors SmolVLAPolicy.forward / VLAFlowMatching.forward (lerobot
        0.4.4) except that embed_prefix is called once for both passes.
        """
        texts = batch.get(REASONING_KEY)
        if texts is None:
            return super().forward(batch, noise=noise, time=time,
                                   reduction=reduction)

        m = self.model
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")

        if noise is None:
            noise = m.sample_noise(actions.shape, actions.device)
        if time is None:
            time = m.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad, prefix_att = m.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state)

        # ----- action pass (identical to stock) -----
        suffix_embs, suffix_pad, suffix_att = m.embed_suffix(x_t, time)
        pad = torch.cat([prefix_pad, suffix_pad], dim=1)
        att = torch.cat([prefix_att, suffix_att], dim=1)
        att_2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        (_, suffix_out), _ = m.vlm_with_expert.forward(
            attention_mask=att_2d, position_ids=pos, past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False, fill_kv_cache=False)
        suffix_out = suffix_out[:, -m.config.chunk_size:].to(torch.float32)
        v_t = m.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        if actions_is_pad is not None:
            losses = losses * (~actions_is_pad).unsqueeze(-1)
        losses = losses[:, :, : self.config.max_action_dim]
        action_loss = losses.mean()

        # ----- reasoning pass -----
        ce = self._reasoning_ce(prefix_embs, prefix_pad, prefix_att, texts)

        total = action_loss + self.ce_weight * ce
        loss_dict = {"loss": total.item(),
                     "action_loss": action_loss.item(),
                     "reasoning_ce": ce.item()}
        return total, loss_dict

    # ---------------------------------------------------------- inference

    @torch.no_grad()
    def generate_reasoning(self, batch: dict, max_new_tokens: int = 60,
                           temperature: float = 0.0,
                           repetition_penalty: float = 1.0) -> list[str]:
        # NOTE: repetition_penalty > 1 was measured to HURT factuality on
        # held-out frames (r1 eval: lane 68->18%, speed 69->47% at 1.15,
        # contradiction-aware scoring) — it trades clause repetition for
        # hallucinated variety. Plain greedy is the honest default.
        """Greedy (or sampled) decode of the reasoning head.

        Offline/eval implementation: re-runs the trunk over the growing
        sequence each step (no KV-cache plumbing), so it is O(n^2) and NOT
        for the 10 Hz action path. ~60 tokens is fine for probes and demos
        on a 1 Hz side cadence.
        """
        m = self.model
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix_embs, prefix_pad, prefix_att = m.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state)

        tok = self.tokenizer
        bos = tok.bos_token_id if tok.bos_token_id is not None \
            else tok.eos_token_id
        eos = tok.eos_token_id
        bsize = prefix_embs.shape[0]
        device = prefix_embs.device
        seq = torch.full((bsize, 1), bos, dtype=torch.long, device=device)
        done = torch.zeros(bsize, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            r_emb = m.vlm_with_expert.embed_language_tokens(seq)
            r_emb = (r_emb * math.sqrt(r_emb.shape[-1])
                     ).to(prefix_embs.dtype)
            embs = torch.cat([prefix_embs, r_emb], dim=1)
            ones = torch.ones(bsize, seq.shape[1], dtype=torch.bool,
                              device=device)
            pad = torch.cat([prefix_pad, ones], dim=1)
            att = torch.cat(
                [prefix_att, ones.to(prefix_att.dtype)], dim=1)
            att_2d = make_att_2d_masks(pad, att)
            pos = torch.cumsum(pad, dim=1) - 1
            (vlm_out, _), _ = m.vlm_with_expert.forward(
                attention_mask=att_2d, position_ids=pos,
                past_key_values=None, inputs_embeds=[embs, None],
                use_cache=True, fill_kv_cache=True)
            logits = self.reasoning_head(
                vlm_out[:, -1].to(torch.float32))
            if repetition_penalty and repetition_penalty != 1.0:
                # penalize already-generated tokens (greedy loves to chain
                # "; slowing from 1.4 m/s" clauses without this)
                prev = seq.clamp(min=0)
                gathered = logits.gather(1, prev)
                gathered = torch.where(gathered > 0,
                                       gathered / repetition_penalty,
                                       gathered * repetition_penalty)
                logits = logits.scatter(1, prev, gathered)
            if temperature > 0:
                nxt = torch.multinomial(
                    F.softmax(logits / temperature, dim=-1), 1).squeeze(-1)
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            seq = torch.cat([seq, nxt[:, None]], dim=1)
            done |= nxt == eos
            if bool(done.all()):
                break

        out = []
        for row in seq[:, 1:].tolist():
            if eos in row:
                row = row[: row.index(eos)]
            out.append(tok.decode(row, skip_special_tokens=True).strip())
        return out
