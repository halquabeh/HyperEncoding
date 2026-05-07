import torch
import torch.nn as nn

from attacks.base import Attack


class Retiming(Attack):
    """First-order spike retiming attack on already encoded inputs.

    Inputs and outputs use this repo's encoded convention: [T*B, C, H, W].
    The attack temporarily evaluates the model with internal encoding disabled.
    """

    def __init__(
        self,
        model,
        device,
        T,
        norm="linf",
        eps=1,
        steps=1,
        max_candidates=200000,
        candidate_multiplier=4,
        **kwargs,
    ):
        super().__init__("Retiming", model)
        norm = norm.lower()
        if norm in ("0", "l0"):
            norm = "l0"
        elif norm in ("1", "l1"):
            norm = "l1"
        elif norm in ("inf", "infty", "linf", "l_inf"):
            norm = "linf"
        else:
            raise ValueError(f"unsupported retiming norm: {norm}")

        self.T = int(T)
        self.norm = norm
        self.eps = int(eps)
        self.steps = int(steps)
        self.max_candidates = int(max_candidates)
        self.candidate_multiplier = int(candidate_multiplier)
        self.loss_fn = nn.CrossEntropyLoss()
        self.supported_mode = ["default", "targeted"]
        self.set_device(device)

    def _encoding_state(self):
        state = {}
        for attr in ("encode", "model_encode"):
            if hasattr(self.model, attr):
                state[attr] = getattr(self.model, attr)
                setattr(self.model, attr, False)
        return state

    def _restore_encoding_state(self, state):
        for attr, value in state.items():
            setattr(self.model, attr, value)

    def _as_time_major(self, images):
        if self.T <= 0 or images.shape[0] % self.T != 0:
            raise ValueError(f"expected encoded [T*B,C,H,W] input for T={self.T}, got {images.shape}")
        batch = images.shape[0] // self.T
        return images.view(self.T, batch, *images.shape[1:]).contiguous()

    def _logits(self, x_tbchw):
        out = self.model(x_tbchw.flatten(0, 1).contiguous())
        if out.dim() == 3:
            return out.mean(0)
        if out.dim() == 2:
            return out
        raise ValueError(f"unexpected model output shape: {out.shape}")

    def _loss_grad(self, x_tbchw, labels):
        x_var = x_tbchw.detach().clone().requires_grad_(True)
        logits = self._logits(x_var)
        if self.targeted:
            target_labels = self._get_target_label(x_var, labels)
            loss = -self.loss_fn(logits, target_labels)
        else:
            loss = self.loss_fn(logits, labels)
        grad = torch.autograd.grad(loss, x_var, retain_graph=False, create_graph=False)[0]
        return grad.detach()

    def _candidate_cap(self, remaining, pair_count, line_count):
        if self.norm == "linf":
            per_pair = max(1, self.max_candidates // max(1, pair_count))
            return min(line_count, per_pair)
        budget_cap = max(1, remaining) * max(1, self.candidate_multiplier)
        return min(line_count, max(1024, min(self.max_candidates, budget_cap)))

    def _greedy_step(self, adv, origin, grad, total_cost):
        adv = adv.contiguous()
        origin = origin.contiguous()
        T = adv.shape[0]
        adv_flat = adv.view(T, -1)
        grad_flat = grad.reshape(T, -1)
        origin_flat = origin.view(T, -1)

        occupied = origin_flat >= 0
        line_count = adv_flat.shape[1]
        remaining = max(0, self.eps - total_cost) if self.norm in ("l0", "l1") else self.eps

        pairs = []
        for s in range(T):
            for t in range(T):
                if s == t:
                    continue
                if self.norm == "linf" and abs(t - s) > self.eps:
                    continue
                pairs.append((s, t))

        cand_scores = []
        cand_sources = []
        cand_targets = []
        cand_lines = []
        cand_increments = []

        neg_inf = torch.finfo(adv.dtype).min
        for s, t in pairs:
            src_origin = origin_flat[s]
            src_mask = src_origin >= 0
            target_free = ~occupied[t]
            if self.norm == "linf":
                valid_budget = (t - src_origin).abs() <= self.eps
                increment = torch.zeros_like(src_origin)
            elif self.norm == "l1":
                old_cost = (s - src_origin).abs()
                new_cost = (t - src_origin).abs()
                increment = new_cost - old_cost
                valid_budget = total_cost + increment <= self.eps
            else:
                old_cost = (s != src_origin).long()
                new_cost = (t != src_origin).long()
                increment = new_cost - old_cost
                valid_budget = total_cost + increment <= self.eps

            score = adv_flat[s] * (grad_flat[t] - grad_flat[s])
            valid = src_mask & target_free & valid_budget & (score > 0)
            if not valid.any():
                continue

            k = self._candidate_cap(remaining, len(pairs), line_count)
            masked_score = torch.where(valid, score, torch.as_tensor(neg_inf, device=adv.device, dtype=adv.dtype))
            vals, idx = torch.topk(masked_score, k=k)
            keep = vals > 0
            if not keep.any():
                continue

            idx = idx[keep]
            cand_scores.append(vals[keep])
            cand_sources.append(torch.full_like(idx, s))
            cand_targets.append(torch.full_like(idx, t))
            cand_lines.append(idx)
            cand_increments.append(increment[idx])

        if not cand_scores:
            return adv, origin, total_cost, 0

        scores = torch.cat(cand_scores)
        sources = torch.cat(cand_sources)
        targets = torch.cat(cand_targets)
        lines = torch.cat(cand_lines)
        increments = torch.cat(cand_increments)

        if scores.numel() > self.max_candidates:
            scores, keep = torch.topk(scores, self.max_candidates)
            sources = sources[keep]
            targets = targets[keep]
            lines = lines[keep]
            increments = increments[keep]

        order = torch.argsort(scores, descending=True)
        used_source = torch.zeros_like(occupied)
        moved = 0

        for pos in order.tolist():
            s = int(sources[pos])
            t = int(targets[pos])
            line = int(lines[pos])
            inc = int(increments[pos])
            src_origin = int(origin_flat[s, line])

            if src_origin < 0 or used_source[s, line] or occupied[t, line]:
                continue
            if self.norm == "linf" and abs(t - src_origin) > self.eps:
                continue
            if self.norm in ("l0", "l1") and total_cost + inc > self.eps:
                continue

            adv_flat[t, line] = adv_flat[s, line]
            adv_flat[s, line] = 0
            origin_flat[t, line] = src_origin
            origin_flat[s, line] = -1
            occupied[t, line] = True
            occupied[s, line] = False
            used_source[s, line] = True
            if self.norm in ("l0", "l1"):
                total_cost += inc
            moved += 1

        return adv_flat.view_as(adv), origin_flat.view_as(origin), total_cost, moved

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        x_tbchw = self._as_time_major(images)
        adv = x_tbchw.clone()
        origin = torch.arange(self.T, device=adv.device).view(self.T, 1, 1, 1, 1)
        origin = origin.expand_as(adv).clone().long()
        origin[adv == 0] = -1

        state = self._encoding_state()
        try:
            total_cost = 0
            for _ in range(max(1, self.steps)):
                grad = self._loss_grad(adv, labels)
                next_adv, next_origin, total_cost, moved = self._greedy_step(adv, origin, grad, total_cost)
                if moved == 0:
                    break
                adv, origin = next_adv.detach(), next_origin.detach()
        finally:
            self._restore_encoding_state(state)

        return adv.flatten(0, 1).contiguous().detach()
