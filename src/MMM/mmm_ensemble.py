import torch
from typing import List, Tuple


def aggregate_trade_decision(probabilities: List[torch.Tensor], reduce: str = "mean") -> torch.Tensor:
    """
    probabilities: list of tensors with shape [B, pred_len, 3] or [B, 3]
    Returns: [B, 3] aggregated decision over pred_len and ensemble
    """
    if not probabilities:
        raise ValueError("probabilities is empty")

    # Align shapes to [B, 3]
    aligned = []
    for p in probabilities:
        if p.dim() == 3:
            p2 = p.mean(dim=1)  # over pred_len
        elif p.dim() == 2:
            p2 = p
        else:
            raise ValueError(f"Unsupported probs shape: {tuple(p.shape)}")
        aligned.append(p2)

    stack = torch.stack(aligned, dim=0)  # [E, B, 3]
    if reduce == "mean":
        agg = stack.mean(dim=0)
    elif reduce == "median":
        agg = stack.median(dim=0).values
    else:
        raise ValueError("reduce must be 'mean' or 'median'")
    return agg


def build_final_order(
    trade_probs_agg: torch.Tensor,  # [B, 3] hold, buy, sell
    volume_pct: torch.Tensor,       # [B, 1] 0..100
    risk_score: torch.Tensor,       # [B, 1] 0..1
    price_now: torch.Tensor | None = None,
    risk_threshold: float = 0.2,
    min_volume_pct: float = 0.5,
) -> List[dict]:
    """
    Produces a simple order plan per batch element. If hold or risk too high -> skip.
    Returns list of dicts: {type: 'buy'|'sell'|'hold', volume_pct: float, risk: float}
    """
    orders: List[dict] = []
    probs = trade_probs_agg
    hold_buy_sell = probs.argmax(dim=-1)  # 0,1,2
    for i in range(probs.size(0)):
        decision = hold_buy_sell[i].item()
        vol = float(volume_pct[i].item()) if volume_pct is not None else 0.0
        risk = float(risk_score[i].item()) if risk_score is not None else 0.0
        if decision == 0:
            orders.append({"type": "hold", "volume_pct": 0.0, "risk": risk})
            continue
        if risk >= risk_threshold or vol < min_volume_pct:
            orders.append({"type": "hold", "volume_pct": 0.0, "risk": risk})
            continue
        side = "buy" if decision == 1 else "sell"
        orders.append({"type": side, "volume_pct": vol, "risk": risk})
    return orders


