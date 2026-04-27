from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class HallucinationReport:
    """
    Result of a HallScan analysis.
    Returned by hallscan.scan()
    """
    # The prompt that was analyzed
    prompt: str
    
    # What the model predicted
    predicted: str
    
    # Top 10 predictions with probabilities
    top10: List[Tuple[str, float]]
    
    # Hallucination type
    # "CORRECT" / "TYPE1_DROPOUT" / "TYPE2A_SUPPRESSION" / "TYPE2B_GAP"
    hallucination_type: str
    
    # Is the prediction correct?
    is_correct: bool
    
    # Rank of correct answer in top 10 (None if not found)
    correct_answer_rank: Optional[int]
    
    # Attention to relation word in final block
    relation_attention: float
    
    # Layer where factual knowledge peaks (blocks 10-11 in GPT-2)
    peak_factual_layer: Optional[int]
    
    # Layer where suppression occurs
    suppression_layer: Optional[int]
    
    # Confidence score: how likely is this a hallucination?
    # 0.0 = definitely correct, 1.0 = definitely hallucinating
    hallucination_risk: float

    def __str__(self):
        lines = [
            f"",
            f"  HallScan Report",
            f"  {'─' * 40}",
            f"  Prompt:     '{self.prompt}'",
            f"  Predicted:  '{self.predicted}'",
            f"  Type:       {self.hallucination_type}",
            f"  Risk:       {self.hallucination_risk:.0%}",
        ]
        if self.correct_answer_rank:
            lines.append(
                f"  Correct answer at rank: {self.correct_answer_rank}"
            )
        if self.suppression_layer:
            lines.append(
                f"  Suppression at block:   {self.suppression_layer}"
            )
        if self.peak_factual_layer:
            lines.append(
                f"  Factual peak at block:  {self.peak_factual_layer}"
            )
        lines.append(f"  {'─' * 40}")
        return "\n".join(lines)