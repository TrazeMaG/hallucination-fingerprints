import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .report import HallucinationReport
from typing import Optional

# Cache loaded models so we don't reload every scan
_model_cache = {}

def _load_model(model_name: str):
    """Load and cache a model"""
    if model_name not in _model_cache:
        print(f"Loading {model_name}...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(
            model_name,
            output_attentions=True,
            output_hidden_states=True
        )
        model.eval()
        _model_cache[model_name] = (model, tokenizer)
        print(f"{model_name} loaded.")
    return _model_cache[model_name]


def scan(
    prompt: str,
    relation_word: Optional[str] = None,
    correct_answer: Optional[str] = None,
    model: str = "gpt2"
) -> HallucinationReport:
    """
    Scan a prompt for hallucination risk.
    
    Args:
        prompt: The text prompt to analyze
        relation_word: The relation word to track (e.g. 'capital')
                      If None, auto-detected from prompt
        correct_answer: The known correct answer (optional)
        model: HuggingFace model name (default: 'gpt2')
    
    Returns:
        HallucinationReport with full analysis
    
    Example:
        from hallscan import scan
        result = scan("The capital of France is", 
                     relation_word="capital",
                     correct_answer="Paris")
        print(result)
    """
    gpt_model, tokenizer = _load_model(model)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    last_pos = len(tokens) - 1
    
    # Run model
    with torch.no_grad():
        outputs = gpt_model(input_ids)
    
    # ── Top 10 predictions ──
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    top10_vals, top10_idx = probs.topk(10)
    top10 = [
        (tokenizer.decode([idx.item()]).strip(), val.item())
        for idx, val in zip(top10_idx, top10_vals)
    ]
    predicted = top10[0][0]
    
    # ── Correct answer rank ──
    correct_rank = None
    if correct_answer:
        for rank, (token, prob) in enumerate(top10, 1):
            if correct_answer.lower() in token.lower():
                correct_rank = rank
                break
    
    is_correct = False
    if correct_answer:
        is_correct = correct_answer.lower() in predicted.lower()
    
    # ── Auto-detect relation word if not provided ──
    if relation_word is None:
        common_relations = [
            'capital', 'president', 'invented', 'written', 
            'discovered', 'founded', 'born', 'died', 'orbits'
        ]
        for word in common_relations:
            if word in prompt.lower():
                relation_word = word
                break
    
    # ── Relation attention in final block ──
    relation_pos = None
    if relation_word:
        for i, tok in enumerate(tokens):
            if relation_word.lower() in tok.lower():
                relation_pos = i
                break
    
    final_attn = outputs.attentions[-1]
    relation_attention = 0.0
    if relation_pos is not None:
        num_heads = final_attn.shape[1]
        for h in range(num_heads):
            relation_attention += final_attn[0, h, last_pos, relation_pos].item()
        relation_attention /= num_heads
    
    # ── Layer analysis: find peak factual layer + suppression ──
    hidden_states = outputs.hidden_states
    ln_f = gpt_model.transformer.ln_f
    lm_head = gpt_model.lm_head
    
    peak_factual_layer = None
    suppression_layer = None
    peak_prob = 0.0
    
    if correct_answer:
        correct_ids = tokenizer.encode(" " + correct_answer)
        correct_id = correct_ids[0] if correct_ids else None
        
        if correct_id:
            prev_correct_prob = 0
            for layer_idx, hidden in enumerate(hidden_states):
                normed = ln_f(hidden[0, last_pos, :])
                layer_logits = lm_head(normed)
                layer_probs = torch.softmax(layer_logits, dim=-1)
                correct_prob = layer_probs[correct_id].item()
                
                # Track peak
                if correct_prob > peak_prob:
                    peak_prob = correct_prob
                    peak_factual_layer = layer_idx
                
                # Detect suppression: correct prob drops significantly
                if (prev_correct_prob > 0.05 and 
                    correct_prob < prev_correct_prob * 0.3 and
                    layer_idx > 1):
                    suppression_layer = layer_idx
                
                prev_correct_prob = correct_prob
    
    # ── Classify hallucination type ──
    if is_correct:
        h_type = "CORRECT"
        risk = 0.1
    elif correct_rank and correct_rank <= 10:
        if suppression_layer and suppression_layer >= 10:
            h_type = "TYPE2A_LAST_LAYER_SUPPRESSION"
            risk = 0.85
        else:
            h_type = "TYPE2A_SUPPRESSION"
            risk = 0.75
    elif relation_attention < 0.05:
        h_type = "TYPE1_RELATION_DROPOUT"
        risk = 0.90
    else:
        h_type = "TYPE2B_KNOWLEDGE_GAP"
        risk = 0.70
    
    return HallucinationReport(
        prompt=prompt,
        predicted=predicted,
        top10=top10,
        hallucination_type=h_type,
        is_correct=is_correct,
        correct_answer_rank=correct_rank,
        relation_attention=relation_attention,
        peak_factual_layer=peak_factual_layer,
        suppression_layer=suppression_layer,
        hallucination_risk=risk
    )