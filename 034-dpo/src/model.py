from transformers import AutoModelForCausalLM


def make_policy(model_name: str):
    """Load a causal LM policy. No extra heads needed for DPO."""
    return AutoModelForCausalLM.from_pretrained(model_name)


def make_reference(model_name: str):
    """Load a frozen reference copy of the policy."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    for p in model.parameters():
        p.requires_grad_(False)
    return model
