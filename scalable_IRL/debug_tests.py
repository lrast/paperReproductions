# test to catch various failure modes

import torch
import torch.nn.functional as F


def data_leakage_to_labels(model, batch):
    """
        Ensure that the hugging face model is actually performing a 'token shift'
        predicting the future rather than the current token
    """
    with torch.no_grad():
        model_outs = model(**batch)

        # huggingface loss
        hf_loss = model_outs.loss
        logits = model_outs.logits  # Shape: [batch, seq_len, vocab_size]

        # --- THE MANUAL SHIFT ---
        # Shift so that tokens < n predict token n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch['labels'][..., 1:].contiguous()
        
        # Calculate CrossEntropy manually
        manual_loss = F.cross_entropy(
            # HF loss path returns fp32 here (on mps/bf16); compute manual in fp32 too.
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

    # 5. The Moment of Truth
    print(f"Hugging Face Internal Loss: {hf_loss.item():.6f}")
    print(f"Manual Shifted Loss:       {manual_loss.item():.6f}")
    print(f"Difference:                {abs(hf_loss.item() - manual_loss.item()):.10f}")

    assert torch.allclose(hf_loss, manual_loss), "Losses do not match! The shift is wrong."
    print("\nSuccess! The model is definitely shifting the labels for you.")


def masked_labels_match_ignore_index(model, batch):
    """
    Ensure that HF loss (ignore_index=-100 on shifted labels) matches cross-entropy
    computed only on positions where the shifted label is not -100.
    """
    with torch.no_grad():
        model_outs = model(**batch)

        hf_loss = model_outs.loss
        logits = model_outs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()

        flat_logits = shift_logits.float().view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        valid = flat_labels != -100

        assert valid.any(), "No non-masked labels after shift; cannot compare losses."

        manual_loss = F.cross_entropy(
            flat_logits[valid],
            flat_labels[valid],
        )

    print(f"Hugging Face Loss (ignore_index): {hf_loss.item():.6f}")
    print(f"Manual filtered (-100 removed):   {manual_loss.item():.6f}")
    print(f"Difference:                       {abs(hf_loss.item() - manual_loss.item()):.10f}")

    assert torch.allclose(hf_loss, manual_loss), (
        "Losses do not match! Masked label handling differs from manual filtering."
    )
    print("\nSuccess! ignore_index=-100 matches filtering out -100 positions.")
