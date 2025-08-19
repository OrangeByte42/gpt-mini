import torch
from typing import Any

from src.data.bpe_tokenizer import BpeTokenizer
from src.models.gpt_mini import GPTMini


class AutoregressiveGPTMini:
    """A class for training and validating the GPT-Mini model in an autoregressive manner."""

    def __init__(self: Any, model: GPTMini, tokenizer: BpeTokenizer, device: torch.device) -> None:
        """Initialize the AutoregressiveGPTMini class with a model and tokenizer."""
        self.model: GPTMini = model
        self.tokenizer: BpeTokenizer = tokenizer
        self.device: torch.device = device

    def generate(self: Any, input_ids: torch.Tensor, max_seq_len: int) -> torch.Tensor:
        """Generate text using the GPT-Mini model."""
        # Necessary variables
        sos_token_id: int = self.tokenizer.sos_id
        eos_token_id: int = self.tokenizer.eos_id
        pad_token_id: int = self.tokenizer.pad_id

        batch_size: int = input_ids.shape[0]
        generated_seqs: torch.Tensor = torch.empty((batch_size, 0), device=self.device)

        # Track which sequences have finished generation (generated EOS token)
        finished: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Autoregressive generation loop
        for _ in range(max_seq_len - 1):    # -1 because we already have SOS token
            # If all sequences are finished, break early
            if torch.all(finished): break

            # Create mask for current generated sequences
            current_input: torch.Tensor = torch.cat([input_ids, generated_seqs], dim=1)
            mask: torch.Tensor = self.model.make_mask(current_input)

            # Forward pass through the model
            logits: torch.Tensor = self.arch(current_input, mask)

            # Greedy decoding: select the token with the highest probability
            next_token_logits: torch.Tensor = logits[:, -1, :]
            next_token_ids: torch.Tensor = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append the next token to the generated sequences
            generated_seqs = torch.cat([generated_seqs, next_token_ids], dim=1)

            # Update finished sequences for sequences that generated EOS token
            finished = finished | (next_token_ids.squeeze(-1) == eos_token_id)

        # Pad sequences to max_seq_len if necessary
        padded_seqs: torch.Tensor = torch.full((batch_size, max_seq_len), pad_token_id, device=self.device)
        padded_seqs[:, :generated_seqs.shape[1]] = generated_seqs
        generated_seqs = padded_seqs

        # Return the generated sequences
        return generated_seqs

