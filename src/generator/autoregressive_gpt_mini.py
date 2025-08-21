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

        input_ids: torch.Tensor = input_ids.to(self.device)

        batch_size: int = input_ids.shape[0]
        generated_seqs: torch.Tensor = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)

        # Track which sequences have finished generation (generated EOS token)
        finished: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Calculate maximum generation steps based on input length and max_seq_len
        input_length: int = input_ids.shape[1]
        max_generation_steps: int = max_seq_len - input_length - 1  # -1 for safety margin

        # Autoregressive generation loop
        for step in range(max_generation_steps):
            # If all sequences are finished, break early
            if torch.all(finished): break

            # Create mask for current generated sequences
            current_input: torch.Tensor = torch.cat([input_ids, generated_seqs], dim=1)

            # Ensure we don't exceed max_seq_len
            if current_input.shape[1] >= max_seq_len:
                break

            mask: torch.Tensor = self.model.make_mask(current_input)

            # Forward pass through the model
            logits: torch.Tensor = self.model.arch(current_input, mask)

            # Greedy decoding: select the token with the highest probability
            next_token_logits: torch.Tensor = logits[:, -1, :]
            next_token_ids: torch.Tensor = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Mark sequences that generated EOS token as finished
            eos_generated: torch.Tensor = (next_token_ids.squeeze(-1) == eos_token_id)
            finished = finished | eos_generated

            # For finished sequences, replace the generated token with PAD token
            next_token_ids[finished.unsqueeze(-1)] = pad_token_id

            # Append the next token to the generated sequences
            generated_seqs = torch.cat([generated_seqs, next_token_ids], dim=1)

            # Update finished sequences for sequences that generated EOS token
            finished = finished | (next_token_ids.squeeze(-1) == eos_token_id)

        # Pad sequences to max_seq_len if necessary
        current_total_length = input_ids.shape[1] + generated_seqs.shape[1]
        if current_total_length < max_seq_len:
            padding_length = max_seq_len - current_total_length
            padding = torch.full((batch_size, padding_length), pad_token_id, dtype=torch.long, device=self.device)
            generated_seqs = torch.cat([generated_seqs, padding], dim=1)
        elif current_total_length > max_seq_len:
            # Truncate if somehow exceeded max_seq_len
            excess = current_total_length - max_seq_len
            generated_seqs = generated_seqs[:, :-excess]

        # Return the generated sequences (input + generated)
        final_sequences = torch.cat([input_ids, generated_seqs], dim=1)
        return final_sequences

