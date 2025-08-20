import torch
from typing import Any, Dict

from src.data.bpe_tokenizer import BpeTokenizer
from src.models.gpt_mini_model_parallel import GPTMiniModelParallel


class AutoregressiveGPTMiniModelParallel:
    """A class for autoregressive generation with the GPT-Mini Model Parallel version."""

    def __init__(self: Any, model: GPTMiniModelParallel, tokenizer: BpeTokenizer, device_map: Dict[str, torch.device]) -> None:
        """Initialize the AutoregressiveGPTMiniModelParallel class with a model and tokenizer."""
        self.model: GPTMiniModelParallel = model
        self.tokenizer: BpeTokenizer = tokenizer
        self.device_map: Dict[str, torch.device] = device_map
        self.primary_device: torch.device = list(device_map.values())[0]  # First device for input handling

    def generate(self: Any, input_ids: torch.Tensor, max_seq_len: int) -> torch.Tensor:
        """Generate text using the GPT-Mini Model Parallel version."""
        # Necessary variables
        sos_token_id: int = self.tokenizer.sos_id
        eos_token_id: int = self.tokenizer.eos_id
        pad_token_id: int = self.tokenizer.pad_id

        batch_size: int = input_ids.shape[0]
        
        # Move input to primary device initially
        input_ids = input_ids.to(self.primary_device)
        generated_seqs: torch.Tensor = torch.empty((batch_size, 0), device=self.primary_device)

        # Track which sequences have finished generation (generated EOS token)
        finished: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool, device=self.primary_device)

        # Set model to evaluation mode
        self.model.eval()

        # Autoregressive generation loop
        with torch.no_grad():
            for step in range(max_seq_len - 1):    # -1 because we already have SOS token
                # If all sequences are finished, break early
                if torch.all(finished):
                    break

                # Create mask for current generated sequences
                current_input: torch.Tensor = torch.cat([input_ids, generated_seqs], dim=1)
                
                # Move current input to embedding device (model handles device transfers internally)
                current_input = self.model.move_batch_to_devices(current_input)
                
                # Forward pass through the model (model handles internal device transfers)
                logits: torch.Tensor = self.model(current_input)

                # Greedy decoding: select the token with the highest probability
                # Get the logits for the last position
                next_token_logits: torch.Tensor = logits[:, -1, :]
                
                # Move logits to primary device for token selection
                if next_token_logits.device != self.primary_device:
                    next_token_logits = next_token_logits.to(self.primary_device)
                
                next_token_ids: torch.Tensor = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Mark sequences that generated EOS token as finished
                eos_generated: torch.Tensor = (next_token_ids.squeeze(-1) == eos_token_id)
                finished = finished | eos_generated

                # For finished sequences, replace the generated token with PAD token
                next_token_ids[finished.unsqueeze(-1)] = pad_token_id

                # Append the next token to the generated sequences
                generated_seqs = torch.cat([generated_seqs, next_token_ids], dim=1)

        # Combine input and generated sequences
        full_sequences: torch.Tensor = torch.cat([input_ids, generated_seqs], dim=1)

        return full_sequences

    def beam_search_generate(self: Any, input_ids: torch.Tensor, max_seq_len: int, beam_size: int = 4) -> torch.Tensor:
        """Generate text using beam search with model parallelism."""
        # Note: This is a simplified beam search implementation
        # For production use, consider more sophisticated beam search algorithms
        
        eos_token_id: int = self.tokenizer.eos_id
        pad_token_id: int = self.tokenizer.pad_id
        
        batch_size: int = input_ids.shape[0]
        input_ids = input_ids.to(self.primary_device)
        
        # Initialize beam search
        beam_scores = torch.zeros(batch_size, beam_size, device=self.primary_device)
        beam_sequences = input_ids.unsqueeze(1).repeat(1, beam_size, 1)  # [batch_size, beam_size, seq_len]
        finished_sequences = []
        
        self.model.eval()
        
        with torch.no_grad():
            for step in range(max_seq_len - input_ids.size(1)):
                # Reshape for model input
                current_shape = beam_sequences.shape
                flat_sequences = beam_sequences.view(-1, current_shape[2])  # [batch_size * beam_size, seq_len]
                
                # Move to embedding device and get logits
                flat_sequences = self.model.move_batch_to_devices(flat_sequences)
                logits = self.model(flat_sequences)
                
                # Get last position logits and move to primary device
                next_token_logits = logits[:, -1, :]
                if next_token_logits.device != self.primary_device:
                    next_token_logits = next_token_logits.to(self.primary_device)
                
                # Reshape back to beam structure
                next_token_logits = next_token_logits.view(batch_size, beam_size, -1)
                
                # Apply softmax and get log probabilities
                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                # Add to beam scores
                vocab_size = log_probs.size(-1)
                expanded_scores = beam_scores.unsqueeze(-1) + log_probs  # [batch_size, beam_size, vocab_size]
                expanded_scores = expanded_scores.view(batch_size, beam_size * vocab_size)
                
                # Get top beam_size candidates
                top_scores, top_indices = torch.topk(expanded_scores, beam_size, dim=-1)
                
                # Convert indices back to beam and vocab indices
                beam_indices = top_indices // vocab_size
                vocab_indices = top_indices % vocab_size
                
                # Update sequences and scores
                new_sequences = []
                new_scores = []
                
                for batch_idx in range(batch_size):
                    batch_new_sequences = []
                    batch_new_scores = []
                    
                    for beam_idx in range(beam_size):
                        orig_beam_idx = beam_indices[batch_idx, beam_idx]
                        next_token = vocab_indices[batch_idx, beam_idx]
                        
                        # Get the sequence from the selected beam
                        seq = beam_sequences[batch_idx, orig_beam_idx]
                        new_seq = torch.cat([seq, next_token.unsqueeze(0)])
                        
                        batch_new_sequences.append(new_seq)
                        batch_new_scores.append(top_scores[batch_idx, beam_idx])
                    
                    new_sequences.append(torch.stack(batch_new_sequences))
                    new_scores.append(torch.stack(batch_new_scores))
                
                beam_sequences = torch.stack(new_sequences)
                beam_scores = torch.stack(new_scores)
        
        # Return the best sequence for each batch item
        best_sequences = beam_sequences[:, 0, :]  # Take the first (best) beam
        
        return best_sequences

    def get_memory_usage(self: Any) -> Dict[str, Dict[str, float]]:
        """Get memory usage across all devices."""
        return self.model.get_memory_usage()
