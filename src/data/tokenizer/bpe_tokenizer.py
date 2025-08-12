import os
from collections import Counter
from typing import Any, List, Tuple, Dict, Set, Optional

from src.data.tokenizer.tokenizer import Tokenizer


class BPETokenizer(Tokenizer):
    """Byte-Pair Encoding (BPE) Tokenizer"""

    def __init__(self: Any, model_name: str, max_seq_len: int,
                    cache_dir: Optional[str] = None) -> None:
        """Byte-Pair Encoding (BPE) Tokenizer constructor
        @param model_name: Name of the model
        @param max_seq_len: Maximum sequence length
        @param cache_dir: Directory to cache the tokenizer files
        """
        super(BPETokenizer, self).__init__(model_name, max_seq_len, cache_dir)  # Call parent constructor

        # Store model name and max sequence length
        self.model_name: str = model_name
        self.max_seq_len: int = max_seq_len
        # self.tokenizer: Any = None

        # Special tokens
        self._pad_token: bytes = b"<pad>"
        self._unk_token: bytes = b"<unk>"
        self._sos_token: bytes = b"<sos>"
        self._eos_token: bytes = b"<eos>"

        self.special_tokens: Set[bytes] = {
            self._pad_token,
            self._unk_token,
            self._sos_token,
            self._eos_token,
        }

        # Special token IDs
        self._pad_token_id: int = 0     # Padding token ID
        self._unk_token_id: int = 1     # Unknown token ID
        self._sos_token_id: int = 2     # Start of sequence token ID
        self._eos_token_id: int = 3     # End of sequence token ID

        # Initialize vocabulary with special tokens
        self.vocab: Dict[bytes, int] = {
            self._pad_token: self._pad_token_id,
            self._unk_token: self._unk_token_id,
            self._sos_token: self._sos_token_id,
            self._eos_token: self._eos_token_id,
        }

        self.id2token: Dict[int, bytes] = {v:k for k, v in self.vocab.items()}

class ByteLevelBPETokenizer(BPETokenizer):
    """Byte-Level Byte-Pair Encoding (BPE) Tokenizer"""

    def __init__(self: Any, model_name: str, max_seq_len: int,
                    cache_dir: Optional[str] = None) -> None:
        """Byte-Level Byte-Pair Encoding (BPE) Tokenizer constructor
        @param model_name: Name of the model
        @param max_seq_len: Maximum sequence length
        @param cache_dir: Directory to cache the tokenizer files
        """
        super(ByteLevelBPETokenizer, self).__init__(model_name, max_seq_len, cache_dir)

        self.merges: Dict[Tuple[bytes], int] = {}  # Dictionary to store merges

    def _get_stats(self: Any, byte_corpus: List[List[bytes]]) -> Counter:
        """Get byte pair statistics from the byte corpus
        @param byte_corpus: List of byte sequences
        @return: Counter with byte pair frequencies
        """
        pairs: Counter = Counter()
        for bytes_seq in byte_corpus:
            for i in range(len(bytes_seq) - 1):
                pair = (bytes_seq[i], bytes_seq[i + 1])
                pairs[pair] += 1
        return pairs

    def _merge_bytes(self: Any, byte_corpus: List[List[bytes]],
                    pair: Tuple[bytes, bytes]) -> List[List[bytes]]:
        """Merge byte pairs in the byte corpus
        @param byte_corpus: List of byte sequences
        @param pair: Byte pair to merge
        @return: List of byte sequences with the merged byte pair
        """
        new_bytes_seqs: List[List[bytes]] = []
        new_bytes: bytes = b''.join(pair)  # Create new byte from the pair
        for bytes_seq in byte_corpus:
            new_bytes_seq: List[bytes] = []
            for i in range(len(bytes_seq)):
                if (i < len(bytes_seq) - 1 and bytes_seq[i] == pair[0] and bytes_seq[i + 1] == pair[1]):
                    new_bytes_seq.append(new_bytes)
                    i += 1
                else:
                    new_bytes_seq.append(bytes_seq[i])
            new_bytes_seqs.append(new_bytes_seq)
        return new_bytes_seqs

    def build_vocab(self: Any, texts: List[str], max_vocab_size: int) -> None:
        """Build vocabulary from the provided texts
        @param texts: List of texts to build the vocabulary from
        @param vocab_size: Desired vocabulary size
        @param min_freq: Minimum frequency for a token to be included in the vocabulary
        """
        # Ensure vocabulary size is greater than 256 (a byte range) plus the number of special tokens
        assert max_vocab_size > 256 + len(self.special_tokens), "Vocabulary size must be greater than 256 plus the number of special tokens."
        self.vocab += {bytes(i): i + len(self.special_tokens) for i in range(256)}

        # Build byte-level corpus
        byte_corpus: List[List[bytes]] = [text.encode('utf-8') for text in texts]
        byte_corpus = [list(bytes) for bytes in byte_corpus]

        # BPE algorithm
        merges: Dict[bytes, int] = {}
        num_merge: int = max_vocab_size - len(self.special_tokens) - 256  # Number of merges to perform

        for i in range(num_merge):
            # Get byte pair statistics
            stats: Counter = self._get_stats(byte_corpus)   # Get byte pair statistics
            if not stats: break

            # Get the most common byte pair and update the byte corpus
            top_freq_pair: Tuple[bytes, bytes] = stats.most_common(1)[0][0] # Get the most common byte pair
            byte_corpus = self._merge_bytes(byte_corpus, top_freq_pair) # Merge the most common byte pair

            # Add the new token to the vocabulary and merges
            merges[top_freq_pair] = i  # Store the merge operation
            new_token: bytes = b''.join(top_freq_pair)
            self.vocab[new_token] = len(self.vocab)

        # Update id2token mapping
        self.id2token = {v: k for k, v in self.vocab.items()}
        self.merges = merges

    def convert_ids_to_tokens(self: Any, ids: List[int], skip_special_tokens: bool) -> List[bytes]:
        """Convert token IDs to tokens
        @param ids: List of token IDs
        @param skip_special_tokens: Whether to skip special tokens
        @return: List of tokens corresponding to the provided IDs
        """
        # Initialize list to hold tokens
        tokens: List[bytes] = []

        # Convert each ID to its corresponding token
        for token_id in ids:
            token: bytes = self.id2token.get(token_id, self._unk_token)
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)

        # Return list of tokens
        return tokens

    def convert_tokens_to_ids(self: Any, tokens: List[bytes], skip_special_tokens: bool) -> List[int]:
        """Convert a list of tokens to token IDs
        @param tokens: List of tokens to convert
        @param skip_special_tokens: Whether to skip special tokens in the input
        @return: List of token IDs corresponding to the provided tokens
        """
        # Initialize list to hold token IDs
        token_ids: List[int] = list()

        # Convert each token to its corresponding ID
        for token in tokens:
            if skip_special_tokens and token in self.special_tokens:
                continue
            token_id: int = self.vocab.get(token, self._unk_token_id)
            token_ids.append(token_id)

        # Return list of token IDs
        return token_ids

    def encode(self: Any, text: str, add_special_tokens: bool = True, padding: bool = True,
                truncation: bool = True) -> List[int]:
        """Encode a text string into token IDs
        @param text: Input text string to encode
        @param add_special_tokens: Whether to add special tokens to the encoded sequence (sos, eos)
        @param padding: Whether to pad the encoded sequence to the maximum sequence length
        @param truncation: Whether to truncate the encoded sequence to the maximum sequence length
        @return: List of token IDs corresponding to the encoded text
        """
        # Tokenize the text
        text_bytes: List[bytes] = list(text.encode('utf-8'))

        # Convert bytes to token IDs
        while len(text_bytes) >= 2:
            # Get pairs of adjacent bytes
            pairs: List[Tuple[bytes, bytes]] = [(text_bytes[i], text_bytes[i + 1]) for i in range(len(text_bytes) - 1)]
            # Find the pair with the highest priority based on merges
            top_priority_pair: Tuple[bytes, bytes] = min(pairs, key=lambda pair: self.merges.get(pair, float('inf')))
            if top_priority_pair not in self.merges:
                break
            # Merge the pair
            text_bytes = self._merge_bytes([text_bytes], top_priority_pair)[0]

        # Convert bytes to token IDs
        token_ids: List[int] = [self.vocab.get(bytes_seq, self._unk_token_id) for bytes_seq in text_bytes]

        # Deal with special tokens
        if add_special_tokens:
            token_ids = [self._sos_token_id] + token_ids + [self._eos_token_id]

        # Deal with truncation
        if truncation == True and len(token_ids) > self.max_seq_len:
            if add_special_tokens:
                token_ids = token_ids[:self.max_seq_len - 1] + [self._eos_token_id]
            else:
                token_ids = token_ids[:self.max_seq_len]

        # Deal with padding
        if padding == True:
            token_ids += [self._pad_token_id] * (self.max_seq_len - len(token_ids))

        # Return the list of token IDs
        return token_ids

    def decode(self: Any, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs into a text string
        @param token_ids: List of token IDs to decode
        @param skip_special_tokens: Whether to skip special tokens in the output
        @return: Decoded text string
        """
        # Convert token IDs to tokens
        tokens: List[bytes] = self.convert_ids_to_tokens(token_ids, skip_special_tokens)

        # Join tokens into a single string and decode from bytes to string
        decoded_text: str = b''.join(tokens).decode('utf-8')

        # Return the decoded text
        return decoded_text

