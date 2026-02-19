import torch
import torch.nn as nn
import numpy as np
import json
import copy
from math import exp
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections.abc import MutableMapping
import logging
# Transformers imports
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Logits Processors (combining warpers and processors)
# =============================================================================

class TemperatureLogitsProcessor(LogitsProcessor):
    """Temperature scaling"""

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"temperature has to be a positive float, but is {temperature}")
        self.temperature = temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores / self.temperature
        return scores


class TopPLogitsProcessor(LogitsProcessor):
    """Top-p (nucleus) sampling"""

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"top_p has to be a float > 0 and < 1, but is {top_p}")
        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        # Shift to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopKLogitsProcessor(LogitsProcessor):
    """Top-k sampling"""

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k has to be a strictly positive integer, but is {top_k}")
        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TailFreeSamplingLogitsProcessor(LogitsProcessor):
    """Tail Free Sampling"""

    def __init__(self, tfs: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        tfs = float(tfs)
        if tfs < 0 or tfs > 1.0:
            raise ValueError(f"tfs has to be a float > 0 and < 1, but is {tfs}")
        self.tfs = tfs
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.tfs >= 1.0:
            return scores

        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        probs = sorted_logits.softmax(dim=-1)

        # Compute second derivative normalized CDF
        d2 = probs.diff().diff().abs()
        normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
        normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

        # Remove tokens with CDF value above the threshold
        sorted_indices_to_remove = normalized_d2_cdf > self.tfs
        sorted_indices_to_remove = torch.cat(
            (
                torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                sorted_indices_to_remove,
                torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
            ),
            dim=-1,
        )
        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopALogitsProcessor(LogitsProcessor):
    """Top-A sampling"""

    def __init__(self, threshold: float, filter_value: float = -float("inf")):
        if not isinstance(threshold, float) or (threshold < 0 or threshold > 1.0):
            raise ValueError(f"threshold has to be a float > 0 and < 1, but is {threshold}")
        self.z = threshold
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.nn.functional.softmax(scores, dim=-1)
        limit = torch.pow(torch.max(probs, dim=-1, keepdim=True)[0], 2.0) * self.z
        indices_to_remove = probs < limit
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TypicalLogitsProcessor(LogitsProcessor):
    """Typical sampling"""

    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        mass = float(mass)
        if mass <= 0 or mass >= 1.0:
            raise ValueError(f"typical_p has to be a float > 0 and < 1, but is {mass}")
        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # Shift and sort
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind[last_ind < 0] = 0
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """Repetition penalty with optional slope and range"""

    def __init__(self, penalty: float = 1.0, slope=3.33, penalize_last=250, alpha_frequency=None, alpha_presence=None, whitelist=None):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"penalty has to be a strictly positive float, but is {penalty}")

        self.penalty = 1.0 if penalty < 1.0 else penalty
        self.raw_penalty = penalty
        self.penalize_last = None

        if slope is not None and penalize_last is not None and penalize_last >= 1:
            self.penalty = (torch.arange(penalize_last) / (penalize_last - 1)) * 2. - 1
            self.penalty = (slope * self.penalty) / (1 + torch.abs(self.penalty) * (slope - 1))
            self.penalty = 1 + ((self.penalty + 1) / 2).unsqueeze(0) * (penalty - 1)
            self.penalize_last = penalize_last

        self.alpha_frequency = alpha_frequency if alpha_frequency is not None and alpha_frequency > 0.0 else None
        self.alpha_presence = alpha_presence if alpha_presence is not None and alpha_presence > 0.0 else None
        self.alpha_enable = self.alpha_frequency is not None or self.alpha_presence is not None

        self.whitelist = None
        self.whitelist_list = None
        if whitelist is not None:
            self.whitelist_list = whitelist

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.whitelist is None and self.whitelist_list is not None:
            self.whitelist_list = list(filter(lambda x: x >= 0 and x < scores.shape[1], self.whitelist_list))
            if len(self.whitelist_list) > 0:
                self.whitelist = torch.tensor(self.whitelist_list).long().sort()[0]
                self.whitelist = self.whitelist.to(input_ids.device)

        if self.whitelist is not None:
            unpenalized = scores.gather(1, self.whitelist.view(1, -1))

        if self.raw_penalty > 1.0:
            if self.penalize_last is not None:
                penality_len = min(input_ids.shape[1], self.penalize_last)
                input_ids_slice = input_ids[:, -penality_len:]
            else:
                input_ids_slice = input_ids

            score = torch.gather(scores, 1, input_ids_slice)

            if self.penalize_last is not None:
                penalty = self.penalty.type(score.dtype).to(score.device)
                score = torch.where(score < 0, score * penalty[:, -penality_len:], score / penalty[:, -penality_len:])
            else:
                score = torch.where(score < 0, score * self.penalty, score / self.penalty)

            scores.scatter_(1, input_ids_slice, score)

        if self.alpha_enable:
            c = torch.zeros(scores.shape).long().to(input_ids.device)
            for i in range(input_ids.shape[0]):
                if self.penalize_last is not None:
                    token_input_ids, counts = torch.unique(input_ids[i, -self.penalize_last:], sorted=True, return_counts=True, dim=-1)
                else:
                    token_input_ids, counts = torch.unique(input_ids[i], sorted=True, return_counts=True, dim=-1)
                c[i].scatter_(0, token_input_ids, counts)

            if self.alpha_frequency:
                scores -= c * self.alpha_frequency
            if self.alpha_presence:
                scores[c > 0] -= self.alpha_presence

        if self.whitelist is not None:
            scores.scatter_(1, self.whitelist.view(1, -1), unpenalized)

        return scores


class LogitBiasProcessor(LogitsProcessor):
    """Adds bias to specific tokens"""

    def __init__(self, logit_bias: List[Tuple[int, float]] = []):
        if not isinstance(logit_bias, list) and len(logit_bias) > 0:
            raise ValueError("logit_bias has to be a non-empty list")
        self.logit_bias = [(token, exp(bias)) for token, bias in logit_bias]
        self.bias = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.bias is None:
            self.bias = torch.zeros(scores.shape[1]).float()
            logit_bias = torch.tensor(self.logit_bias)
            self.bias.scatter_(0, logit_bias[:, 0].long(), logit_bias[:, 1].float())
            self.bias = self.bias.to(scores.dtype).to(scores.device).unsqueeze(0)
        return scores + self.bias


class PhraseBiasProcessor(LogitsProcessor):
    """Biases specific phrases/sequences"""

    def __init__(self, words_ids: List[List[int]], bias: float, ensure_sequence_finish: bool, generate_once: bool):
        if not isinstance(words_ids, list) or len(words_ids) == 0:
            return

        if any(not isinstance(word_ids, list) for word_ids in words_ids):
            raise ValueError("words_ids has to be a list of lists")

        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in word_ids)
            for word_ids in words_ids
        ):
            raise ValueError("Each list in words_ids has to be a list of positive integers")

        self.words_ids = words_ids
        self.bias = exp(bias)
        self.ensure_sequence_finish = ensure_sequence_finish
        self.generate_once = generate_once

    def slice_in_list(self, l, s):
        a = 0
        for i in range(l.shape[1]):
            for j in range(len(s)):
                if l[:, i].item() == s[j]:
                    a += 1
        return a

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for phrase_ids in self.words_ids:
            if self.generate_once:
                if phrase_ids[0] not in input_ids:
                    scores[:, phrase_ids[0]] += self.bias
                    continue
            else:
                scores[:, phrase_ids[0]] += self.bias
            idx = self.slice_in_list(input_ids, phrase_ids)
            if idx == len(phrase_ids) or idx > len(phrase_ids):
                continue
            else:
                if self.ensure_sequence_finish:
                    if self.generate_once:
                        scores[:, phrase_ids[idx]] -= self.bias
                    scores[:, phrase_ids[idx]] = 1000.0
                    break
                else:
                    scores[:, phrase_ids[idx]] += self.bias
                continue

        return scores


class MinLengthLogitsProcessor(LogitsProcessor):
    """Prevents EOS token until minimum new tokens are generated"""

    def __init__(self, min_new_tokens: int, prompt_length: int, eos_token_id: int):
        if not isinstance(min_new_tokens, int) or min_new_tokens < 0:
             # [cite: 33] Validating min_length is non-negative integer
            raise ValueError(f"min_new_tokens has to be a non-negative integer, but is {min_new_tokens}")
        if not isinstance(eos_token_id, int) or eos_token_id < 0:
             # [cite: 33] Validating eos_token_id is non-negative integer
            raise ValueError(f"eos_token_id has to be a non-negative integer, but is {eos_token_id}")
        
        self.min_new_tokens = min_new_tokens
        self.prompt_length = prompt_length # Store the prompt length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Calculate target length: original prompt + minimum new tokens
        target_length = self.prompt_length + self.min_new_tokens
        
        # Only mask EOS if total length is less than target length 
        if input_ids.shape[-1] < target_length:
            scores[:, self.eos_token_id] = -float("inf")
        return scores

class NoBadWordsLogitsProcessor(LogitsProcessor):
    """Prevents generation of bad words"""

    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: Optional[int] = None):
        if not isinstance(bad_words_ids, list) or len(bad_words_ids) == 0:
            raise ValueError("bad_words_ids has to be a non-empty list")
        self.bad_words_ids = bad_words_ids
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for bad_word_ids in self.bad_words_ids:
            # If it's a single token, ban it outright
            if len(bad_word_ids) == 1:
                scores[:, bad_word_ids[0]] = -float("inf")
            else:
                # Check if we're currently in a bad word sequence
                for i in range(1, len(bad_word_ids)):
                    if input_ids.shape[-1] >= i:
                        if input_ids[0, -i:].tolist() == bad_word_ids[:i]:
                            scores[:, bad_word_ids[i]] = -float("inf")
        return scores


class MaxLengthCriteria(StoppingCriteria):
    """Stops generation when max length is reached"""

    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids.shape[-1] >= self.max_length


class MaxTimeCriteria(StoppingCriteria):
    """Stops generation after max time"""

    def __init__(self, max_time: float):
        self.max_time = max_time
        self.start_time = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_time is None:
            import time
            self.start_time = time.time()
        import time
        return (time.time() - self.start_time) >= self.max_time


# =============================================================================
# GPTHF Model Class
# =============================================================================

class GPTHF:
    """GPT Text Generation with Hugging Face"""

    def __init__(self, model_name='EleutherAI/gpt-j-6B', device=None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f'Loading model {model_name} on {self.device}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        config = AutoConfig.from_pretrained(model_name)
        config.dtype = "auto"
        config.attn_dtype = "float8_e5m2"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            device_map={"": "cuda"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()

        logger.info(f'Model loaded successfully')

    @torch.inference_mode()
    def generate(self, args):
        """Generate text based on provided arguments"""
        logits_processors = []
        stopping_criterion = []
        eos_token_id = None
        output_scores = False
        prompt_length = None
        max_new_tokens = None

        input_ids = self.tokenizer(args["prompt"], return_tensors='pt').input_ids.to(self.device)
        prompt_length = input_ids.shape[-1]

        # Validate args
        if not isinstance(args, dict):
            raise TypeError("Arguments must be a dictionary")

        if "prompt" not in args:
            raise KeyError("Arguments must contain a prompt")
        else:
            prompt = args["prompt"]

        if "gen_args" not in args:
            raise KeyError("Arguments must contain generation arguments")

        if "sample_args" not in args:
            raise KeyError("Arguments must contain sampling arguments")

        # Stopping criteria
        if "max_length" in args["gen_args"] and args["gen_args"]["max_length"]:
            if not isinstance(args["gen_args"]["max_length"], int) or args["gen_args"]["max_length"] < 0:
                raise TypeError("max_length must be a positive integer")
            max_new_tokens = args["gen_args"]["max_length"]

        if "max_time" in args["gen_args"] and args["gen_args"]["max_time"]:
            if not isinstance(args["gen_args"]["max_time"], float) or args["gen_args"]["max_time"] < 0.0:
                raise TypeError("max_time must be a positive float")
            stopping_criterion.append(MaxTimeCriteria(args["gen_args"]["max_time"]))

        if "eos_token_id" in args["gen_args"] and args["gen_args"]["eos_token_id"]:
            if not isinstance(args["gen_args"]["eos_token_id"], int) or args["gen_args"]["eos_token_id"] < 0:
                raise TypeError("eos_token_id must be a positive integer")
            eos_token_id = args["gen_args"]["eos_token_id"]

        if "min_length" in args["gen_args"] and args["gen_args"]["min_length"]:
            if not isinstance(args["gen_args"]["min_length"], int):
                #  Type error check
                raise TypeError("min_length must be an integer")
            
            if eos_token_id:
                # Pass the prompt_length to the processor
                logits_processors.append(MinLengthLogitsProcessor(
                    min_new_tokens=args["gen_args"]["min_length"],
                    prompt_length=prompt_length,
                    eos_token_id=eos_token_id
                ))

        if "logprobs" in args["gen_args"] and args["gen_args"]["logprobs"]:
            if not isinstance(args["gen_args"]["logprobs"], int) or args["gen_args"]["logprobs"] < 0 or args["gen_args"]["logprobs"] > 20:
                raise TypeError("logprobs must be an integer between 0 and 20.")
            output_scores = True

        # Check if we have at least one stopping criteria
        if max_new_tokens is None and len(stopping_criterion) == 0:
            raise ValueError("Generation arguments must contain at least one stopping criteria such as max_length or max_time.")

        # Temperature (apply first)
        if "temp" in args["sample_args"] and args["sample_args"]["temp"]:
            if not isinstance(args["sample_args"]["temp"], (float, int)) or (args["sample_args"]["temp"] <= 0.0):
                raise ValueError("Temperature must be a float greater than 0.0")
            logits_processors.append(TemperatureLogitsProcessor(float(args["sample_args"]["temp"])))

        # Sampling processors
        if "top_p" in args["sample_args"] and args["sample_args"]["top_p"]:
            if not isinstance(args["sample_args"]["top_p"], (float, int)) or (args["sample_args"]["top_p"] < 0.0 or args["sample_args"]["top_p"] > 1.0):
                raise ValueError("top_p must be a float between 0 and 1")
            logits_processors.append(TopPLogitsProcessor(float(args["sample_args"]["top_p"])))

        if "top_k" in args["sample_args"] and args["sample_args"]["top_k"]:
            if not isinstance(args["sample_args"]["top_k"], int) or (args["sample_args"]["top_k"] <= 0):
                raise ValueError("top_k must be a positive integer")
            logits_processors.append(TopKLogitsProcessor(args["sample_args"]["top_k"]))

        if "top_a" in args["sample_args"] and args["sample_args"]["top_a"]:
            if not isinstance(args["sample_args"]["top_a"], (float, int)) or (args["sample_args"]["top_a"] < 0.0 or args["sample_args"]["top_a"] > 1.0):
                raise ValueError("top_a must be a float between 0 and 1")
            logits_processors.append(TopALogitsProcessor(float(args["sample_args"]["top_a"])))

        if "typical_p" in args["sample_args"] and args["sample_args"]["typical_p"]:
            if not isinstance(args["sample_args"]["typical_p"], (float, int)) or (args["sample_args"]["typical_p"] < 0.0 or args["sample_args"]["typical_p"] > 1.0):
                raise ValueError("typical_p must be a float between 0 and 1")
            logits_processors.append(TypicalLogitsProcessor(float(args["sample_args"]["typical_p"])))

        if "tfs" in args["sample_args"] and args["sample_args"]["tfs"]:
            if not isinstance(args["sample_args"]["tfs"], (float, int)) or (args["sample_args"]["tfs"] < 0.0 or args["sample_args"]["tfs"] > 1.0):
                raise ValueError("tfs must be a float between 0 and 1")
            logits_processors.append(TailFreeSamplingLogitsProcessor(float(args["sample_args"]["tfs"])))

        # Repetition penalty
        if "rep_p" in args["sample_args"] and args["sample_args"]["rep_p"]:
            rep_slope = None
            rep_range = None

            if "rep_p_slope" in args["sample_args"] and args["sample_args"]["rep_p_slope"]:
                if not isinstance(args["sample_args"]["rep_p_slope"], (float, int)) or args["sample_args"]["rep_p_slope"] < 0.0:
                    raise ValueError("rep_p_slope must be a positive float.")
                rep_slope = float(args["sample_args"]["rep_p_slope"])

            if "rep_p_range" in args["sample_args"] and args["sample_args"]["rep_p_range"]:
                if not isinstance(args["sample_args"]["rep_p_range"], int) or args["sample_args"]["rep_p_range"] < 0:
                    raise ValueError("rep_p_range must be a positive integer.")
                rep_range = args["sample_args"]["rep_p_range"]

            logits_processors.append(RepetitionPenaltyLogitsProcessor(
                penalty=float(args["sample_args"]["rep_p"]),
                slope=rep_slope,
                penalize_last=rep_range
            ))

        if "bad_words" in args["sample_args"] and args["sample_args"]["bad_words"]:
            if not isinstance(args["sample_args"]["bad_words"], list):
                raise ValueError("bad_words must be a non-empty list")
            bad_words_ids = []
            for bad_word in args["sample_args"]["bad_words"]:
                if not isinstance(bad_word, str):
                    raise ValueError("bad_words must be a list of strings")
                bad_words_ids.append(self.tokenizer.encode(bad_word, add_special_tokens=False))
            logits_processors.append(NoBadWordsLogitsProcessor(bad_words_ids, None))

        if "logit_biases" in args["sample_args"] and args["sample_args"]["logit_biases"]:
            if not isinstance(args["sample_args"]["logit_biases"], list):
                raise ValueError("logit_biases must be a list")
            logit_biases = []
            for logit_bias in args["sample_args"]["logit_biases"]:
                if not isinstance(logit_bias, dict) or "id" not in logit_bias or "bias" not in logit_bias:
                    raise ValueError("logit_biases must be a list of dicts with keys 'id' and 'bias'")
                if not isinstance(logit_bias["id"], int):
                    raise ValueError("logit_biases 'id' must be an integer")
                if not isinstance(logit_bias["bias"], (float, int)):
                    raise ValueError("logit_biases 'bias' must be a float")
                logit_biases.append((logit_bias["id"], float(logit_bias["bias"])))
            logits_processors.append(LogitBiasProcessor(logit_biases))

        if "phrase_biases" in args["sample_args"] and args["sample_args"]["phrase_biases"]:
            if not isinstance(args["sample_args"]["phrase_biases"], list):
                raise ValueError("phrase_biases must be a non-empty list")
            for bias in args["sample_args"]["phrase_biases"]:
                if not isinstance(bias, dict):
                    raise ValueError("biases must be a list of dictionaries")
                if "sequences" not in bias or not isinstance(bias["sequences"], list):
                    raise ValueError("phrase_biases must be a list of dictionaries with sequences")
                if "bias" not in bias or not isinstance(bias["bias"], (float, int)):
                    raise ValueError("biases must be a list of dictionaries with a bias key")
                if "ensure_sequence_finish" not in bias or not isinstance(bias["ensure_sequence_finish"], bool):
                    raise ValueError("biases must be a list of dictionaries with an ensure_sequence_finish key")
                if "generate_once" not in bias or not isinstance(bias["generate_once"], bool):
                    raise ValueError("biases must be a list of dictionaries with a generate_once key")
                logits_processors.append(PhraseBiasProcessor(
                    [self.tokenizer.encode(sequence, add_special_tokens=False) for sequence in bias["sequences"]],
                    float(bias["bias"]),
                    bias["ensure_sequence_finish"],
                    bias["generate_once"]
                ))

        logits_processor = LogitsProcessorList(logits_processors)
        stopping_criteria = StoppingCriteriaList(stopping_criterion)
        self.model.config.use_cache = True
        self.model.config.kv_cache_dtype = torch.float8_e5m2  # if kernel supports fp8

        # Generate using model.generate with do_sample=True
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria if len(stopping_criterion) > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=output_scores
        )

        output = {}
        output["output"] = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        if "logprobs" in args["gen_args"] and args["gen_args"]["logprobs"] and output_scores:
            logprobs = []
            for i in range(len(outputs.scores)):
                logprobs_seq = []
                scores_probs = outputs.scores[i].softmax(-1).topk(args["gen_args"]["logprobs"], dim=-1).values.tolist()
                scores_indices = outputs.scores[i].topk(args["gen_args"]["logprobs"], dim=-1).indices.tolist()
                for j in range(args['gen_args']['logprobs']):
                    logprobs_seq.append((scores_indices[0][j], scores_probs[0][j]))
                logprobs.append(logprobs_seq)
            output["logprobs"] = logprobs

        return output


# =============================================================================
# ComfyUI Node
# =============================================================================

class GPTHFTextGenerator:
    """ComfyUI Node for GPT Text Generation"""
    def __init__(self):
        self.model = None
        self.current_model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": """[Chen is a bakeneko that is the shikigami of Ran Yakumo, who is also the shikigami of Yukari Yakumo. Chen likes to troll and make fun of people, she is quite a fun cat.]\n\nme: chen what is life\nchen:"""
                }),
                "model_name": ("STRING", {
                    "default": "model path????"
                }),
                "max_length": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 2048
                }),
                "temperature": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.01
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "tfs": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "rep_p": ("FLOAT", {
                    "default": 1.115,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.001
                }),
                "rep_p_range": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 4096
                }),
                "eos_token_id": ("INT", {
                    "default": 198,
                    "min": 0,
                    "max": 50257
                }),
                "min_length": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 2048
                }),
            },
            "optional": {
                "bad_words_json": ("STRING", {
                    "multiline": True,
                    "default": json.dumps(["~", " ~", " Chen", " chen", "chen", "Chen", " ._.", "._.", "[", " [", " Jews", " jews", "jews", "Jews", " Niggers", " niggers", "Niggers", "niggers", " Nigger", " nigger", "\"", " \"", " —", "—", " @", "@", " sex", "sex"])
                }),
                "logit_biases_json": ("STRING", {
                    "multiline": True,
                    "default": json.dumps([
                        {"id": 13557, "bias": -0.15},
                        {"id": 3228, "bias": -0.05},
                        {"id": 198, "bias": 0.15}
                    ])
                }),
                "phrase_biases_json": ("STRING", {
                    "multiline": True,
                    "default": json.dumps([
                        {
                            "sequences": ["._.", " ._."],
                            "bias": -0.15,
                            "ensure_sequence_finish": False,
                            "generate_once": False
                        }
                    ])
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate_text"
    CATEGORY = "text/generation"

    def generate_text(self, prompt, model_name, max_length, temperature, top_p, tfs,
                     rep_p, rep_p_range, eos_token_id, min_length,
                     bad_words_json=None, logit_biases_json=None, phrase_biases_json=None):
        """Generate text using GPT model"""

        # Load model if not loaded or if model name changed
        if self.model is None or self.current_model_name != model_name:
            print(f"Loading model: {model_name}")
            self.model = GPTHF(model_name=model_name)
            self.current_model_name = model_name

        # Parse JSON inputs
        bad_words = []
        if bad_words_json:
            try:
                bad_words = json.loads(bad_words_json)
            except json.JSONDecodeError:
                print("Warning: Invalid bad_words JSON, using empty list")

        logit_biases = []
        if logit_biases_json:
            try:
                logit_biases = json.loads(logit_biases_json)
            except json.JSONDecodeError:
                print("Warning: Invalid logit_biases JSON, using empty list")

        phrase_biases = []
        if phrase_biases_json:
            try:
                phrase_biases = json.loads(phrase_biases_json)
            except json.JSONDecodeError:
                print("Warning: Invalid phrase_biases JSON, using empty list")

        # Build arguments dictionary
        args = {
            "prompt": prompt,
            "sample_args": {
                "temp": temperature,
                "top_p": top_p,
                "tfs": tfs,
                "rep_p": rep_p,
                "rep_p_range": rep_p_range,
                "bad_words": bad_words,
                "logit_biases": logit_biases,
                "phrase_biases": phrase_biases
            },
            "gen_args": {
                "max_length": max_length,
                "min_length": min_length,
                "eos_token_id": eos_token_id
            }
        }

        result = self.model.generate(args)

        return (result["output"],)


# =============================================================================
# ComfyUI Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "GPTHFTextGenerator": GPTHFTextGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTHFTextGenerator": "modern sukima gen"
}