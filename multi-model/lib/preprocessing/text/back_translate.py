"""
Back-translation text augmentation (UDA-style).

Provides two augmentors:

  BackTranslateAugmentor  — round-trip translation via Helsinki-NLP OPUS-MT
                           models.  Heavier but higher-quality.

  BERTMaskAugmentor       — mask-then-fill with a BERT masked LM.  Lighter but
                           noisier.

Use build_text_augmenter() to pick one by name.
"""

import random
from typing import List, Optional, Union

import torch

# ── Optional dependency imports ────────────────────────────────────────────
_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers import pipeline as hf_pipeline

    _TRANSFORMERS_AVAILABLE = True
except ImportError:

    def _missing_deps(*args: object, **kwargs: object) -> None:
        raise ImportError(
            "The 'transformers' package is required for text augmentation.\n"
            "  pip install transformers"
        )

    AutoModelForSeq2SeqLM = object  # type: ignore
    AutoTokenizer = object  # type: ignore
    hf_pipeline = _missing_deps  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════
# ── Back-Translate ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


class BackTranslateAugmentor:
    """
    Round-trip translation augmentation (UDA approach).

    Translates texts from *src_lang* → *tgt_lang* → back to *src_lang*,
    producing paraphrases that preserve the original meaning while introducing
    lexical and syntactic diversity.

    Models are loaded lazily on the first call so that construction is cheap.

    Parameters
    ----------
    device : str
        Torch device string (default ``'cuda'``).
    batch_size : int
        Default batch size for translation (default ``32``).
    src_lang : str
        Source language code, e.g. ``'en'`` (default ``'en'``).
    tgt_lang : str
        Intermediate language code, e.g. ``'fr'`` (default ``'fr'``).
    max_length : int
        Maximum generation length (default ``128``).
    """

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 32,
        src_lang: str = "en",
        tgt_lang: str = "fr",
        max_length: int = 128,
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The 'transformers' package is required for BackTranslateAugmentor.\n"
                "  pip install transformers"
            )

        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length

        self._forward_tokenizer: Optional[AutoTokenizer] = None  # type: ignore
        self._forward_model: Optional[AutoModelForSeq2SeqLM] = None  # type: ignore
        self._backward_tokenizer: Optional[AutoTokenizer] = None  # type: ignore
        self._backward_model: Optional[AutoModelForSeq2SeqLM] = None  # type: ignore

    # ── Lazy model loading ────────────────────────────────────────────────

    def _lazy_load(self) -> None:
        if self._forward_model is not None:
            return

        forward_name = f"Helsinki-NLP/opus-mt-{self.src_lang}-{self.tgt_lang}"
        backward_name = f"Helsinki-NLP/opus-mt-{self.tgt_lang}-{self.src_lang}"

        self._forward_tokenizer = AutoTokenizer.from_pretrained(forward_name)
        self._forward_model = AutoModelForSeq2SeqLM.from_pretrained(forward_name).to(
            self.device
        )
        self._backward_tokenizer = AutoTokenizer.from_pretrained(backward_name)
        self._backward_model = AutoModelForSeq2SeqLM.from_pretrained(backward_name).to(
            self.device
        )

        self._forward_model.eval()
        self._backward_model.eval()

    # ── Translation helpers ───────────────────────────────────────────────

    def _translate(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,  # type: ignore
        model: AutoModelForSeq2SeqLM,  # type: ignore
        batch_size: int,
    ) -> List[str]:
        results: List[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=self.max_length)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)
        return results

    # ── Public API ────────────────────────────────────────────────────────

    def __call__(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[str]:
        """
        Back-translate a list of strings.

        Parameters
        ----------
        texts : list of str
            Input texts in *src_lang*.
        batch_size : int or None
            Batch size for this call (falls back to instance default).

        Returns
        -------
        list of str
            Back-translated paraphrases.
        """
        self._lazy_load()
        bs = batch_size if batch_size is not None else self.batch_size

        translated = self._translate(
            texts, self._forward_tokenizer, self._forward_model, bs
        )
        back_translated = self._translate(
            translated, self._backward_tokenizer, self._backward_model, bs
        )
        return back_translated

    def augment(
        self, texts: List[str], num_augments: int = 1, keep_original: bool = True
    ) -> List[List[str]]:
        """
        Return multiple back-translated variants per input.

        Parameters
        ----------
        texts : list of str
            Input texts.
        num_augments : int
            Number of back-translated copies per text (default ``1``).
        keep_original : bool
            Whether to include the original text first (default ``True``).

        Returns
        -------
        list of list of str
            ``result[i]`` is a list of strings for the i-th input:
            ``[original, aug_1, aug_2, …]``.
        """
        self._lazy_load()
        bs = self.batch_size

        all_augmented: List[List[str]] = []
        for text in texts:
            variants: List[str] = [text] if keep_original else []
            # Repeat the same text num_augments times and translate in batch
            repeat = [text] * num_augments
            back = self._translate(
                repeat, self._forward_tokenizer, self._forward_model, bs
            )
            back = self._translate(
                back, self._backward_tokenizer, self._backward_model, bs
            )
            variants.extend(back)
            all_augmented.append(variants)

        return all_augmented


# ═══════════════════════════════════════════════════════════════════════════════
# ── BERT Masked-LM Augment ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


class BERTMaskAugmentor:
    """
    Lightweight text augmentation via BERT masked-LM.

    Randomly masks *mask_prob* fraction of tokens and replaces each with a
    sample from its top-*k* predictions.  Much faster than back-translation but
    produces noisier outputs.

    Parameters
    ----------
    model_name : str
        HuggingFace masked LM model name (default ``'bert-base-uncased'``).
    device : str
        Torch device string (default ``'cuda'``).
    top_k : int
        Number of top predictions to sample from (default ``5``).
    mask_prob : float
        Probability of masking each token (default ``0.15``).
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "cuda",
        top_k: int = 5,
        mask_prob: float = 0.15,
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The 'transformers' package is required for BERTMaskAugmentor.\n"
                "  pip install transformers"
            )

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.top_k = top_k
        self.mask_prob = mask_prob

        self._pipe = None

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _lazy_load(self) -> None:
        if self._pipe is not None:
            return
        self._pipe = hf_pipeline(
            "fill-mask",
            model=self.model_name,
            top_k=self.top_k,
            device=0 if self.device == "cuda" else -1,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def __call__(self, texts: List[str]) -> List[str]:
        """
        Augment texts by masking and filling random tokens.

        Parameters
        ----------
        texts : list of str
            Input strings.

        Returns
        -------
        list of str
            Augmented strings where some tokens have been replaced.
        """
        self._lazy_load()
        results: List[str] = []

        for text in texts:
            tokens = text.split()
            if not tokens:
                results.append(text)
                continue

            masked_text = list(tokens)
            # Determine which positions to mask
            for i in range(len(masked_text)):
                if random.random() < self.mask_prob:
                    masked_text[i] = self._pipe.tokenizer.mask_token

            masked_str = " ".join(masked_text)

            # If nothing was masked, keep original
            if self._pipe.tokenizer.mask_token not in masked_str:
                results.append(text)
                continue

            try:
                predictions = self._pipe(masked_str)
            except Exception:
                results.append(text)
                continue

            # predictions is a list of lists, one per mask
            if isinstance(predictions[0], list):
                filled_tokens = list(tokens)
                mask_idx = 0
                for i, tok in enumerate(masked_text):
                    if tok == self._pipe.tokenizer.mask_token:
                        candidates = predictions[mask_idx]
                        choice = random.choice(candidates)
                        filled_tokens[i] = choice["token_str"]
                        mask_idx += 1
                results.append(" ".join(filled_tokens))
            else:
                # Single mask — pipeline returns flat list
                filled = random.choice(predictions)["token_str"]
                results.append(filled)

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# ── Factory ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════


def build_text_augmenter(method: str = "back_translate", **kwargs: object) -> object:
    """
    Factory: return the appropriate text augmentor instance.

    Parameters
    ----------
    method : str
        One of ``'back_translate'`` or ``'bert_mask'``.
    **kwargs
        Passed directly to the augmentor constructor.

    Returns
    -------
    BackTranslateAugmentor or BERTMaskAugmentor

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    method = method.lower().strip()
    if method == "back_translate":
        return BackTranslateAugmentor(**kwargs)  # type: ignore
    elif method == "bert_mask":
        return BERTMaskAugmentor(**kwargs)  # type: ignore
    else:
        raise ValueError(
            f"Unknown text augmentation method: {method!r}. "
            f"Expected one of: 'back_translate', 'bert_mask'."
        )
