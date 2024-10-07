from math import ceil
from typing import Generator, List, Union

import numpy as np
import torch
from oneliner_utils import read_jsonl
from torch import Tensor, einsum
from torch.nn.functional import normalize
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..paths import embeddings_folder_path, encoder_state_path, index_path
import logging

pbar_kwargs = dict(position=0, dynamic_ncols=True, mininterval=1.0)


def last_token_pool(
        last_hidden_states: Tensor,
        attention_mask: Tensor
) -> Tensor:
    """
    Helper method for Salesforce/SFR-Embedding-2_R

    Parameters
    ----------
    last_hidden_states
    attention_mask

    Returns
    -------

    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


pooling_methods = {
    "nvidia/NV-Embed-v1": "model_specific_encoder",
    "Salesforce/SFR-Embedding-2_R": "last_token_pool",
}



def count_lines(path: str):
    """Counts the number of lines in a file."""
    return sum(1 for _ in open(path))


def generate_batch(docs: Union[List, Generator], batch_size: int) -> Generator:
    texts = []

    for doc in docs:
        texts.append(doc["text"])

        if len(texts) == batch_size:
            yield texts
            texts = []

    if texts:
        yield texts


class Encoder:
    def __init__(
            self,
            index_name: str = "new-index",
            model: str = "sentence-transformers/all-MiniLM-L6-v2",
            normalize: bool = True,
            return_numpy: bool = True,
            max_length: int = None,
            device: str = "cpu",
            hidden_size: int = None,
            pooling_method: str = None,
            transformers_cache_dir: str = None,
    ):

        if index_name is not None:
            # this is if we want to initialize an index without an encoder.
            logging.info(f"Initializing MyEncoder with model: {model}")
            ind_path = index_path(index_name)
            logging.info(f"Collections Path: {ind_path}")
            self.index_name = index_name

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, cache_dir=transformers_cache_dir)
        self.encoder = AutoModel.from_pretrained(
            model,
            trust_remote_code=True, 
            cache_dir=transformers_cache_dir,
            device_map=device
        ).eval()
        config = AutoConfig.from_pretrained(model, trust_remote_code=True, cache_dir=transformers_cache_dir)

        # Set the hidden size based on the model configuration if not provided.
        if hasattr(config, 'hidden_size') and hidden_size is None:
            self.embedding_dim = config.hidden_size
        elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size') and hidden_size is None:
            self.embedding_dim = config.text_config.hidden_size
        else:
            self.embedding_dim = hidden_size

        # Set the maximum length based on the model configuration if not provided.
        if hasattr(config, 'max_position_embeddings') and max_length is None:
            self.max_length = config.max_position_embeddings
        elif (
                hasattr(config, 'text_config') and
                hasattr(config.text_config, 'max_position_embeddings') and
                max_length is None
        ):
            self.max_length = config.text_config.max_position_embeddings
        else:
            self.max_length = max_length

        # Set the pooling method based on the model type if not provided.
        if pooling_method is None:
            self.pooling_method = pooling_methods.get(model, "mean_pooling")
        else:
            self.pooling_method = pooling_method

        self.normalize = normalize
        self.return_numpy = return_numpy
        self.device = device
        self.tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }

    def save(self):
        state = dict(
            index_name=self.index_name,
            model=self.model,
            normalize=self.normalize,
            return_numpy=self.return_numpy,
            max_length=self.max_length,
            device=self.device,
        )
        np.save(encoder_state_path(self.index_name), state)

    def embed(self, texts_to_embed: List):
        with torch.no_grad():
            # Encode the texts using the specified pooling method.
            if self.pooling_method == "mean_pooling":
                tokens = self.tokenize(texts_to_embed)
                emb = self.encoder(**tokens).last_hidden_state
                emb = self.mean_pooling(emb, tokens["attention_mask"])
            elif self.pooling_method == "model_specific_encoder":
                emb = self.encoder.encode(texts_to_embed)
            elif self.pooling_method == "last_token_pool":
                batch_dict = self.tokenizer(
                    texts_to_embed, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                outputs = self.encoder(**batch_dict)
                emb = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            # Normalize the embeddings if specified.
            if self.normalize:
                emb = normalize(emb, dim=-1)

        return emb

    @staticmethod
    def load(index_name: str, device: str = None):
        state = np.load(encoder_state_path(index_name), allow_pickle=True)[()]
        if device is not None:
            state["device"] = device
        return Encoder(**state)

    def change_device(self, device: str = "cpu"):
        self.device = device
        self.encoder.to(device)

    def tokenize(self, texts: List[str]):
        tokens = self.tokenizer(texts, **self.tokenizer_kwargs)
        return {k: v.to(self.device) for k, v in tokens.items()}

    def mean_pooling(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        numerators = einsum("xyz,xy->xyz", embeddings, mask).sum(dim=1)
        denominators = torch.clamp(mask.sum(dim=1), min=1e-9)
        return einsum("xz,x->xz", numerators, 1 / denominators)

    def __call__(
        self,
        x: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = True,
    ):
        if isinstance(x, str):
            return self.encode(x)
        else:
            return self.bencode(x, batch_size=batch_size, show_progress=show_progress)

    def encode(self, text: str):
        return self.bencode([text], batch_size=1, show_progress=False)[0]


    def bencode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ):
        embeddings = []
        pbar = tqdm(
            total=len(texts),
            desc="Generating embeddings",
            disable=not show_progress,
            **pbar_kwargs,
        )
        for i in range(ceil(len(texts) / batch_size)):
            start, stop = i * batch_size, (i + 1) * batch_size
            emb = self.embed(texts[start:stop])
            embeddings.append(emb)
            pbar.update(stop - start)
        pbar.close()

        embeddings = torch.cat(embeddings)
        if self.return_numpy:
            embeddings = embeddings.detach().cpu().numpy()
        return embeddings

    def encode_collection(
        self,
        path: str = None,
        collection = None,
        batch_size: int = 512,
        callback: callable = None,
        show_progress: bool = True,
    ):
            """
                Helper function to embed a collection.
                Collection must be a list of dictionaries with {"text": ""}.
            """
        n_docs = count_lines(path)
        assert (collection is not None) and (path is not None), "Must pass in either `path` or `collection` to this function."
        if collection is None:
                collection = read_jsonl(path, callback=callback, generator=True)
        reservoir = np.empty((1_000_000, self.embedding_dim), dtype=np.float32)
        reservoir_n = 0
        offset = 0

        pbar = tqdm(
            total=n_docs,
            desc="Embedding documents",
            disable=not show_progress,
            **pbar_kwargs,
        )

        for texts in generate_batch(collection, batch_size):
            # Compute embeddings -----------------------------------------------
            embeddings = self.bencode(texts, batch_size=len(texts), show_progress=False)

            # Compute new offset -----------------------------------------------
            new_offset = offset + len(embeddings)

            if new_offset >= len(reservoir):
                np.save(
                    embeddings_folder_path(self.index_name)
                    / f"chunk_{reservoir_n}.npy",
                    reservoir[:offset],
                )
                reservoir = np.empty((1_000_000, self.embedding_dim), dtype=np.float32)
                reservoir_n += 1
                offset = 0
                new_offset = len(embeddings)

            # Save embeddings in the reservoir ---------------------------------
            reservoir[offset:new_offset] = embeddings

            # Update offeset ---------------------------------------------------
            offset = new_offset

            pbar.update(len(embeddings))

        if offset < len(reservoir):
            np.save(
                embeddings_folder_path(self.index_name) / f"chunk_{reservoir_n}.npy",
                reservoir[:offset],
            )
            reservoir = []

        assert len(reservoir) == 0, "Reservoir is not empty."
