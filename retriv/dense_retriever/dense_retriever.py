import os
import shutil
from typing import Dict, Iterable, List, Union, Callable
import faiss

import numpy as np
import orjson
from numba import njit, prange
from numba.typed import List as TypedList
from oneliner_utils import create_path
from tqdm import tqdm

from ..base_retriever import BaseRetriever
from ..paths import docs_path, dr_state_path, embeddings_folder_path
from .ann_searcher import ANN_Searcher
from .encoder import Encoder
from urllib.parse import urlparse, urljoin


def make_inverse_index_url(id_mapping):
    """
    Constructs an inverse index for the id_mapping, where the keys are urls + their cleaned versions
    and the values are the ids.

    """
    def clean_url(to_get_url):
        return urljoin(to_get_url, urlparse(to_get_url).path)

    inverse_index = {}
    for id, url in id_mapping.items():
        inverse_index[url] = id
        cleaned_url = clean_url(url)
        if cleaned_url not in inverse_index:
            inverse_index[cleaned_url] = id
        if cleaned_url[-1] == '/':
            cleaned_url = cleaned_url[:-1]
            if cleaned_url not in inverse_index:
                inverse_index[cleaned_url] = id

    return inverse_index


def get_search_params(subset_ids, index_info):
    """
    Helper function to configure and return the search parameters for a FAISS index based on provided index information.

    This function dynamically selects the appropriate FAISS search parameters class
    based on the type of index key indicated in `index_info`. It also parses additional
    parameters specified as a string and sets them appropriately.

    Parameters:
    subset_ids (list[int]): A list of subset IDs for which search parameters are to be configured.
    index_info (dict): Dictionary containing the index key and additional parameter string.
                       Expected keys are 'index_key' and 'index_param'.

    Returns:
    object: An instance of a FAISS search parameters class configured with the provided selector
            and additional parameters.
    """
    # Extract the index key from the index information dictionary.
    index_key = index_info["index_key"]

    # Choose the appropriate search parameters class based on the index key.
    params_class = (
        faiss.SearchParametersIVF if "IVF" in index_key else
        faiss.SearchParametersPQ if "PQ" in index_key else
        faiss.SearchParametersHNSW if 'HNSW' in index_key else
        faiss.SearchParameters
    )
    # Retrieve additional index parameters from the index information dictionary.
    index_kwargs = index_info['index_param']
    # Initialize an empty dictionary to store additional parameters.
    add_params = {}
    # Split the index parameters string by commas and iterate over each parameter.
    for p in index_kwargs.split(','):
        # Split each parameter into key and value pairs on the '=' character.
        k, v = p.split('=')
        # If the value is a digit, convert it from string to integer.
        if v.isdigit():
            v = int(v)
        add_params[k] = v

    # Create a selector for a batch of IDs using the provided subset_ids.
    sel = faiss.IDSelectorBatch(subset_ids)
    return params_class(sel=sel, **add_params)


class DenseRetriever(BaseRetriever):
    def __init__(
            self,
            index_name: str = "new-index",
            model: str = "sentence-transformers/all-MiniLM-L6-v2",
            normalize: bool = True,
            max_length: int = None,
            embedding_dim: int = None,
            use_ann: bool = True,
            make_inverse_index: Union[None, Callable] = None,
            device: str = "cpu",
            *args,
            **kwargs
    ):
        """Initialize MyDenseRetriever with the option to create an inverse index.

        Args:
            index_name (str, optional): [retriv](https://github.com/AmenRa/retriv) will use `index_name` as the identifier of your index. Defaults to "new-index".
            model (str, optional): defines the encoder model to encode queries and documents into vectors. You can use an [HuggingFace's Transformers](https://huggingface.co/models) pre-trained model by providing its ID or load a local model by providing its path.  In the case of local models, the path must point to the directory containing the data saved with the [`PreTrainedModel.save_pretrained`](https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) method. Note that the representations are computed with `mean pooling` over the `last_hidden_state`. Defaults to "sentence-transformers/all-MiniLM-L6-v2".
            normalize (bool, optional): whether to L2 normalize the vector representations. Defaults to True.
            max_length (int, optional): texts longer than `max_length` will be automatically truncated. Choose this parameter based on how the employed model was trained or is generally used. Defaults to 128.
            use_ann (bool, optional): whether to use approximate nearest neighbors search. Set it to `False` to use nearest neighbors search without approximation. If you have less than 20k documents in your collection, you probably want to disable approximation. Defaults to True.
            make_inverse_index: a callable that takes a dictionary and returns its inverse. Defaults to None.
        """
        self.index_name = index_name
        self.model = model
        self.normalize = normalize
        self.use_ann = use_ann
        self.device = device

        self.encoder = Encoder(
            index_name=index_name,
            model=model,
            normalize=normalize,
            max_length=max_length,
            hidden_size=embedding_dim,
            device=device,
        )

        if max_length is None:
            self.max_length = self.encoder.max_length
        else:
            self.max_length = max_length

        self.ann_searcher = ANN_Searcher(index_name=index_name)
        self.id_mapping = None
        self.doc_count = None
        self.doc_index = None
        self.embeddings = None
        self.id_mapping_reverse = None
        if make_inverse_index is None:
            self.make_inverse_index = lambda x: {v: k for k, v in x.items()}
        else:
            self.make_inverse_index = kwargs['make_inverse_index']


    def save(self):
        """Save the state of the retriever to be able to restore it later."""

        state = dict(
            init_args=dict(
                index_name=self.index_name,
                model=self.model,
                normalize=self.normalize,
                max_length=self.max_length,
                use_ann=self.use_ann,
            ),
            id_mapping=self.id_mapping,
            doc_count=self.doc_count,
            embeddings=True if self.embeddings is not None else None,
        )
        np.savez_compressed(dr_state_path(self.index_name), state=state)

    @staticmethod
    @staticmethod
    def load(
            index_name: str = "new-index",
            make_inverse_index: Union[None, Callable] = None,
            *args,
            **kwargs
    ):
        """Static method to load a previously saved index and its associated retriever.

        Args:
            index_name (str, optional): Name of the index to load. Defaults to "new-index".
            make_inverse_index (Union[None, Callable], optional): Function to generate an inverse index.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            MyDenseRetriever: The loaded dense retriever object.
        """
        state = np.load(dr_state_path(index_name), allow_pickle=True)["state"][()]
        dr = DenseRetriever(**state["init_args"])
        dr.initialize_doc_index()
        dr.id_mapping = state["id_mapping"]
        dr.doc_count = state["doc_count"]
        if state["embeddings"]:
            dr.load_embeddings()
        if dr.use_ann:
            dr.ann_searcher = ANN_Searcher.load(index_name)

        if 'id_mapping_reverse' not in state:
            if make_inverse_index is None:
                dr.id_mapping_reverse = {v: k for k, v in dr.id_mapping.items()}
            else:
                dr.id_mapping_reverse = make_inverse_index(dr.id_mapping)
        else:
            dr.id_mapping_reverse = state['id_mapping_reverse']

        return dr


    def load_embeddings(self):
        """Internal usage."""
        path = embeddings_folder_path(self.index_name)
        npy_file_paths = sorted(os.listdir(path))
        self.embeddings = np.concatenate(
            [np.load(path / npy_file_path) for npy_file_path in npy_file_paths]
        )

    def import_embeddings(self, path: str):
        """Internal usage."""
        shutil.copyfile(path, embeddings_folder_path(self.index_name) / "chunk_0.npy")

    def index_aux(
        self,
        embeddings_path: str = None,
        use_gpu: bool = False,
        batch_size: int = 512,
        callback: callable = None,
        show_progress: bool = True,
    ):
        """Internal usage."""
        if embeddings_path is not None:
            self.import_embeddings(embeddings_path)
        else:
            self.encoder.change_device("cuda" if use_gpu else "cpu")
            self.encoder.encode_collection(
                path=docs_path(self.index_name),
                batch_size=batch_size,
                callback=callback,
                show_progress=show_progress,
            )
            self.encoder.change_device("cpu")

        if self.use_ann:
            if show_progress:
                print("Building ANN Searcher")
            self.ann_searcher.build()
        else:
            if show_progress:
                print("Loading embeddings...")
            self.load_embeddings()

    def index(
        self,
        collection: Iterable,
        callback: callable = None,
        show_progress: bool = True,
        batch_size: int = 1,
    ):
        """Indexes the provided collection and generates an inverse index mapping.

        Args:
            collection (Iterable): The collection of documents to index.
            callback (callable, optional): A callback function for progress updates. Defaults to None.
            show_progress (bool, optional): Flag to show progress of indexing. Defaults to True.
        """
        if self.device == 'cuda':
            use_gpu = True
        else:
            use_gpu = False

        super().index(
            collection=collection,
            callback=callback,
            show_progress=show_progress,
            batch_size=batch_size,
            use_gpu=use_gpu
        )
        self.id_mapping_reverse = self.make_inverse_index(self.id_mapping)

    def index_file(
        self,
        path: str,
        embeddings_path: str = None,
        use_gpu: bool = False,
        batch_size: int = 512,
        callback: callable = None,
        show_progress: bool = True,
    ):
        """Index the collection contained in a given file.

        Args:
            path (str): path of file containing the collection to index.

            embeddings_path (str, optional): in case you want to load pre-computed embeddings, you can provide the path to a `.npy` file. Embeddings must be in the same order as the documents in the collection file. Defaults to None.

            use_gpu (bool, optional): whether to use the GPU for document encoding. Defaults to False.

            batch_size (int, optional): how many documents to encode at once. Regulate it if you ran into memory usage issues or want to maximize throughput. Defaults to 512.

            callback (callable, optional): callback to apply before indexing the documents to modify them on the fly if needed. Defaults to None.

            show_progress (bool, optional): whether to show a progress bar for the indexing process. Defaults to True.

        Returns:
            DenseRetriever: Dense Retriever.
        """

        collection = self.collection_generator(path, callback)
        return self.index(
            collection,
            embeddings_path,
            use_gpu,
            batch_size,
            None,
            show_progress,
        )

    def search(
            self,
            query: str,
            include_id_list: List[str]=None,
            return_docs: bool = True,
            cutoff: int = 100,
            verbose: bool = False,
    ):
        """Searches the indexed collection using the given query.

        Args:
            query (str): The query string to search for.
            include_id_list (List[str], optional): List of ids to include in the search. Defaults to None.
            return_docs (bool, optional): Whether to return full documents or just ids and scores. Defaults to True.
            cutoff (int, optional): The number of results to return. Defaults to 100.
            verbose (bool, optional): If set to True, outputs additional log messages. Defaults to False.

        Returns:
            Either a list of documents or a dictionary of ids and their corresponding scores, based on return_docs.
        """
        encoded_query = self.encoder(query)
        encoded_query = encoded_query.reshape(1, len(encoded_query))

        if self.use_ann:
            if include_id_list is not None:
                internal_subset_ids = []
                for reverse_id in include_id_list:
                    if reverse_id in self.id_mapping_reverse:
                        internal_subset_ids.append(self.id_mapping_reverse[reverse_id])
                    else:
                        if verbose:
                            logger.warning(f'Warning: {reverse_id} not in id_mapping')
                search_params = get_search_params(internal_subset_ids, self.ann_searcher.faiss_index_infos)
            else:
                search_params = None
            scores, doc_ids = self.ann_searcher.faiss_index.search(encoded_query, cutoff, params=search_params)
            doc_ids, scores = doc_ids[0], scores[0]
            to_keep = list(filter(lambda x: x[0] != -1, zip(doc_ids, scores)))
            if len(to_keep) > 0:
                doc_ids, scores = zip(*to_keep)
            else:
                doc_ids, scores = [], []
        else:
            raise NotImplementedError('use ANN....')

        doc_ids = self.map_internal_ids_to_original_ids(doc_ids)
        return (
            self.prepare_results(doc_ids, scores)
            if return_docs
            else dict(zip(doc_ids, scores))
        )

    def msearch(
        self,
        queries: List[Dict[str, str]],
        cutoff: int = 100,
        batch_size: int = 32,
    ) -> Dict:
        """Compute results for multiple queries at once.

        Args:
            queries (List[Dict[str, str]]): what to search for.

            cutoff (int, optional): number of results to return. Defaults to 100.

            batch_size (int, optional): how many queries to search at once. Regulate it if you ran into memory usage issues or want to maximize throughput. Defaults to 32.

        Returns:
            Dict: results.
        """

        q_ids = [x["id"] for x in queries]
        q_texts = [x["text"] for x in queries]
        encoded_queries = self.encoder(q_texts, batch_size, show_progress=False)

        if self.use_ann:
            doc_ids, scores = self.ann_searcher.msearch(encoded_queries, cutoff)
        else:
            if self.embeddings is None:
                self.load_embeddings()
            doc_ids, scores = compute_scores_multi(
                encoded_queries, self.embeddings, cutoff
            )

        doc_ids = [
            self.map_internal_ids_to_original_ids(_doc_ids) for _doc_ids in doc_ids
        ]

        results = {q: dict(zip(doc_ids[i], scores[i])) for i, q in enumerate(q_ids)}

        return {q_id: results[q_id] for q_id in q_ids}

    def bsearch(
        self,
        queries: List[Dict[str, str]],
        cutoff: int = 100,
        batch_size: int = 32,
        show_progress: bool = True,
        qrels: Dict[str, Dict[str, float]] = None,
        path: str = None,
    ):
        """Batch-Search is similar to Multi-Search but automatically generates batches of queries to evaluate and allows dynamic writing of the search results to disk in [JSONl](https://jsonlines.org) format. bsearch is handy for computing results for hundreds of thousands or even millions of queries without hogging your RAM.

        Args:
            queries (List[Dict[str, str]]): what to search for.

            cutoff (int, optional): number of results to return. Defaults to 100.

            batch_size (int, optional): how many queries to search at once. Regulate it if you ran into memory usage issues or want to maximize throughput. Defaults to 32.

            show_progress (bool, optional): whether to show a progress bar for the search process. Defaults to True.

            qrels (Dict[str, Dict[str, float]], optional): query relevance judgements for the queries. Defaults to None.

            path (str, optional): where to save the results. Defaults to None.

        Returns:
            Dict: results.
        """

        batches = [
            queries[i : i + batch_size] for i in range(0, len(queries), batch_size)
        ]

        results = {}

        pbar = tqdm(
            total=len(queries),
            disable=not show_progress,
            desc="Batch search",
            dynamic_ncols=True,
            mininterval=0.5,
        )

        if path is None:
            for batch in batches:
                new_results = self.msearch(
                    queries=batch, cutoff=cutoff, batch_size=len(batch)
                )
                results = {**results, **new_results}
                pbar.update(min(batch_size, len(batch)))
        else:
            path = create_path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "wb") as f:
                for batch in batches:
                    new_results = self.msearch(queries=batch, cutoff=cutoff)

                    for i, (k, v) in enumerate(new_results.items()):
                        x = {
                            "id": k,
                            "text": batch[i]["text"],
                            "dense_doc_ids": list(v.keys()),
                            "dense_scores": [float(s) for s in list(v.values())],
                        }
                        if qrels is not None:
                            x["rel_doc_ids"] = list(qrels[k].keys())
                            x["rel_scores"] = list(qrels[k].values())
                        f.write(orjson.dumps(x) + "\n".encode())

                    pbar.update(min(batch_size, len(batch)))

        return results


@njit(cache=True)
def compute_scores(query: np.ndarray, docs: np.ndarray, cutoff: int):
    """Internal usage."""

    scores = docs @ query
    indices = np.argsort(-scores)[:cutoff]

    return indices, scores[indices]


@njit(cache=True, parallel=True)
def compute_scores_multi(queries: np.ndarray, docs: np.ndarray, cutoff: int):
    """Internal usage."""

    n = len(queries)
    ids = TypedList([np.empty(1, dtype=np.int64) for _ in range(n)])
    scores = TypedList([np.empty(1, dtype=np.float32) for _ in range(n)])

    for i in prange(len(queries)):
        ids[i], scores[i] = compute_scores(queries[i], docs, cutoff)

    return ids, scores
