import asyncio
import html
import os
from dataclasses import dataclass
from typing import Any, Union, cast
import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB
from motor.motor_asyncio import AsyncIOMotorClient
from .utils import load_json, logger, write_json
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)
import faiss
from .llm import openai_embedding
from typing import Dict, List
import logging
logger = logging.getLogger(__name__)
import hashlib



@dataclass

@dataclass
class MongoKVStorage(BaseKVStorage):
    def __post_init__(self):
        self.client = None  # Client will be initialized lazily
        logger.info(f"Initialized MongoKVStorage for namespace {self.namespace}")

    def get_collection(self):
        if self.client is None:
            mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            self.client = AsyncIOMotorClient(mongodb_uri)
            self.db = self.client["LightRAG"]
            self.collection = self.db[f"kv_store_{self.namespace}"]
            logger.info(f"Connected to MongoDB collection kv_store_{self.namespace}")
        return self.collection

    async def all_keys(self) -> List[str]:
        collection = self.get_collection()
        return await collection.distinct("_id")

    async def index_done_callback(self):
        logger.info(f"Indexing completed for {self.namespace}")

    async def get_by_id(self, id):
        collection = self.get_collection()
        return await collection.find_one({"_id": id})

    async def get_by_ids(self, ids, fields=None):
        collection = self.get_collection()
        projection = {field: 1 for field in fields} if fields else None
        cursor = collection.find({"_id": {"$in": ids}}, projection)
        return [doc async for doc in cursor]

    async def filter_keys(self, data: List[str]) -> set:
        existing_ids = await self.all_keys()
        return set(data) - set(existing_ids)

    async def upsert(self, data: Dict[str, dict]):
        collection = self.get_collection()
        operations = []
        for key, value in data.items():
            operations.append(
                collection.update_one(
                    {"_id": key}, {"$set": value}, upsert=True
                )
            )
        results = await asyncio.gather(*operations)
        logger.info(f"Upserted {len(results)} documents")
        return data

    async def drop(self):
        collection = self.get_collection()
        await collection.drop()
        logger.info(f"Dropped collection {self.namespace}")


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        operations = []
        for key, value in data.items():
            operations.append(
                self.collection.update_one(
                    {"_id": key}, {"$set": value}, upsert=True
                )
            )
        results = await asyncio.gather(*operations)
        logger.info(f"Upserted {len(results)} documents")
        return data

    async def drop(self):
        self._data = {}


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()


@dataclass
@dataclass
class MongoFAISSVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self.client = None  # Client will be initialized lazily

        # FAISS setup
        self.embedding_dim = self.embedding_func.embedding_dim
        self.faiss_index_file = os.path.join(
            self.global_config["working_dir"], f"faiss_{self.namespace}.index"
        )
        if os.path.exists(self.faiss_index_file):
            self.index = faiss.read_index(self.faiss_index_file)
            logger.info(f"Loaded FAISS index from {self.faiss_index_file}")
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
            logger.info(f"Created new FAISS index for {self.namespace}")

        self._max_batch_size = self.global_config["embedding_batch_num"]
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    def get_collection(self):
        if self.client is None:
            mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            self.client = AsyncIOMotorClient(mongodb_uri)
            self.db = self.client["LightRAG"]
            self.collection = self.db[self.namespace]
            logger.info(f"Connected to MongoDB collection {self.namespace}")
        return self.collection

    async def upsert(self, data: Dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not data:
            logger.warning("You inserted empty data into the vector DB.")
            return []
        collection = self.get_collection()

        # Prepare the data for MongoDB
        list_data = [
            {
                "_id": k,  # MongoDB uses _id for primary key
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]

        # Extract content and compute embeddings
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)

        # Ensure embeddings are contiguous and of type float32
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Normalize embeddings
        faiss.normalize_L2(embeddings)

        # Convert _id to int64 IDs for FAISS
        ids = [
            int(hashlib.sha1(d["_id"].encode()).hexdigest(), 16) % (2**63)
            for d in list_data
        ]

        # Insert embeddings into FAISS index with IDs
        self.index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))

        # Insert metadata into MongoDB
        operations = []
        for d, idx in zip(list_data, ids):
            d["faiss_id"] = int(idx)
            operations.append(
                collection.update_one(
                    {"_id": d["_id"]}, {"$set": d}, upsert=True
                )
            )
        await asyncio.gather(*operations)
        return list_data

    async def query(self, query: str, top_k=5):
        collection = self.get_collection()
        # Get the embedding of the query
        embedding = await self.embedding_func([query])
        embedding = np.array(embedding[0], dtype=np.float32)

        # Normalize the embedding
        faiss.normalize_L2(embedding.reshape(1, -1))

        # Perform the vector search using FAISS
        distances, indices = self.index.search(embedding.reshape(1, -1), top_k)

        # Filter results by threshold
        filtered_results = [
            (idx, dist)
            for idx, dist in zip(indices[0], distances[0])
            if dist > self.cosine_better_than_threshold
        ]

        # Retrieve metadata from MongoDB for the closest vectors
        tasks = [
            collection.find_one({"faiss_id": int(idx)}) for idx, _ in filtered_results
        ]
        metadatas = await asyncio.gather(*tasks)

        results = []
        for metadata, (_, distance) in zip(metadatas, filtered_results):
            if metadata:
                results.append({
                    "id": metadata["_id"],
                    "distance": distance,
                    **metadata
                })
        return results

    async def index_done_callback(self):
        # Save the FAISS index to disk
        faiss.write_index(self.index, self.faiss_index_file)
        logger.info(f"FAISS index saved to {self.faiss_index_file}")

    async def drop(self):
        collection = self.get_collection()
        await collection.drop()
        logger.info(f"Dropped collection {self.namespace}")


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {
            node: html.unescape(node.upper().strip()) for node in graph.nodes()
        }  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
