from dataclasses import dataclass, asdict

import datetime, time
from time import strftime, localtime
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
import numpy as np

from remembr.memory.memory import Memory, MemoryItem

from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient, DataType


FIXED_SUBTRACT=1721761000 # this is just a large value that brings us close to 1970
DIM = 1024


class MilvusWrapper:

    def __init__(self, collection_name='test', db_path='./remembr.db', drop_collection=False):
        self.collection_name = collection_name
        self.db_path = db_path
        self.client = MilvusClient(db_path)

        if drop_collection and self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)

        if not self.client.has_collection(collection_name):
            self._create_collection(collection_name, DIM)

    def drop_collection(self):
        self.client.drop_collection(self.collection_name)

    def _create_collection(self, collection_name, dim):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, max_length=1000, is_primary=True)
        schema.add_field("text_embedding", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("position", DataType.FLOAT_VECTOR, dim=3)
        schema.add_field("theta", DataType.FLOAT)
        schema.add_field("time", DataType.FLOAT_VECTOR, dim=2)
        schema.add_field("caption", DataType.VARCHAR, max_length=3000)

        index_params = self.client.prepare_index_params()
        index_params.add_index("text_embedding", index_type="FLAT", metric_type="L2")
        index_params.add_index("position", index_type="FLAT", metric_type="L2")
        index_params.add_index("time", index_type="FLAT", metric_type="L2")

        self.client.create_collection(collection_name, schema=schema, index_params=index_params)

    def insert(self, data_list):
        self.client.insert(self.collection_name, data_list)

    def search(self, data, anns_field="text_embedding", limit=10, output_fields=None):
        if output_fields is None:
            output_fields = ["id", "caption", "position", "time", "theta"]
        res = self.client.search(
            collection_name=self.collection_name,
            data=[data],
            anns_field=anns_field,
            limit=limit,
            output_fields=output_fields,
        )
        return res


class MilvusMemory(Memory):

    def __init__(self, db_collection_name: str, db_path='./remembr.db', db_ip=None, db_port=19530, time_offset=FIXED_SUBTRACT):
        self.db_collection_name = db_collection_name
        self.db_path = db_path
        self.time_offset = time_offset

        self.embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
        self.working_memory = []
        self.reset(drop_collection=False)

    def insert(self, item: MemoryItem, text_embedding=None):
        memory_dict = asdict(item)
        memory_dict['id'] = str(time.time())

        if text_embedding is None:
            text_embedding = self.embedder.embed_query(memory_dict['caption'])

        memory_dict['time'] = [(memory_dict['time'] - self.time_offset), 0.0]
        memory_dict['text_embedding'] = text_embedding
        self.milv_wrapper.insert([memory_dict])

    def get_working_memory(self) -> list[MemoryItem]:
        return self.working_memory

    def reset(self, drop_collection=True):
        if drop_collection:
            print("Resetting memory. We are dropping the current collection")
        self.milv_wrapper = MilvusWrapper(self.db_collection_name, self.db_path, drop_collection=drop_collection)

    def _results_to_documents(self, results) -> List[Document]:
        docs = []
        for hits in results:
            for hit in hits:
                entity = hit.get('entity', {})
                doc = Document(
                    page_content=entity.get('caption', ''),
                    metadata={
                        'position': entity.get('position', [0, 0, 0]),
                        'time': entity.get('time', [0, 0]),
                        'theta': entity.get('theta', 0),
                    }
                )
                docs.append(doc)
        return docs

    def search_by_position(self, query: tuple) -> str:
        results = self.milv_wrapper.search(
            np.array(query).astype(float).tolist(),
            anns_field="position",
            limit=4,
        )
        docs = self._results_to_documents(results)
        self.working_memory += docs
        return self.memory_to_string(docs)

    def search_by_time(self, hms_time: str) -> str:
        t = localtime(self.time_offset)
        mdy_date = strftime('%m/%d/%Y', t)
        template = "%m/%d/%Y %H:%M:%S"

        try:
            res = bool(datetime.datetime.strptime(hms_time, template))
        except ValueError:
            res = False

        hms_time = hms_time.strip()
        if not res:
            hms_time = mdy_date + ' ' + hms_time

        query = time.mktime(datetime.datetime.strptime(hms_time, template).timetuple()) - self.time_offset

        results = self.milv_wrapper.search(
            [query, 0.0],
            anns_field="time",
            limit=4,
        )
        docs = self._results_to_documents(results)
        self.working_memory += docs
        return self.memory_to_string(docs)

    def search_by_text(self, query: str) -> str:
        query_embedding = self.embedder.embed_query(query)
        results = self.milv_wrapper.search(
            query_embedding,
            anns_field="text_embedding",
            limit=5,
        )
        docs = self._results_to_documents(results)
        self.working_memory += docs
        return self.memory_to_string(docs)

    def memory_to_string(self, memory_list: List[Document], ref_time: float = None):
        if ref_time is None:
            ref_time = self.time_offset

        out_string = ""
        for doc in memory_list:
            time_val = doc.metadata.get('time', [0, 0])
            if isinstance(time_val, list) and len(time_val) >= 1:
                t = time_val[0]
            else:
                t = time_val

            if ref_time:
                t += ref_time
            t = localtime(t)
            t = strftime('%Y-%m-%d %H:%M:%S', t)

            s = f"At time={t}, the robot was at an average position of {np.array(doc.metadata.get('position', [0,0,0])).round(3).tolist()}."
            s += f"The robot saw the following: {doc.page_content}\n\n"
            out_string += s
        return out_string
