import json
from pathlib import Path
from selectors import EpollSelector
from datapipe.run_config import RunConfig
from datapipe.store.database import DBConn, TableStoreDB
from datapipe.store.table_store import TableStore
from datapipe.types import DataDF, DataSchema, IndexDF, MetaSchema, data_to_index
import numpy as np
import fsspec
from typing import Any, Dict, IO, Iterator, Optional, Tuple, List, Type, Union

from datapipe.store.filedir import (
    ItemStoreFileAdapter, TableStoreFiledir, _pattern_to_attrnames
)
import pandas as pd
from sqlalchemy import JSON, Column
from cv_pipeliner.core.data import ImageData, BboxData
from cv_pipeliner.utils.imagesize import get_image_size


class ImageDataFile(ItemStoreFileAdapter):
    '''
    Converts each ImageData file into Pandas record
    '''

    mode = 't'

    def __init__(
        self,
        image_data_cls: Type[ImageData] = ImageData,
        bbox_data_cls: Type[BboxData] = BboxData
    ):
        self.image_data_cls = image_data_cls
        self.bbox_data_cls = bbox_data_cls

    def load(self, f: IO) -> ImageData:
        image_data_json = json.load(f)
        image_data = ImageData.from_json(
            image_data_json,
            image_data_cls=self.image_data_cls,
            bbox_data_cls=self.bbox_data_cls
        ) if image_data_json is not None else None
        return {'image_data': image_data}

    def dump(self, obj: Dict[str, Any], f: IO) -> None:
        image_data: ImageData = obj['image_data']
        return json.dump(image_data.json() if image_data is not None else None, f, indent=4, ensure_ascii=False)


class NumpyDataFile(ItemStoreFileAdapter):
    '''
    Converts each npy file into Pandas record
    '''

    mode = 'b'

    def load(self, f: IO) -> np.ndarray:
        ndarray = np.load(f)
        return {'ndarray': ndarray}

    def dump(self, obj: Dict[str, Any], f: IO) -> None:
        ndarray: np.ndarray = obj['ndarray']
        return np.save(f, ndarray)


class GetImageSizeFile(ItemStoreFileAdapter):
    mode = 'b'

    def load(self, f: IO) -> Dict[str, Tuple[int, int]]:
        image_size = get_image_size(f)
        return {'image_size': image_size}

    def dump(self, obj: Dict[str, Tuple[int, int]], f: IO) -> None:
        raise NotImplementedError


class COCOLabelsFile(ItemStoreFileAdapter):
    mode = 'b'

    def __init__(self, class_names: List[str], img_format: str):
        from cv_pipeliner.data_converters.coco import COCODataConverter
        self.coco_converter = COCODataConverter(class_names)
        self.img_format = img_format

    def load(self, f: fsspec.core.OpenFile) -> Dict[str, Tuple[int, int]]:
        filepath = Path(f.path)
        assert filepath.parent.name == 'labels'
        image_path = filepath.parent.parent / 'images' / f"{filepath.stem}.{self.img_format}"
        image_data = self.coco_converter.get_image_data_from_annot(
            image_path=f"{f.fs.protocol}://{image_path}", annot=f
        )
        return {'image_size': image_data}

    def dump(self, obj: Dict[str, Tuple[int, int]], f: fsspec.core.OpenFile) -> None:
        image_data: ImageData = obj['image_data']
        coco_data = self.coco_converter.get_annot_from_image_data(image_data)
        f.write('\n'.join(coco_data).encode())


class ImageDataTableStoreDB(TableStoreDB):
    def __init__(
        self,
        dbconn: Union[DBConn, str],
        name: str,
        data_sql_schema: List[Column],
        create_table: bool = True,
        image_data_cls: Type[ImageData] = ImageData,
        bbox_data_cls: Type[BboxData] = BboxData
    ) -> None:
        assert all([column.primary_key for column in data_sql_schema])
        assert 'image_data' not in [column.name for column in data_sql_schema]
        data_sql_schema += [Column('image_data', JSON)]
        super().__init__(
            dbconn=dbconn,
            name=name,
            data_sql_schema=data_sql_schema,
            create_table=create_table
        )
        self.image_data_cls = image_data_cls
        self.bbox_data_cls = bbox_data_cls

    def insert_rows(self, df: DataDF) -> None:
        df['image_data'] = df['image_data'].apply(
            lambda image_data: image_data.json() if image_data is not None else None
        )
        super().insert_rows(df)

    def read_rows(self, idx: Optional[IndexDF] = None) -> pd.DataFrame:
        df = super().read_rows(idx=idx)
        df['image_data'] = df['image_data'].apply(
            lambda image_data_json: ImageData.from_json(
                image_data_json,
                image_data_cls=self.image_data_cls,
                bbox_data_cls=self.bbox_data_cls
            ) if image_data_json is not None else None
        )
        return df


class EmptyItemStoreFileAdapter(ItemStoreFileAdapter):
    mode = 'b'


class ConnectedImageDataTableStore(TableStore):
    def __init__(
        self,
        images_table_store: TableStoreFiledir,
        images_data_table_store: Union[ImageDataTableStoreDB, TableStoreFiledir]
    ) -> None:
        if isinstance(images_data_table_store, TableStoreFiledir):
            assert isinstance(images_data_table_store.adapter, ImageDataFile)
        self.images_data_store = images_data_table_store
        self.images_store = images_table_store
        for key in self.images_store.primary_keys:
            assert key in self.images_data_store.primary_keys, f"Missing key for images_data_store: {key}"
        self.attrnames = self.images_store.attrnames
        self.primary_schema = self.images_store.primary_schema

    def get_primary_schema(self) -> DataSchema:
        return [column for column in self.primary_schema if column.primary_key]

    def get_meta_schema(self) -> MetaSchema:
        return []

    def delete_rows(self, idx: IndexDF) -> None:
        self.images_data_store.delete_rows(idx)

    def _set_images_paths(self, df: DataDF, force_update_meta: bool = False) -> None:
        for row_idx in df.index:
            if df.loc[row_idx, 'image_data'] is None:
                continue
            idxs_values = df.loc[row_idx, self.attrnames].tolist()
            image_path = self.images_store._filenames_from_idxs_values(idxs_values)[0]
            df.loc[row_idx, 'image_data'].image_path = image_path
            if force_update_meta:
                df.loc[row_idx, 'image_data'].get_image_size()

    def insert_rows(self, df: DataDF, force_update_meta: bool = False) -> None:
        self._set_images_paths(df, force_update_meta=force_update_meta)
        self.images_data_store.insert_rows(df)

    def update_rows(self, df: DataDF, force_update_meta: bool = False) -> None:
        if df.empty:
            return
        self.delete_rows(data_to_index(df, self.primary_keys))
        self.insert_rows(df, force_update_meta=True)

    def read_rows(self, idx: IndexDF = None) -> DataDF:
        df_images_data = self.images_data_store.read_rows(idx)
        self._set_images_paths(df_images_data)
        return df_images_data

    def read_rows_meta_pseudo_df(self, chunksize: int = 1000, run_config: RunConfig = None) -> Iterator[DataDF]:
        # FIXME сделать честную чанкированную реализацию во всех сторах
        for df_meta in self.images_data_store.read_rows_meta_pseudo_df(chunksize=chunksize, run_config=run_config):
            yield df_meta
