import json
from pathlib import Path
from threading import Semaphore
from datapipe.run_config import RunConfig
from datapipe.store.database import DBConn, TableStoreDB
from datapipe.store.table_store import TableStore
from datapipe.types import (
    DataDF, DataSchema, IndexDF, MetaSchema, data_to_index, index_difference, index_intersection, index_to_data
)
import numpy as np
import fsspec
from typing import Any, Dict, IO, Iterator, Optional, Tuple, List, Type, Union, Callable

from datapipe.store.filedir import (
    ItemStoreFileAdapter, TableStoreFiledir, _pattern_to_attrnames
)
import pandas as pd
from sqlalchemy import JSON, Column, String, Integer
from cv_pipeliner.core.data import ImageData, BboxData
from cv_pipeliner.utils.imagesize import get_image_size

from cv_pipeliner.utils.fiftyone import FifyOneSession


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
        if f.fs.protocol in ['file'] or f.fs.protocol is None:
            prefix = ''
        else:
            protocol = f.fs.protocol
            if isinstance(protocol, tuple):
                protocol = protocol[0]
            prefix = f'{protocol}://'
        image_data = self.coco_converter.get_image_data_from_annot(
            image_path=f"{prefix}{image_path}", annot=f
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
        self.images_table_store = images_table_store
        for key in self.images_table_store.primary_keys:
            assert key in self.images_data_store.primary_keys, f"Missing key for images_data_store: {key}"
        self.attrnames = self.images_table_store.attrnames
        self.primary_schema = self.images_data_store.get_primary_schema()

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
            image_path_candidates = self.images_table_store._filenames_from_idxs_values(idxs_values)
            if len(image_path_candidates) > 1:
                for image_path in image_path_candidates:
                    openfile = fsspec.open(image_path)
                    if openfile.fs.exists(openfile.path):
                        break
            else:
                image_path = image_path_candidates[0]
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


class FiftyOneImagesDataTableStore(TableStore):
    def __init__(
        self,
        dataset: str,
        fo_session: FifyOneSession,
        fo_detections_label: Optional[str] = None,
        fo_classification_label: Optional[str] = None,
        fo_keypoints_label: Optional[str] = None,
        primary_schema: DataSchema = None,
        images_table_store: Optional[TableStoreFiledir] = None,
        mapping_filepath: Callable[[str], str] = lambda filepath: filepath,
        inverse_mapping_filepath: Callable[[str], str] = lambda filepath: filepath,
        rm_only_fo_fields: bool = True,
        additional_info_keys_in_fo_detections: List[str] = [],
        additional_info_keys_in_sample: List[str] = [],
        create_dataset_if_empty: bool = True,
        image_data_cls: Type[ImageData] = ImageData,
        bbox_data_cls: Type[BboxData] = BboxData
    ):
        self.dataset = dataset
        self.fo_detections_label = fo_detections_label
        self.fo_classification_label = fo_classification_label
        self.fo_keypoints_label = fo_keypoints_label
        self.fo_session = fo_session
        self.images_table_store = images_table_store
        self.mapping_filepath = mapping_filepath
        self.inverse_mapping_filepath = inverse_mapping_filepath
        self.rm_only_fo_fields = rm_only_fo_fields
        self.additional_info_keys_in_fo_detections = additional_info_keys_in_fo_detections
        self.additional_info_keys_in_sample = additional_info_keys_in_sample
        self.image_data_cls = image_data_cls
        self.bbox_data_cls = bbox_data_cls
        if primary_schema is not None:
            assert all([isinstance(column.type, (String, Integer)) for column in primary_schema])
            self.primary_schema = primary_schema
        else:
            self.primary_schema = [
                Column('filepath', String(100), primary_key=True)
            ]
        self.attrnames = [column.name for column in self.primary_schema]
        if self.images_table_store is not None:
            for key in self.images_table_store.primary_keys:
                assert key in self.attrnames, f"Missing key for images_data_store: {key}"
        assert 'id' not in self.attrnames, "The key 'id' is reserved for this TableStore. Use other key name instead."
        self.attrnames_no_filepath = [attrname for attrname in self.attrnames if attrname != 'filepath']
        self.fo_dataset = None
        self.create_dataset_if_empty = create_dataset_if_empty
        self.semaphore = Semaphore(value=1)

    def get_primary_schema(self) -> DataSchema:
        return self.primary_schema

    def get_meta_schema(self) -> MetaSchema:
        return []

    def _get_dataset(self):
        fo_dataset = None
        datasets = self.fo_session.fiftyone.list_datasets()
        if self.dataset in datasets:
            fo_dataset = self.fo_session.fiftyone.load_dataset(self.dataset)
        return fo_dataset

    def _get_or_create_dataset(self):
        if self.fo_dataset is not None:
            return self.fo_dataset

        self.semaphore.acquire()
        try:
            datasets = self.fo_session.fiftyone.list_datasets()
            if self.dataset in datasets:
                self.fo_dataset = self.fo_session.fiftyone.load_dataset(self.dataset)
            else:
                if self.create_dataset_if_empty:
                    self.fo_dataset = self.fo_session.fiftyone.Dataset(self.dataset)
                    self.fo_dataset.persistent = True
                else:
                    return ValueError("Dataset {self.dataset} not found.")
        finally:
            self.semaphore.release()
        return self.fo_dataset

    def _get_view(self, idx: IndexDF = None) -> 'fiftyone.DatasetView':
        view = self.fo_session.fiftyone.DatasetView(self._get_or_create_dataset())
        if idx is not None:
            for attrname in self.attrnames:
                if attrname == 'filepath':
                    values = idx[attrname].apply(self.mapping_filepath)
                else:
                    values = idx[attrname]

                view = view.select_by(field=attrname, values=values)
        return view

    def _attend_image_path_to_image_data(self, image_data: ImageData, idxs_values: List[str]) -> ImageData:
        if self.images_table_store is not None:
            image_path_candidates = self.images_table_store._filenames_from_idxs_values(idxs_values)
            if len(image_path_candidates) > 1:
                for image_path in image_path_candidates:
                    openfile = fsspec.open(image_path)
                    if openfile.fs.exists(openfile.path):
                        break
            else:
                image_path = image_path_candidates[0]
            image_data.image_path = image_path
        assert image_data.image_path is not None and str(image_data.image_path) != '', (
            f"This {image_data=} have empty image_path"
        )
        return image_data

    def read_rows(self, idx: IndexDF = None, read_data: bool = True) -> DataDF:
        view = self._get_view(idx)
        df_result = []
        for sample in view:
            image_data = self.fo_session.convert_sample_to_image_data(
                sample=sample,
                fo_detections_label=self.fo_detections_label,
                fo_classification_label=self.fo_classification_label,
                fo_keypoints_label=self.fo_keypoints_label,
                mapping_filepath=self.inverse_mapping_filepath,
                additional_info_keys_in_fo_detections=self.additional_info_keys_in_fo_detections,
                additional_info_keys_in_sample=self.additional_info_keys_in_sample,
                image_data_cls=self.image_data_cls,
                bbox_data_cls=self.bbox_data_cls
            )
            df_result.append({
                'filepath': self.inverse_mapping_filepath(sample.filepath),
                **{'image_data': image_data if read_data else {}},
                **{attrname: sample[attrname] for attrname in self.attrnames_no_filepath},
            })
        df_result = pd.DataFrame(df_result)
        if len(df_result) == 0:
            df_result = pd.DataFrame(columns=self.attrnames)
        return df_result

    def _delete_samples(self, dataset: 'fo.Dataset', samples: List['fo.Sample']):
        # Во избежания ошибки в префекте AttributeError: 'RedirectToLog' object has no attribute 'encoding'
        # dataset.add_samples(samples_to_be_updated)
        self.semaphore.acquire()
        try:
            dataset.delete_samples(samples)
        finally:
            self.semaphore.release()

    def _add_samples(self, dataset: 'fo.Dataset', samples: List['fo.Sample']):
        # Во избежания ошибки в префекте AttributeError: 'RedirectToLog' object has no attribute 'encoding'
        # dataset.add_samples(samples_to_be_updated)
        self.semaphore.acquire()
        try:
            for batch in self.fo_session.fiftyone.core.utils.DynamicBatcher(
                samples, target_latency=0.2, init_batch_size=1, max_batch_beta=2.0
            ):
                dataset._add_samples_batch(batch, expand_schema=True, validate=True)
        finally:
            self.semaphore.release()

    def insert_rows(self, df: DataDF) -> None:
        dataset = self._get_or_create_dataset()

        df_indexes = data_to_index(df, self.attrnames)
        df_current_samples = self.read_rows(df_indexes)
        current_indexes = data_to_index(df_current_samples, self.attrnames)

        idxs_to_be_deleted = index_difference(current_indexes, df_indexes)
        idxs_to_be_added = index_difference(df_indexes, current_indexes)
        idxs_to_be_updated = index_intersection(df_indexes, current_indexes)

        # To be deleted:
        self.delete_rows(idxs_to_be_deleted)

        # To be updated:
        df_to_be_updated = index_to_data(df, idxs_to_be_updated).reset_index(drop=True)
        samples_to_be_updated_unordered = list(self._get_view(idxs_to_be_updated))
        df_samples_to_be_updated_unordered = pd.DataFrame({
            'sample': [sample for sample in samples_to_be_updated_unordered],
            **{
                field: [sample[field] for sample in samples_to_be_updated_unordered]
                for field in self.attrnames
            }
        })
        samples_to_be_updated = index_to_data(df_samples_to_be_updated_unordered, idxs_to_be_updated)['sample']
        samples_to_be_updated_with_new_values = [
            self.fo_session.convert_image_data_to_fo_sample(
                image_data=self._attend_image_path_to_image_data(
                    image_data=df_to_be_updated.loc[idx, 'image_data'],
                    idxs_values=df_to_be_updated.loc[idx, self.attrnames]
                ),
                fo_detections_label=self.fo_detections_label,
                fo_classification_label=self.fo_classification_label,
                fo_keypoints_label=self.fo_keypoints_label,
                mapping_filepath=self.mapping_filepath,
                additional_info_keys_in_bboxes_data=self.additional_info_keys_in_fo_detections,
                additional_info_keys_in_image_data=self.additional_info_keys_in_sample,
                additional_info={field: df_to_be_updated.loc[idx, field] for field in self.attrnames_no_filepath},
            )
            for idx in df_to_be_updated.index
        ]
        for current_sample, sample_with_updated_values in zip(samples_to_be_updated, samples_to_be_updated_with_new_values):
            current_sample.merge(sample_with_updated_values)

        self._delete_samples(dataset, samples_to_be_updated)  # Работает по поведению так же, как и sample.save()
        self._add_samples(dataset, samples_to_be_updated)

        # To be added:
        df_to_be_added = index_to_data(df, idxs_to_be_added).reset_index(drop=True)
        samples_to_be_added = [
            self.fo_session.convert_image_data_to_fo_sample(
                image_data=self._attend_image_path_to_image_data(
                    image_data=df_to_be_added.loc[idx, 'image_data'],
                    idxs_values=df_to_be_added.loc[idx, self.attrnames]
                ),
                fo_detections_label=self.fo_detections_label,
                fo_classification_label=self.fo_classification_label,
                fo_keypoints_label=self.fo_keypoints_label,
                mapping_filepath=self.mapping_filepath,
                additional_info_keys_in_bboxes_data=self.additional_info_keys_in_fo_detections,
                additional_info_keys_in_image_data=self.additional_info_keys_in_sample,
                additional_info={field: df_to_be_added.loc[idx, field] for field in self.attrnames_no_filepath}
            )
            for idx in df_to_be_added.index
        ]
        if len(samples_to_be_added) > 0:
            self._add_samples(dataset, samples_to_be_added)

    def delete_rows(self, idx: IndexDF) -> None:
        dataset = self._get_or_create_dataset()
        view = self._get_view(idx)
        samples = list(view)
        if len(samples) > 0:
            if self.rm_only_fo_fields:
                for sample in samples:
                    if sample.has_field(self.fo_detections_label):
                        del sample[self.fo_detections_label]
                    if self.fo_keypoints_label is not None and sample.has_field(self.fo_keypoints_label):
                        del sample[self.fo_keypoints_label]
            self._delete_samples(dataset, samples)
            if self.rm_only_fo_fields:
                self._add_samples(dataset, samples)
