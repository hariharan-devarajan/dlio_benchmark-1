from time import time
import logging
import math

import tensorflow as tf

from src.common.constants import MODULE_DATA_LOADER
from src.common.enumerations import DataLoaderType, Shuffle, FormatType, DatasetType
from src.data_loader.base_data_loader import BaseDataLoader
from src.reader.reader_factory import ReaderFactory
from src.utils.utility import utcnow, Profile

import numpy as np

dlp = Profile(MODULE_DATA_LOADER)


class TensorflowDataset(tf.data.Dataset):
    @staticmethod
    @dlp.log
    def _generator(format_type, dataset_type, epoch_number, thread_index):
        format_type = format_type.decode('ascii')
        dataset_type = dataset_type.decode('ascii')
        logging.debug(f"{utcnow()} format_type {format_type} dataset_type {dataset_type} tensors")
        reader = ReaderFactory.get_reader(type=FormatType.get_enum(format_type),
                                          dataset_type=DatasetType.get_enum(dataset_type),
                                          thread_index=thread_index,
                                          epoch_number=epoch_number)
        for batch in reader.next():
            yield batch

    @dlp.log
    def __new__(cls, format_type, dataset_type, epoch, shape, thread_index):
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.uint8,
            output_shapes=shape,
            args=(format_type.value, dataset_type.value, epoch, thread_index,),
        )
        return dataset


class TFDataLoader(BaseDataLoader):

    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch):
        super().__init__(format_type, dataset_type, epoch)
        self._dataset = None

    @dlp.log
    def read(self):
        read_threads = self._args.read_threads
        if read_threads == 0:
            if self._args.my_rank == 0:
                logging.warning(
                    f"{utcnow()} `read_threads` is set to be 0 for tf.data loader. We change it to 1")
            read_threads = 1

        options = tf.data.Options()
        if "threading" in dir(options):
            options.threading.private_threadpool_size = read_threads
            options.threading.max_intra_op_parallelism = read_threads
        elif "experimental_threading" in dir(options):
            options.experimental_threading.private_threadpool_size = read_threads
            options.experimental_threading.max_intra_op_parallelism = read_threads

        batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval
        self._dataset = tf.data.Dataset.from_tensor_slices(np.arange(read_threads)).with_options(options)

        self._dataset = self._dataset.interleave(lambda x: TensorflowDataset(self.format_type, self.dataset_type,
                                                                             self.epoch_number, (
                                                                                 batch_size,
                                                                                 self._args.max_dimension,
                                                                                 self._args.max_dimension), x),
                                                 cycle_length=read_threads,
                                                 num_parallel_calls=read_threads)
        if self._args.prefetch_size > 0:
            self._dataset = self._dataset.prefetch(buffer_size=self._args.prefetch_size)

    @dlp.log
    def next(self):
        for batch in self._dataset:
            yield batch

    @dlp.log
    def finalize(self):
        pass
