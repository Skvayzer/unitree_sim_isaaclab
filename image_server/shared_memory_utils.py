# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
A simplified multi-image shared memory tool module
When writing, concatenate three images (head, left, right) horizontally and write them
When reading, split the concatenated image into three independent images
"""

import ctypes
import time
import numpy as np
import cv2
from multiprocessing import shared_memory
from typing import Optional, Dict, List
import struct

# shared memory configuration
SHM_NAME = "isaac_multi_image_shm"
# allocate enough space for stereo head (2x width) plus two wrist cameras
SHM_SIZE = 640 * 480 * 3 * 4 + 1024  # max 4 frames' worth of data + header buffer

# define the simplified header structure
class SimpleImageHeader(ctypes.Structure):
    """Simplified image header structure"""
    _fields_ = [
        ('timestamp', ctypes.c_uint64),    # timestamp
        ('height', ctypes.c_uint32),       # image height
        ('width', ctypes.c_uint32),        # total width after concatenation
        ('channels', ctypes.c_uint32),     # number of channels
        ('single_width', ctypes.c_uint32), # single image width
        ('image_count', ctypes.c_uint32),  # number of images
        ('data_size', ctypes.c_uint32),    # data size
    ]


class MultiImageWriter:
    """A simplified multi-image shared memory writer"""
    
    def __init__(self, shm_name: str = SHM_NAME, shm_size: int = SHM_SIZE):
        """Initialize the multi-image shared memory writer
        
        Args:
            shm_name: the name of the shared memory
            shm_size: the size of the shared memory
        """
        self.shm_name = shm_name
        self.shm_size = shm_size
        
        try:
            # try to open the existing shared memory
            self.shm = shared_memory.SharedMemory(name=shm_name)
        except FileNotFoundError:
            # if not exist, create a new shared memory
            self.shm = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)
        
        print(f"[MultiImageWriter] Shared memory initialized: {shm_name}")

    def write_images(self, images: Dict[str, np.ndarray]) -> bool:
        """Write multiple images to the shared memory (concatenate and write)
        
        Args:
            images: the image dictionary, the key is the image name ('head', 'left', 'right'), the value is the image array
            
        Returns:
            bool: whether the writing is successful
        """
        if not images or self.shm is None:
            return False
            
        try:
            # get the images in order: head, left, right
            frames_to_concat = []
            image_order = ['head', 'left', 'right']
            
            for image_name in image_order:
                if image_name in images:
                    image = images[image_name]
                    if not image.flags['C_CONTIGUOUS']:
                        image = np.ascontiguousarray(image)
                    
                    frames_to_concat.append(image)
            
            if not frames_to_concat:
                return False
            
            # concatenate the images horizontally
            if len(frames_to_concat) > 1:
                concatenated_image = cv2.hconcat(frames_to_concat)
            else:
                concatenated_image = frames_to_concat[0]
            
            # get the image information
            height, total_width, channels = concatenated_image.shape
            single_width = total_width // len(frames_to_concat)
            data_size = height * total_width * channels
            
            # prepare the header information
            header = SimpleImageHeader()
            header.timestamp = int(time.time() * 1000)  # millisecond timestamp
            header.height = height
            header.width = total_width
            header.channels = channels
            header.single_width = single_width
            header.image_count = len(frames_to_concat)
            header.data_size = data_size
            
            # write the header
            header_size = ctypes.sizeof(SimpleImageHeader)
            header_bytes = ctypes.string_at(ctypes.byref(header), header_size)
            header_view = memoryview(self.shm.buf)
            header_view[:header_size] = header_bytes
            
            # write the image data
            image_bytes = concatenated_image.tobytes()
            data_view = memoryview(self.shm.buf)
            data_view[header_size:header_size + len(image_bytes)] = image_bytes
            return True
            
        except Exception as e:
            print(f"shared_memory_utils [MultiImageWriter] Error writing to shared memory: {e}")
            print(f"Images: {list(images.keys())}")
            return False

    def close(self):
        """Close the shared memory"""
        if hasattr(self, 'shm') and self.shm is not None:
            self.shm.close()
            print(f"[MultiImageWriter] Shared memory closed: {self.shm_name}")


class MultiImageReader:
    """A simplified multi-image shared memory reader"""
    
    def __init__(self, shm_name: str = SHM_NAME):
        """Initialize the multi-image shared memory reader
        
        Args:
            shm_name: the name of the shared memory
        """
        self.shm_name = shm_name
        self.last_timestamp = 0
        self.buffer = {}
        
        try:
            # open the shared memory
            self.shm = shared_memory.SharedMemory(name=shm_name)
            print(f"[MultiImageReader] Shared memory opened: {shm_name}")
        except FileNotFoundError:
            print(f"[MultiImageReader] Shared memory {shm_name} not found")
            self.shm = None

    def read_images(self) -> Optional[Dict[str, np.ndarray]]:
        """Read multiple images from the shared memory (read the concatenated images and split them)
        
        Returns:
            Dict[str, np.ndarray]: the image dictionary, the key is the image name, the value is the image array; if the reading fails, return None
        """
        if self.shm is None:
            return None
            
        try:
            # read the header data
            header_size = ctypes.sizeof(SimpleImageHeader)
            header_data = bytes(self.shm.buf[:header_size])
            header = SimpleImageHeader.from_buffer_copy(header_data)
            
            # check if there is new data
            if header.timestamp <= self.last_timestamp:
                return self.buffer
                
            # read the concatenated image data
            start_offset = header_size
            end_offset = start_offset + header.data_size
            image_data = bytes(self.shm.buf[start_offset:end_offset])
            
            # convert to numpy array
            concatenated_image = np.frombuffer(image_data, dtype=np.uint8)
            
            # ensure the data size is correct
            expected_size = header.height * header.width * header.channels
            if concatenated_image.size != expected_size:
                print(f"[MultiImageReader] Data size mismatch: expected {expected_size}, got {concatenated_image.size}")
                return None
                
            # reshape the array
            concatenated_image = concatenated_image.reshape(header.height, header.width, header.channels)
            
            # split the images
            images = {}
            image_names = ['head', 'left', 'right']
            single_width = header.single_width
            
            for i in range(header.image_count):
                if i < len(image_names):
                    start_col = i * single_width
                    end_col = start_col + single_width
                    
                    # split the single image
                    single_image = concatenated_image[:, start_col:end_col, :]
                    images[image_names[i]] = single_image
            
            # update the buffer and timestamp
            self.buffer = images
            self.last_timestamp = header.timestamp
            return images
            
        except Exception as e:
            print(f"[MultiImageReader] Error reading from shared memory: {e}")
            return None

    def read_concatenated_image(self) -> Optional[np.ndarray]:
        """Read the concatenated image (without splitting)
        
        Returns:
            np.ndarray: the concatenated image array; if the reading fails, return None
        """
        if self.shm is None:
            return None
            
        try:
            # read the header data
            header_size = ctypes.sizeof(SimpleImageHeader)
            header_data = bytes(self.shm.buf[:header_size])
            header = SimpleImageHeader.from_buffer_copy(header_data)
            
            # check if there is new data
            if header.timestamp <= self.last_timestamp:
                return None
            
            # read the concatenated image data
            start_offset = header_size
            end_offset = start_offset + header.data_size
            image_data = bytes(self.shm.buf[start_offset:end_offset])
            
            # convert to numpy array
            concatenated_image = np.frombuffer(image_data, dtype=np.uint8)
            
            # ensure the data size is correct
            expected_size = header.height * header.width * header.channels
            if concatenated_image.size != expected_size:
                print(f"[MultiImageReader] Data size mismatch: expected {expected_size}, got {concatenated_image.size}")
                return None
                
            # reshape the array
            concatenated_image = concatenated_image.reshape(header.height, header.width, header.channels)
            
            # update the timestamp
            self.last_timestamp = header.timestamp
            # print(f"concatenated_image: {concatenated_image.shape}")
            return concatenated_image
            
        except Exception as e:
            print(f"[MultiImageReader] Error reading concatenated image from shared memory: {e}")
            return None

    def close(self):
        """Close the shared memory"""
        if self.shm is not None:
            self.shm.close()
            print(f"[MultiImageReader] Shared memory closed: {self.shm_name}")


# backward compatible class (single image)
class SharedMemoryWriter:
    """Backward compatible single image writer"""
    
    def __init__(self, shm_name: str = SHM_NAME, shm_size: int = SHM_SIZE):
        self.multi_writer = MultiImageWriter(shm_name, shm_size)
    
    def write_image(self, image: np.ndarray) -> bool:
        """Write a single image (as the head image)"""
        return self.multi_writer.write_images({'head': image})
    
    def close(self):
        self.multi_writer.close()


class SharedMemoryReader:
    """Backward compatible single image reader"""
    
    def __init__(self, shm_name: str = SHM_NAME):
        self.multi_reader = MultiImageReader(shm_name)
    
    def read_image(self) -> Optional[np.ndarray]:
        """Read a single image (the head image)"""
        images = self.multi_reader.read_images()
        return images.get('head') if images else None
    
    def close(self):
        self.multi_reader.close() 
