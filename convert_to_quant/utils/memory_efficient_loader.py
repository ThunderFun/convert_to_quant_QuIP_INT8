"""
Unified safetensors loader with optional memory-efficient mode.

Provides a consistent interface for tensor loading regardless of mode.
"""
import gc
import mmap
import json
import struct
import os
import torch
from safetensors import safe_open
from typing import Dict, Optional, Union


class MemoryMappedTensorLoader:
    """
    Memory-mapped tensor loader for zero-copy access to safetensors files.
    
    Uses mmap for efficient loading of large files without loading entire
    content into RAM.
    """
    
    def __init__(self, filename: str):
        self.filename = filename
        self._header = None
        self._header_size = None
        self._mmap = None
        self._metadata = {}
        self._tensor_offsets = {}
        
    def __enter__(self):
        # Open file and create memory map
        self._file = open(self.filename, "rb")
        self._file_size = os.fstat(self._file.fileno()).st_size
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Read header
        self._header_size = struct.unpack("<Q", self._mmap[:8])[0]
        header_json = self._mmap[8:8 + self._header_size].decode("utf-8")
        self._header = json.loads(header_json)
        
        # Extract metadata
        self._metadata = self._header.get("__metadata__", {})
        
        # Compute tensor offsets (accounting for header)
        base_offset = 8 + self._header_size
        for key, info in self._header.items():
            if key == "__metadata__":
                continue
            start, end = info["data_offsets"]
            self._tensor_offsets[key] = (base_offset + start, base_offset + end)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def close(self):
        """Close memory map and file handles."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if hasattr(self, '_file'):
            self._file.close()
    
    def keys(self):
        """Return list of all tensor keys."""
        return list(self._tensor_offsets.keys())
    
    def metadata(self) -> Dict[str, str]:
        """Return file metadata."""
        return self._metadata
    
    def get_shape(self, key: str) -> tuple:
        """Get tensor shape without loading data."""
        if key not in self._header:
            raise KeyError(f"Tensor '{key}' not found")
        return tuple(self._header[key]["shape"])
    
    def get_ndim(self, key: str) -> int:
        """Get tensor dimensionality."""
        return len(self.get_shape(key))
    
    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Map safetensors dtype to torch dtype."""
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str, torch.float32)
    
    def get_tensor(self, key: str) -> torch.Tensor:
        """
        Load tensor via memory map (creates a copy).
        
        Note: This creates a copy to ensure tensor is writable.
        For true zero-copy (read-only), use get_tensor_view.
        """
        if key not in self._tensor_offsets:
            raise KeyError(f"Tensor '{key}' not found")
        
        start, end = self._tensor_offsets[key]
        info = self._header[key]
        dtype = self._get_torch_dtype(info["dtype"])
        shape = info["shape"]
        
        # Read bytes from mmap and create tensor
        data = self._mmap[start:end]
        byte_tensor = torch.frombuffer(data, dtype=torch.uint8).clone()
        
        # Handle float8 types
        if info["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, info["dtype"], shape)
        
        return byte_tensor.view(dtype).reshape(shape)
    
    def get_tensor_view(self, key: str) -> torch.Tensor:
        """
        Get a read-only view of the tensor.
        
        Returns a copy to prevent use-after-free crashes when the loader
        is closed while the tensor is still in use.
        
        Args:
            key: Tensor key to retrieve
            
        Returns:
            A copy of the tensor
        """
        return self.get_tensor(key)
    
    def _convert_float8(self, byte_tensor: torch.Tensor, dtype_str: str, shape: list) -> torch.Tensor:
        """Convert bytes to float8 tensor."""
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str}")

class UnifiedSafetensorsLoader:
    """Unified safetensors loader supporting preload, streaming, and memory-mapped modes.

    Modes:
    - standard (low_memory=False, use_mmap=False):
        - Loads all tensors upfront (fast, uses more RAM)
        - Tensors remain in memory until explicitly deleted
        
    - streaming (low_memory=True, use_mmap=False):
        - Loads tensors on-demand via get_tensor()
        - Caller should delete tensors after processing
        
    - memory-mapped (low_memory=False, use_mmap=True):
        - Uses OS memory mapping for zero-copy access
        - Efficient for very large files
        - Tensors are read-only views until copied

    Usage:
        # Standard mode
        with UnifiedSafetensorsLoader("model.safetensors") as loader:
            tensor = loader.get_tensor("key")
            
        # Low-memory streaming mode
        with UnifiedSafetensorsLoader("model.safetensors", low_memory=True) as loader:
            for key in loader.keys():
                tensor = loader.get_tensor(key)
                loader.mark_processed(key)
                
        # Memory-mapped mode (read-only, zero-copy)
        with UnifiedSafetensorsLoader("model.safetensors", use_mmap=True) as loader:
            tensor = loader.get_tensor(key)  # Returns a copy
            tensor_view = loader.get_tensor_view(key)  # Returns zero-copy view
    """

    def __init__(self, filename: str, low_memory: bool = False, use_mmap: bool = False):
        """Initialize the loader.

        Args:
            filename: Path to safetensors file
            low_memory: If True, use streaming mode; if False, preload or mmap
            use_mmap: If True, use memory-mapped file access (overrides low_memory)
        """
        self.filename = filename
        self.low_memory = low_memory
        self.use_mmap = use_mmap
        self._tensors: Dict[str, torch.Tensor] = {}
        self._all_keys = []
        self._file = None
        self._header = None
        self._header_size = None
        self._metadata: Dict[str, str] = {}
        self._mmap_loader: Optional[MemoryMappedTensorLoader] = None

        if use_mmap:
            # Memory-mapped mode: use specialized loader
            self._mmap_loader = MemoryMappedTensorLoader(filename)
            self._mmap_loader.__enter__()
            self._all_keys = self._mmap_loader.keys()
            self._metadata = self._mmap_loader.metadata()
            print(f"Memory-mapped mode: found {len(self._all_keys)} tensors")
        elif low_memory:
            # Streaming mode: read header only, keep file open
            self._header, self._header_size = self._read_header()
            self._file = open(filename, "rb")
            self._all_keys = [k for k in self._header.keys() if k != "__metadata__"]
            # Extract metadata from header (safetensors stores it under __metadata__ key)
            self._metadata = self._header.get("__metadata__", {})
            print(f"Low-memory mode: found {len(self._all_keys)} tensors (streaming)")
        else:
            # Standard mode: preload all tensors
            with safe_open(filename, framework="pt", device="cpu") as f:
                self._metadata = f.metadata() or {}
                self._all_keys = list(f.keys())
                print(f"Loading {len(self._all_keys)} tensors from source file...")
                from tqdm import tqdm
                for key in tqdm(self._all_keys, desc="Loading tensors"):
                    self._tensors[key] = f.get_tensor(key)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close file handle and release resources."""
        if self._mmap_loader:
            self._mmap_loader.close()
            self._mmap_loader = None
        if self._file:
            self._file.close()
            self._file = None
        self._tensors.clear()

    def keys(self):
        """Return list of all tensor keys."""
        return self._all_keys

    def metadata(self) -> Dict[str, str]:
        """Return file metadata."""
        return self._metadata

    def get_shape(self, key: str) -> tuple:
        """Get tensor shape without loading tensor data.

        In low-memory mode, reads from header.
        In memory-mapped mode, reads from header.
        In standard mode, returns shape from loaded tensor.
        """
        if self._mmap_loader:
            return self._mmap_loader.get_shape(key)
        elif self.low_memory:
            if key not in self._header:
                raise KeyError(f"Tensor '{key}' not found in file")
            return tuple(self._header[key]["shape"])
        else:
            return tuple(self._tensors[key].shape)

    def get_ndim(self, key: str) -> int:
        """Get tensor ndim without loading tensor data."""
        return len(self.get_shape(key))

    def get_tensor(self, key: str) -> torch.Tensor:
        """Get a tensor by key.

        In standard mode, returns from cache.
        In memory-mapped mode, returns a copy from mmap.
        In low-memory mode, loads from file on-demand.
        """
        if self._mmap_loader:
            # Memory-mapped mode: return copy of tensor
            return self._mmap_loader.get_tensor(key)
        
        if not self.low_memory:
            # Standard mode: return from preloaded cache
            return self._tensors[key]

        # Low-memory mode: load on-demand
        if key not in self._header:
            raise KeyError(f"Tensor '{key}' not found in file")

        metadata = self._header[key]
        offset_start, offset_end = metadata["data_offsets"]

        if offset_start != offset_end:
            self._file.seek(self._header_size + 8 + offset_start)
            # Use bytearray to create a writable buffer, avoiding PyTorch warning
            # about non-writable tensors from read-only bytes.
            tensor_bytes = bytearray(offset_end - offset_start)
            self._file.readinto(tensor_bytes)
        else:
            tensor_bytes = None

        return self._deserialize_tensor(tensor_bytes, metadata)
    
    def get_tensor_view(self, key: str) -> torch.Tensor:
        """
        Get a read-only view of the tensor (zero-copy).
        
        Only available in memory-mapped mode.
        
        Warning: The returned tensor shares memory with the file.
        Do not modify the tensor or use after the loader is closed.
        """
        if not self._mmap_loader:
            raise RuntimeError("get_tensor_view() only available in memory-mapped mode. "
                             "Use use_mmap=True when creating the loader.")
        return self._mmap_loader.get_tensor_view(key)

    def mark_processed(self, key: str):
        """Mark a tensor as processed, freeing memory if in low-memory mode.

        In standard mode, optionally deletes from cache.
        In low-memory mode, this is a no-op (tensor was never cached).
        """
        if not self.low_memory and key in self._tensors:
            del self._tensors[key]
            gc.collect()

    def _read_header(self):
        """Read and parse the safetensors header."""
        with open(self.filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata) -> torch.Tensor:
        """Deserialize raw bytes into a torch tensor."""
        dtype_str = metadata["dtype"]
        shape = metadata["shape"]
        dtype = self._get_torch_dtype(dtype_str)

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            byte_tensor = torch.frombuffer(tensor_bytes, dtype=torch.uint8)

        if dtype_str in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, dtype_str, shape)

        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str: str) -> torch.dtype:
        """Map safetensors dtype string to torch dtype."""
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn

        dtype = dtype_map.get(dtype_str)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return dtype

    @staticmethod
    def _convert_float8(byte_tensor: torch.Tensor, dtype_str: str, shape: list) -> torch.Tensor:
        """Convert bytes to float8 tensor."""
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str}")


# Backward compatibility alias
MemoryEfficientSafeOpen = UnifiedSafetensorsLoader
