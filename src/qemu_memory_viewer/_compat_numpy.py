"""Minimal numpy-compatible helpers with consistent ``tolist`` behaviour.

When the real ``numpy`` dependency is available we provide thin wrappers that
delegate to it while normalising a handful of quirks the tests rely on (most
notably the flattened ``tolist`` output).  In environments without ``numpy`` we
fall back to a tiny pure-Python implementation that mimics the required
interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence, Tuple, Union

Number = Union[int, float]

try:  # pragma: no cover - exercised when numpy is installed
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - exercised in the test environment
    
    def _prod(shape: Sequence[int]) -> int:
        result = 1
        for dim in shape:
            result *= int(dim)
        return result


    def _normalize_shape(shape: Union[int, Sequence[int], Tuple[Sequence[int], ...]]) -> Tuple[int, ...]:
        if isinstance(shape, int):
            return (shape,)
        if isinstance(shape, (tuple, list)) and len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            return _normalize_shape(shape[0])
        if isinstance(shape, (tuple, list)):
            normalized = tuple(int(dim) for dim in shape)
            if not normalized:
                raise ValueError("shape must contain at least one dimension")
            return normalized
        raise TypeError(f"Unsupported shape specification: {shape!r}")


    def _result_shape_static(shape: Sequence[int], indices: Tuple[object, ...]) -> Tuple[int, ...]:
        if not indices:
            return tuple(shape)
        if not shape:
            raise IndexError("too many indices for array")
        idx, *rest = indices
        axis = shape[0]
        if isinstance(idx, slice):
            start, stop, step = idx.indices(axis)
            length = len(range(start, stop, step))
            tail = _result_shape_static(shape[1:], tuple(rest))
            return (length,) + tail
        if isinstance(idx, tuple):  # pragma: no cover - defensive branch
            raise TypeError("tuple indices are not supported")
        return _result_shape_static(shape[1:], tuple(rest))


    def _extract_flat_static(flat: Sequence[int], shape: Sequence[int], indices: Tuple[object, ...]) -> list[int]:
        if not indices:
            return list(flat)
        if not shape:
            raise IndexError("too many indices for array")
        idx, *rest = indices
        axis = shape[0]
        block = _prod(shape[1:]) if len(shape) > 1 else 1
        if isinstance(idx, slice):
            start, stop, step = idx.indices(axis)
            collected: list[int] = []
            for pos in range(start, stop, step):
                start_idx = pos * block
                end_idx = start_idx + block
                sub_flat = flat[start_idx:end_idx]
                collected.extend(_extract_flat_static(sub_flat, shape[1:], tuple(rest)))
            return collected
        pos = int(idx)
        if pos < 0:
            pos += axis
        if pos < 0 or pos >= axis:
            raise IndexError("index out of range")
        start_idx = pos * block
        end_idx = start_idx + block
        sub_flat = flat[start_idx:end_idx]
        return _extract_flat_static(sub_flat, shape[1:], tuple(rest))


    class _UInt8DType:
        name = "uint8"
        size = 1

        def __repr__(self) -> str:  # pragma: no cover - debugging helper
            return "uint8"

        def __eq__(self, other: object) -> bool:
            return getattr(other, "name", None) == self.name

        def __hash__(self) -> int:
            return hash(self.name)

        def cast(self, value: Number) -> int:
            return int(value) & 0xFF


    uint8 = _UInt8DType()


    class ndarray:
        """Very small stand-in for :class:`numpy.ndarray`."""

        __slots__ = ("_flat", "shape", "dtype")

        def __init__(self, data: Iterable[Number], shape: Sequence[int], dtype: _UInt8DType = uint8, *, _copy: bool = True) -> None:
            self.shape = tuple(int(dim) for dim in shape)
            expected = _prod(self.shape)
            if isinstance(data, ndarray):
                flat = list(data._flat)
            else:
                flat = list(data)
            if expected != len(flat):
                raise ValueError(f"cannot reshape array of size {len(flat)} into shape {self.shape}")
            if _copy:
                self._flat = [dtype.cast(v) for v in flat]
            else:
                self._flat = flat
            self.dtype = dtype

        def __len__(self) -> int:
            return self.shape[0]

        def __iter__(self) -> Iterator[Union[int, "ndarray"]]:
            if len(self.shape) == 1:
                for value in self._flat:
                    yield self.dtype.cast(value)
                return
            block = _prod(self.shape[1:])
            for i in range(self.shape[0]):
                start = i * block
                end = start + block
                yield ndarray(self._flat[start:end], self.shape[1:], self.dtype)

        def reshape(self, *shape: int) -> "ndarray":
            new_shape = _normalize_shape(shape)
            if _prod(new_shape) != len(self._flat):
                raise ValueError("cannot reshape array with incompatible dimensions")
            return ndarray(self._flat, new_shape, self.dtype, _copy=False)

        def _result_shape(self, indices: Tuple[object, ...]) -> Tuple[int, ...]:
            return _result_shape_static(self.shape, indices)

        def _extract_flat(self, indices: Tuple[object, ...]) -> list[int]:
            return _extract_flat_static(self._flat, self.shape, indices)

        def __getitem__(self, index: object) -> Union[int, "ndarray"]:
            if not isinstance(index, tuple):
                index_tuple = (index,)
            else:
                index_tuple = index
            result_shape = self._result_shape(index_tuple)
            flat = self._extract_flat(index_tuple)
            if not result_shape:
                if not flat:
                    raise IndexError("index resulted in empty selection")
                return self.dtype.cast(flat[0])
            return ndarray(flat, result_shape, self.dtype)

        @property
        def ndim(self) -> int:  # pragma: no cover - currently unused helper
            return len(self.shape)

        def tolist(self) -> list[int]:  # pragma: no cover - helper for debugging
            return list(self._flat)


    def frombuffer(buffer: Iterable[int], dtype: _UInt8DType = uint8) -> ndarray:
        data = list(buffer)
        return ndarray(data, (len(data),), dtype)


    def zeros(shape: Union[int, Sequence[int]], dtype: _UInt8DType = uint8) -> ndarray:
        normalized = _normalize_shape(shape)
        count = _prod(normalized)
        return ndarray([dtype.cast(0)] * count, normalized, dtype, _copy=False)


    def array(data: Iterable[Number], dtype: _UInt8DType = uint8) -> ndarray:
        return asarray(data, dtype)


    def _infer_shape(obj: Union[Sequence[object], object]) -> Tuple[int, ...]:
        if isinstance(obj, (list, tuple)):
            length = len(obj)
            if length == 0:
                return (0,)
            first_shape = _infer_shape(obj[0])
            for item in obj[1:]:
                if _infer_shape(item) != first_shape:
                    raise ValueError("inconsistent nested sequence lengths")
            return (length,) + first_shape
        return ()


    def _flatten(obj: Union[Sequence[object], object]) -> Iterator[Number]:
        if isinstance(obj, (list, tuple)):
            for item in obj:
                yield from _flatten(item)
        else:
            yield obj


    def asarray(obj: object, dtype: _UInt8DType = uint8) -> ndarray:
        if isinstance(obj, ndarray):
            if obj.dtype == dtype:
                return obj
            return ndarray(obj._flat, obj.shape, dtype)
        if isinstance(obj, (bytes, bytearray)):
            return ndarray(list(obj), (len(obj),), dtype)
        if isinstance(obj, (list, tuple)):
            shape = _infer_shape(obj)
            flat = list(_flatten(obj))
            return ndarray(flat, shape, dtype)
        if hasattr(obj, "tobytes") and hasattr(obj, "size"):
            data = obj.tobytes()  # type: ignore[no-any-return]
            width, height = obj.size  # type: ignore[attr-defined]
            mode = getattr(obj, "mode", "L")
            channels = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}.get(mode, 1)
            flat = list(data)
            if channels == 1:
                shape = (height, width)
            else:
                shape = (height, width, channels)
            return ndarray(flat, shape, dtype)
        raise TypeError(f"Unsupported input type for asarray(): {type(obj)!r}")


    def memmap(path: Union[str, Path], dtype: _UInt8DType = uint8, mode: str = "r") -> ndarray:
        if mode not in {"r", "rb"}:
            raise ValueError("compat memmap only supports read-only mode")
        raw_path = Path(path)
        data = raw_path.read_bytes()
        return ndarray(data, (len(data),), dtype)


    __all__ = [
        "array",
        "asarray",
        "frombuffer",
        "memmap",
        "ndarray",
        "uint8",
        "zeros",
    ]

else:  # pragma: no cover - exercised when numpy is installed
    uint8 = _np.uint8

    class _CompatNDArray(_np.ndarray):
        """Subclass that flattens :meth:`tolist` results for compatibility."""

        def __new__(cls, input_array: object, dtype: object | None = None):
            base = _np.asarray(input_array, dtype=dtype)
            if isinstance(base, cls):
                return base
            return base.view(cls)

        def __array_finalize__(self, obj) -> None:  # pragma: no cover - numpy protocol hook
            return None

        def tolist(self) -> list[int]:  # pragma: no cover - behaviour verified via tests
            base = self.view(_np.ndarray)
            flat = base.reshape(-1).tolist()
            return [int(value) for value in flat]


    ndarray = _CompatNDArray


    def _wrap(arr: _np.ndarray, *, dtype: object | None = None) -> _CompatNDArray:
        if not isinstance(arr, _CompatNDArray):
            arr = arr.view(_CompatNDArray)
        if dtype is not None and arr.dtype != dtype:
            return _CompatNDArray(arr, dtype=dtype)
        return arr


    def asarray(obj: object, dtype: object = uint8) -> _CompatNDArray:
        arr = _np.asarray(obj, dtype=dtype)
        return _wrap(arr, dtype=dtype)


    def array(obj: object, dtype: object = uint8) -> _CompatNDArray:
        return asarray(obj, dtype)


    def zeros(shape: Union[int, Sequence[int]], dtype: object = uint8) -> _CompatNDArray:
        arr = _np.zeros(shape, dtype=dtype)
        return _wrap(arr, dtype=dtype)


    def frombuffer(buffer: Iterable[int], dtype: object = uint8) -> _CompatNDArray:
        arr = _np.frombuffer(buffer, dtype=dtype)
        return _wrap(arr, dtype=dtype)


    def memmap(path: Union[str, Path], dtype: object = uint8, mode: str = "r") -> _CompatNDArray:
        arr = _np.memmap(path, dtype=dtype, mode=mode)
        return _wrap(arr, dtype=dtype)


    __all__ = [
        "array",
        "asarray",
        "frombuffer",
        "memmap",
        "ndarray",
        "uint8",
        "zeros",
    ]

    def __getattr__(name: str):  # pragma: no cover - simple delegation helper
        return getattr(_np, name)
