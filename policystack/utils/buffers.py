import torch
import torch.nn as nn
from torch.utils.data import Dataset, default_collate

from abc import ABC, abstractmethod
from typing import Callable


class TransitionBuffer(ABC):
    """Abstract transition storage with flexible contents"""
    
    collate: Callable = default_collate
    
    def __init__(
        self, 
        fields: list[str], 
        length: int
    ) -> None:
        # enforce persistent field names across resets
        # dims are inferred upon the first call of add()
        self.field_names = fields[:]
        self.original_field_names = fields[:]
        self.fields = dict()
        self.length = length
        self.reset()
        
        
    @abstractmethod
    def __len__() -> int:
        ...
        
    
    def __getitem__(self, idx: int | str) -> dict[str, torch.Tensor]:
        # enforce integer or string keys
        if isinstance(idx, int):
            return {field: self.fields[field][..., idx] for field in self.field_names}
        elif isinstance(idx, str):
            # enforce key existence
            if not idx in self.field_names:
                raise KeyError(f"'{idx}'")
            return self.fields[idx]
        elif isinstance(idx, slice):
            raise NotImplementedError()
        else:
            raise ValueError(f"Expected idx to be of datatype 'int' or 'str, got '{type(idx)}'")
        
    
    def __getattr__(self, name: str) -> torch.Tensor:
        if name in self.field_names:
            return self.__getitem__(idx=name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
    
    def reset(self) -> None:
        self.index = 0
        # remove any annotated field names from the field_name list
        self.field_names = self.original_field_names[:]
        # transition fields for one step may be staged at different points before being commited to the buffer
        self.staged = dict()
        self.populated = False
        
        
    def populate(self, field_dims: dict[str, torch.Size]) -> None:
        # empty fields to expunge all residual annotated field names
        self.fields = dict()
        for field, size in field_dims.items():
            # fields will generally be of shape (E, field_dims..., length)
            self.fields[field] = torch.zeros(size + torch.Size((self.length,)))
        
        
    def stage(self, fields: dict[int, torch.Tensor]) -> None:
        """Enables fields to be staged at different times before being commited upon .commit()"""
        self.staged.update(fields)
        
        
    def from_staged(self, field_name: str) -> torch.Tensor:
        """Access a field from the temporary staged storage"""
        return self.staged[field_name]
        
    
    def commit(self) -> None:
        """Pushes staged fields to the buffer; all fields must have been previously staged in order to commit"""
        self.add(self.staged)
        # empty staged fields
        self.staged = dict()
    
    
    def add(self, fields: dict[str, torch.Tensor]) -> None:
        """Adds a new transition to the buffer"""
        # resolve field shapes if unpopulated
        if not self.populated:
            self.populate({field: fields[field].shape for field in fields})
            self.populated = True
            
        for field in self.field_names:
            self.fields[field][..., self.index] = fields[field] # copy the reference from the passed transition
        # the next available index should be used
        self.index += 1
            
    
    def annotate(self, field_name: str, field: torch.Tensor, persistent: bool = False) -> None:
        """Annotate a new column to the buffer"""
        # verify that the correct batch dimensions and length exist
        if field.shape[-1] != self.__len__():
            raise ValueError(f"Expected field of trailing dimension {self.__len__()}, got {field.shape}")
        # verify that field does not already exist
        if field_name in self.field_names:
            raise ValueError(f"Field {field_name} is already in buffer ({list(self.field_names)})")
        
        self.fields[field_name] = field
        self.field_names.append(field_name)
        
        # persistence defines whether or not the field will be expected after future resets
        if persistent: self.original_field_names.append(self.field_names)
        
        
class Replay(TransitionBuffer):
    """Replay buffer for off-policy transition storage; commonly stores 1mil+ transitions, overflow removes the oldest samples"""
    def __len__(self,):
        return self.length if self.has_overflown else self.index
    
    
    def __getitem__(self, idx):
        # DataLoader only batches using positive indices
        # negative indicies should be more precise
        if isinstance(idx, int) and idx < 0:
            idx = (self.index + idx + self.length) % self.length
        return super().__getitem__(idx)
    
    
    def reset(self) -> None:
        super().reset()
        self.has_overflown = False
        
        
    def add(self, fields: dict[str, torch.Tensor]) -> None:
        super().add(fields)
        # if increment overflowed, begin overriding buffer
        if self.index >= self.length:
            self.index = 0
            self.has_overflown = True
        


class Rollout(TransitionBuffer):
    """Rollout buffer for on-policy transition storage; commonly stores ~4096 transitions, lack of overflow is enforced"""
    def __len__(self) -> int:
        return self.index
    
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if isinstance(idx, int) and idx < 0:
            idx = self.index + (idx + 1) # intuitively handle negative indices
        return super().__getitem__(idx)
    
    
    def full(self) -> bool:
        return self.index >= self.length
    
    
    def add(self, fields: dict[str, torch.Tensor]) -> None:
        # verify that capacity has not been reached
        if self.full():
            raise BufferError("Rollout is at capacity, failed to add excess transition")
        super().add(fields)