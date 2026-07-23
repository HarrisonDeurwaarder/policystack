import torch
import torch.nn as nn
from torch.utils.data import Dataset

from abc import ABC, abstractmethod


class TransitionBuffer(ABC):
    """Abstract transition storage with flexible contents"""
    
    def __init__(
        self, 
        batch_dims: tuple[int, ...], 
        fields: dict[str, tuple[int, ...]], 
        length: int
    ) -> None:
        # batch dims are necessary to create empty transitions of arbitrary dimensionality (e.g. B, E)
        self.batch_dims = batch_dims
        self.field_names = fields
        self.fields = dict()
        self.length = length
        
        
    @abstractmethod
    def __len__() -> int:
        ...
        
    
    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ...
        
    
    def __getattr__(self, name: str) -> torch.Tensor:
        return self.fields[name]
        
    
    def reset(self) -> None:
        """Clear the buffer and all associated attributes"""
        for field, shape in self.fields_names.items():
            # fields will generally be of shape (B, E, field_dims..., length)
            self.fields[field] = torch.zeros(self.batch_dims + shape + (self.length,))
        self.index = 0
        # transition fields for one step may be staged at different points before being commited to the buffer
        self.staged = dict()
        
        
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
        for field_name in self.field_names.keys():
            self.fields[field_name][..., self.index] = fields[field_name] # copy the reference from the passed transition
        # the next available index should be used
        self.index += 1
            
    
    def annotate(self, field_name: str, field: torch.Tensor) -> None:
        """Annotate a new column to the buffer"""
        # verify that the correct batch dimensions and length exist
        shape = field.shape()
        if shape[:len(self.batch_dims)] != self.batch_dims or shape[-1] != self.__len__():
            raise ValueError(f"Expected field of leading shape {self.batch_dims} and trailing dimension {self.__len__()}, got {shape}")
        # verify that field does not already exist
        if field_name in self.fields.keys():
            raise ValueError(f"Field {field_name} is already in buffer ({list(self.fields.keys())})")
        
        self.fields[field_name] = field
        
        
        
class Replay(TransitionBuffer):
    """Replay buffer for off-policy transition storage; commonly stores 1mil+ transitions, overflow removes the oldest samples"""
    def __len__(self,):
        return self.length if self.has_overflown else self.index
    
    
    def __getitem__(self, idx):
        # DataLoader only batches using positive indices
        # negative indicies should be more precise
        if idx < 0:
            idx = (idx + 1) % self.length
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
        if idx < 0:
            idx = self.index + (idx + 1) # intuitively handle negative indices
        return super().__getitem__(idx)
    
    
    def full(self) -> bool:
        return self.index >= self.length
    
    
    def add(self, fields: dict[str, torch.Tensor]) -> None:
        # verify that capacity has not been reached
        if self.full():
            raise BufferError("Rollout is at capacity, failed to add excess transition")
        super().add(fields)