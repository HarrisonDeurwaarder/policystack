import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from typing import List, Any, Literal


class Rollout(Dataset):
    """
    flexible transition storage for PPO
    """
    def __init__(
        self, 
        stackable: List[str] = ["obs", "actions", "log_probs", "rewards", "values", "dones", "advantages"], 
        sequential: List[str] = list(), 
        opaque: List[str] = list()
    ) -> None:
        """
        Args:
            stackable (List[str]): ids of items which may be concatenated for batching (i.e. invariably-dimensioned tensors)
            sequential (List[str]): ids of items which must be packed for batching (e.g. time-series tensors with variable dimensions)
            opaque (List[str]): ids of items which must be returned as-is, in the list format they are stored for batching (i.e. non-tensors)
        """
        super().__init__()
        # repeatable function call
        self.process_keys = lambda keys: {key: list() for key in keys}
        # store ids for reference in 
        self.stackable_ids, self.sequential_ids, self.opaque_ids = stackable, sequential, opaque
        
        
    def __len__(self) -> int:
        # find a valid list to defer to
        if self.stackable_ids:    return len(self.stackables[self.stackable_ids[0]]) - int(self.stackable_ids[0] in self.extra_ids)
        elif self.sequential_ids: return len(self.sequentials[self.sequential_ids[0]]) - int(self.sequential_ids[0] in self.extra_ids)
        else:                     return len(self.opaques[self.opaque_ids[0]]) - int(self.opaque_ids[0] in self.extra_ids)
    
    
    def __getitem__(self, idx: int | str) -> dict[str, Any]:
        # access by transition for batching (by integer indicies)
        if isinstance(idx, int):
            # merge all items
            out = {key: self.stackables[key][idx] for key in self.stackable_ids}
            out |= {key: self.sequentials[key][idx] for key in self.sequential_ids}
            out |= {key: self.opaques[key][idx] for key in self.opaque_ids}
            
        # access by item type with string keys
        else:
            # find correct container
            if idx in self.stackable_ids:    out = self.stackables[idx]
            elif idx in self.sequential_ids: out = self.stackables[idx]
            else:                            out = self.stackables[idx]
            
        return out
        
    
    def reset(self, items: dict[str, Any] = dict()) -> None:
        """
        clear all containers
        """
        def _process_keys(keys: List[str]) -> dict[str, List]: {key: list() for key in keys}
        # remake all containers
        self.stackables, self.sequentials, self.opaques = (
            _process_keys(self.stackable_ids), _process_keys(self.sequential_ids), _process_keys(self.opaque_ids)
        )
        # add pre-rollout items
        self.add(items)
        # mark that the given ids correspond to items that are added +1 more than the length
        self.extra_ids = list(items.keys())
        
        
    def add(self, items: dict[str, Any]) -> None:
        for key, element in items.items():
            # attempt to add to each collections
            if key in self.stackable_ids:   self.stackables[key].append(element)
            elif key in self.stackable_ids: self.stackables[key].append(element)
            else:                           self.stackables[key].append(element)
            
            
    def annotate(self, id: str, column: list, container: Literal["stackable", "sequential", "opaque"] = "stackable") -> None:
        # error if lengths are mismatched
        if self.__len__() != len(column):
            raise ValueError(f"expected column of length {self.__len__()}, got {len(column)}")
        # error if an invalid container is passed
        if not container in ["stackable", "sequential", "opaque"]:
            raise ValueError(f"expected container to be one of 'stackable', 'sequential', or 'opaque', got {container}")

        # match column to correct container
        if container == "stackable":
            self.stackable_ids.append(id)
            self.stackables[id] = column
        elif container == "sequential":
            self.sequential_ids.append(id)
            self.sequentials[id] = column
        else:
            self.opaque_ids.append(id)
            self.opaques[id] = column
            
       
    def collate(self, batch: List[dict[str, Any]]) -> dict[str, Any]:
        # reformate from a list of dicts to a dict of lists
        transformed_batch = {key: [sample[key] for sample in batch] for key in batch[0].keys()}
        out = dict()
        # handle vectorization
        for key, items in transformed_batch.items():
            
            # stack all stackables
            if key in self.stackable_ids:
                out[key] = torch.stack(items)
                
            # pad all varianble-dimensioned elements
            elif key in self.sequential_ids:
                lengths = [s.size(0) for s in items] # (T_i, ...)
                # pads all variable-length sequences to the same dimensionality
                padded = pad_sequence(items, batch_first=True) # (B, T_max, ...)
                # ensure no computation is done on the padding
                packed = pack_padded_sequence(padded, lengths, batch_first=True)
                out[key] = packed
                
            # opaque items are not handled internally
            else:
                out[key] = items
                
        return out