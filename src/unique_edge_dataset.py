"""Distinct-edge training augmentation for the separately installed upstream dataset.

Requires `models.utils.ContagionDataset` from the ORIGINAL research repository
on sys.path. The upstream code is intentionally not vendored here.
"""
import copy
import torch
from models.utils import ContagionDataset as OriginalContagionDataset


class UniqueEdgeContagionDataset(OriginalContagionDataset):
    """Remove exactly floor(rate * eligible_nonself_edges) unique edges each access."""

    def __getitem__(self, i):
        graph = self.graphs[i]
        if self.drop_edges > 0:
            graph = copy.deepcopy(graph)
            src, dst = graph.edges(order='eid')
            eligible = torch.nonzero(src != dst, as_tuple=False).flatten()
            count = int(len(eligible) * self.drop_edges)
            if count:
                selection = torch.randperm(
                    len(eligible), generator=self.random_generator
                )[:count]
                graph.remove_edges(eligible[selection])
        if self.add_self_loop:
            graph = graph.remove_self_loop().add_self_loop()
        return graph
