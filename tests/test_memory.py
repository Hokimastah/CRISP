import numpy as np

from crisp.memory import MemoryBank


def test_memory_add():
    memory = MemoryBank()
    memory.add(np.array([1.0, 0.0], dtype=np.float32), label="A")
    assert len(memory) == 1
    assert memory.labels() == ["A"]
