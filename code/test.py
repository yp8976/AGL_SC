import mindspore as ms
from mindspore import Tensor, COOTensor

indices = Tensor([[0, 1, 2], [1, 0, 2]], dtype=ms.int32)
values = Tensor([0.1, 5, 4], dtype=ms.float32)
shape = (3, 3)
coo_tensor = COOTensor(indices.transpose(), values, shape)
print(coo_tensor.to_dense())
