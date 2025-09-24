import torch
import torch_npu
from mindspeed_mm.utils.async_offload import async_save_on_cpu

# 模型初始化的时候初始化swap流
swap_stream = torch_npu.npu.Stream()

# 在模型的forward函数中
for idx， decode_layer in enumerate(self.layers):
    with async_save_on_cpu(
      h2d_stream = swap_stream,
      d2h_stream = swap_stream,
      block_idx = int(idx),
      depth=len(self.layers),
      custom_check_fn = lambda x: x.data_ptr()==hidden_states.data_ptr() 
    ):
          hidden_states, router_results = decode_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx
          )
