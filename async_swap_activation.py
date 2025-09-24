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


# h2d_stream/d2h_stream: 执行swap操作的stream，一般不用计算和通信的流，单独创建一条流，实现一步操作
# block idx：当前layer的编号，前向计算是，函数会在i+1层开始计算的时候触发第i层激活值得H2D；反向计算时，会在第i+1层计算时，触发第i层的d2H
# depth：模型总层数
# custom_check_fn:可以自己定义校验函数，满足该校验函数的激活值才会被offload
