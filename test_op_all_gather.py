import torch

def validate_allgather(rank, world_size, device, input_numel=5, dtype=torch.bfloat16, excu_nums=94):
    all_gahter_output = torch.empty(input_numel*world_size, dtype=dtype, device=device)
    all_gather_input = all_gahter_output.narrow(0, input_numel*rank, input_numel)
    
    all_gather_input = all_gather_input.clone()
    
    for _ in range(excu_nums):
        all_gather_work = torch.distributed.all_gahter_into_tensor(
            output_tensor=all_gather_output,
            input_tensor=all_gather_input,
        )
