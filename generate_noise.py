bs = 1
epoch = 10
steps = 500
rank = 8
noise_shape = [1, 1, 16, 21, 60, 104] # 期望生成noise的维度，第一维度不变！
output_path = "/home/hemuhui"

for e in range(epoch): # 10个epoch
output_path = f"{output_path}/generated_noise/epoch_{e}"
os.makedirs(output_path, exist_ok=True)

for r in range(rank): # 8个rank
    t = []  # [torch.randint(0, steps, (bs,), device=device, dtype=torch.int64)]
    noise = []  # [torch.rand(noise_shape, device=device)]

    # 生成一个大tensor，包含step步数个noise
    for i in range(steps): # 500步
        t_i = torch.randint(0, steps, (bs,), device=device, dtype=torch.int64)
        t.append(t_i)

        noise_i = torch.rand(noise_shape, device=device)
        noise.append(noise_i)
        if i % 100 == 0:
            torch.cuda.synchronize()
            print(f"=================================epoch{e},step{i}")
        #torch.cuda.empty_cache()

    # 保存 timestep
    t = torch.cat(t, dim=0)
    print(f"*************timestep shape: {t.shape}")
    torch.save(t, f"{output_path}/timestamps_{r}.pth")


    # 保存noise
    noise = torch.cat(noise, dim=0)
    print(f"*************noise shape: {noise.shape}")
    torch.save(noise, f"{output_path}/noise_{r}.pth")
