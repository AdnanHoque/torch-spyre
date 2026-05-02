[CRITICAL] [core_division] buf0: per-core tensor span 512.00 MB (shape=[32768, 8192], dtype=torch.float16, device_size=[128, 32768, 64], splits={d0: 32, d1: 1, d2: 1}) exceeds hardware limit of 256.00 MB
terminate called after throwing an instance of 'DtException'
  what():  DtException: EAR overflow detected, file /home/adnan/dt-inductor/deeptools/dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp line 780
# Tile-shape probe — fixed M=128, N=8192, K ∈ [4096, 16384, 32768]
# warmup=3 iters=15


## Shape (128, 8192, 4096)  per-core flops = 268,435,456

| (m, n, 1) | M_per | N_per | wall ms | err |
|---|---:|---:|---:|---|
| ( 1,32,1) | 128 | 256 | 3.44 | |
| ( 2,16,1) | 64 | 512 | 3.44 | |
| ( 4, 8,1) | 32 | 1024 | 3.50 | |
| ( 8, 4,1) | 16 | 2048 | 3.47 | |
| (16, 2,1) | 8 | 4096 | 3.72 | |
| (32, 1,1) | 4 | 8192 | 4.70 | |

## Shape (128, 8192, 16384)  per-core flops = 1,073,741,824

| (m, n, 1) | M_per | N_per | wall ms | err |
|---|---:|---:|---:|---|
| ( 1,32,1) | 128 | 256 | 5.17 | |
| ( 2,16,1) | 64 | 512 | 4.94 | |
| ( 4, 8,1) | 32 | 1024 | 5.15 | |
| ( 8, 4,1) | 16 | 2048 | 4.99 | |
| (16, 2,1) | 8 | 4096 | 5.86 | |
| (32, 1,1) | 4 | 8192 | 10.07 | |

## Shape (128, 8192, 32768)  per-core flops = 2,147,483,648

| (m, n, 1) | M_per | N_per | wall ms | err |
|---|---:|---:|---:|---|
| ( 1,32,1) | 128 | 256 | 7.37 | |
| ( 2,16,1) | 64 | 512 | 6.83 | |
| ( 4, 8,1) | 32 | 1024 | 6.99 | |
| ( 8, 4,1) | 16 | 2048 | 6.91 | |
| (16, 2,1) | 8 | 4096 | 8.70 | |
| (32, 1,1) | 4 | 8192 | err | InductorError: CalledProcessError: Command '['dxp_ |
