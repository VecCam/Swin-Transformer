```bash
python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --pretrained swin_tiny_patch4_window7_224.pth --data-path {DATA_PATH} --batch-size 16 --accumulation-steps 2 [--use-checkpoint]
```