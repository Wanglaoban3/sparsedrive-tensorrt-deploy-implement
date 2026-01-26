python ./tools/test.py \
    projects/configs/sparsedrive_small_stage2.py \
    ckpt/sparsedrive_stage2.pth \
    --deterministic \
    --eval bbox \
    --out ./work_dirs/sparsedrive_small_stage2/results.pkl