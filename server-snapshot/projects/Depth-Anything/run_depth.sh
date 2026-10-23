#!/bin/bash

# ================= [配置区域] =================
# 数据集根目录 (请确保路径正确)
DATA_ROOT="/root/autodl-tmp/.autodl/Deformable-3D-Gaussians/D_NeRF_Data"

# 使用的编码器 (与你下载的权重对应)
ENCODER="vits"

# 需要处理的分辨率文件夹名称
RESOLUTIONS=("1x")
# ============================================

echo "============================================="
echo "开始批量提取深度图"
echo "数据集路径: $DATA_ROOT"
echo "使用模型: Depth Anything ($ENCODER)"
echo "============================================="

# 遍历 DATA_ROOT 下的所有场景文件夹
for SCENE_PATH in "$DATA_ROOT"/*; do
    # 确保它是一个目录
    if [ -d "$SCENE_PATH" ]; then
        # 获取场景的文件夹名称 (例如 basin_novel_view)
        SCENE_NAME=$(basename "$SCENE_PATH")
        echo -e "\n>>> 正在处理场景: $SCENE_NAME <<<"

        # 遍历每个分辨率
        for RES in "${RESOLUTIONS[@]}"; do
            IMG_PATH="$SCENE_PATH/rgb/$RES"
            OUT_DIR="$SCENE_PATH/rgb/${RES}_depth"

            # 检查该分辨率的 rgb 文件夹是否存在
            if [ -d "$IMG_PATH" ]; then
                echo "  -> 提取 [$RES] 分辨率深度图..."
                
                # 执行你的 Python 命令
                python run.py \
                    --encoder "$ENCODER" \
                    --img-path "$IMG_PATH" \
                    --outdir "$OUT_DIR" \
                    --pred-only
                
                echo "  -> [$RES] 完成，保存在 $OUT_DIR"
            else
                echo "  -> [跳过] 找不到输入目录: $IMG_PATH"
            fi
        done
    fi
done

echo -e "\n============================================="
echo "所有场景深度图提取完毕！"
echo "============================================="