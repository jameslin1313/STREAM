#!/bin/bash

# 設定主目錄的路徑
main_dir="/media/ai2lab/4TB SSD/Datasets/TransGlobe/Dataset-filled/leave-one-out/"

# 檢查主目錄是否存在
if [ -d "$main_dir" ]; then
    echo "列出 $main_dir 中的子目錄："
    # 使用 ls 和 grep 來篩選出目錄
    for dir in "$main_dir"/*/; do
        if [ -d "$dir" ]; then
            # 列出子目錄的名稱
            dir_name=$(basename "$dir")
            dir_id=$(echo "$dir_name" | sed 's/[^0-9]*//g' | awk '{print $1+0}')
            python to_xfund.py --dataset TransGlobe --data_type train --version_id "$dir_id"
            echo "=========================="
            python to_xfund.py --dataset TransGlobe --data_type test --version_id "$dir_id"

            echo $dir_id
            

        fi
    done
else
    echo "目錄 $main_dir 不存在"
fi



# python to_xfund.py --dataset TransGlobe --data_type test --version_id 0