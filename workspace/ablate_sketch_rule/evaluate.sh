#!/bin/bash

cur_time='TZ=UTC-8 date +"%Y-%m-%d %H:%M:%S"'
echo "Begin evaluationg on: "$(eval $cur_time)
#python -u no_schedule.py -b 16 -d 7
#python -u apply_tuned.py -b 64 -d 1 --tuned_dir ./result/0615-bs64
log_file=./log/bs64-default.debug
python -u print_best.py -b 64 -d 1 --tuned_dir ./result/0615-bs64/resnet-50-NHWC-B64-cuda.disable-.json > $log_file 2>&1 &
#python -u print_best.py -b 64 -d 1 --tuned_dir ./result/0615-bs64/resnet-50-NHWC-B64-cuda.disable-InitThreadBind.json > $log_file 2>&1 &
#python -u print_best.py -b 64 -d 1 --tuned_dir ./result/0615-bs64/resnet-50-NHWC-B64-cuda.disable-MutateAutoUnroll.json > $log_file 2>&1 &
echo "End at: $(eval $cur_time)"
