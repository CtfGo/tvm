#!/bin/bash

ablated_rules="RuleAddCacheRead RuleSpecialComputeLocationGPU RuleAlwaysInline RuleSimplifyComputeWithConstTensor RuleCrossThreadReduction RuleAddCacheWrite RuleMultiLevelTilingWithFusion RuleMultiLevelTiling InitFillTileSize InitThreadBind InitUnroll MutateTileSize MutateAutoUnroll"

cur_time='TZ=UTC-8 date +"%Y-%m-%d %H:%M:%S"'
echo "Default Tuning with bathc_size=16 at: "$(eval $cur_time)
python -u tune_network_cuda.py -b 16 -d 6 -n 300 --tuned_dir ./result/0613-bs16 > ./log/0613-bs16/default.log 2>&1
echo "Default Tuning with bathc_size=64 at: "$(eval $cur_time)
python -u tune_network_cuda.py -b 64 -d 6 -n 3000 --tuned_dir ./result/0614-bs64 > ./log/0614-bs64/default.log 2>&1

echo "Begin ablating rules at: "$(eval $cur_time)
for rule in $ablated_rules; do
    log_file=./log/0614-bs64/disable-$rule.log
    echo "Start test at:$(eval $cur_time), rule: $rule, log: $log_file"
    python -u tune_network_cuda.py -b 64 -d 6 -n 3000 --tuned_dir ./result/0614-bs64 -e $rule > $log_file 2>&1
    echo "End at: $(eval $cur_time)"
done
