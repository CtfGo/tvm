#!/bin/bash

#ablated_rules="RuleAddCacheRead RuleSpecialComputeLocationGPU RuleAlwaysInline RuleSimplifyComputeWithConstTensor RuleCrossThreadReduction RuleAddCacheWrite RuleMultiLevelTilingWithFusion RuleMultiLevelTiling InitFillTileSize InitThreadBind InitUnroll MutateTileSize MutateAutoUnroll"
ablated_rules="RuleAddCacheWrite RuleMultiLevelTilingWithFusion InitFillTileSize"

cur_time='TZ=UTC-8 date +"%Y-%m-%d %H:%M:%S"'
#echo "Default Tuning with batch_size=64 at: "$(eval $cur_time)
#python -u tune_network_cuda.py -b 64 -d 7 -n 300 --tuned_dir ./result/0615-bs64 > ./log/0615-bs64/default.log 2>&1

echo "Begin ablating rules with bs=16 at: "$(eval $cur_time)
for rule in $ablated_rules; do
    log_file=./log/0620-pair/bs16-disable-RuleCrossThreadReduction-$rule.log
    echo "Start test at:$(eval $cur_time), rule: $rule, log: $log_file"
    python -u tune_network_cuda.py -b 16 -d 7 -n 300 --tuned_dir ./result/0620-pair -e RuleCrossThreadReduction -e $rule > $log_file 2>&1
    echo "End at: $(eval $cur_time)"
done

echo "Begin ablating rules with bs=64 at: "$(eval $cur_time)
for rule in $ablated_rules; do
    log_file=./log/0620-pair/bs64-disable-RuleCrossThreadReduction-$rule.log
    echo "Start test at:$(eval $cur_time), rule: $rule, log: $log_file"
    python -u tune_network_cuda.py -b 64 -d 7 -n 300 --tuned_dir ./result/0620-pair -e RuleCrossThreadReduction -e $rule > $log_file 2>&1
    echo "End at: $(eval $cur_time)"
done
