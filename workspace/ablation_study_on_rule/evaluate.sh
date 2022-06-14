#!/bin/bash

cur_time='TZ=UTC-8 date +"%Y-%m-%d %H:%M:%S"'
echo "Begin evaluationg on: "$(eval $cur_time)
echo "End at: $(eval $cur_time)"
