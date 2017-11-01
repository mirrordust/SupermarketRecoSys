#!/bin/sh

echo "export customer profile (huiyuanziliao) data..."
hive -e "
use recoclasreg;
select * from huiyuanziliao;" > ./csv_data/huiyuanziliao.csv
