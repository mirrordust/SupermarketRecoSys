#!/bin/sh

begindate=$1
enddate=$2

#begindate=20160801
#enddate=20160831

begin=`date -d "${begindate}" +%Y-%m-%d`
echo begindate: $begin
end=`date -d "${enddate}" +%Y-%m-%d`
echo enddate: $end

echo "export data..."
hive -e "
use recoclasreg;
select * from trade_sd_${begindate}_${enddate};" > ./csv_data/trade_data_${begindate}_${enddate}.csv
