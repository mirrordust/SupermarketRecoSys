#!/bin/sh

begindate=$1
enddate=$2

#begindate=20160801
#enddate=20160831

begin=`date -d "${begindate}" +%Y-%m-%d`
echo begindate: $begin
end=`date -d "${enddate}" +%Y-%m-%d`
echo enddate: $end

hive -e "
use recoclasreg;
drop table if exists trade_sd_${begindate}_${enddate};
create table trade_sd_${begindate}_${enddate} as
select *
from trade_sd
where to_date(sldat) between '${begin}' and '${end}';"
#and pluno not in ('30380001', '30380002', '30380003');"

#30380001 小号塑料购物袋
#30380001 大号塑料购物袋
#30380001 中号塑料购物袋

# hive -e "
# use reco;
# drop table if exists testset_${begindate}_${enddate};
# create table testset_${begindate}_${enddate} as
# select vipno, pluno, count(*) as cnt, sum(qty) as qty, sum(amt) as amt
# from trade_${begindate}_${enddate}
# group by vipno,pluno;"
