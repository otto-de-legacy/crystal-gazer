import os
import re
import subprocess
import tensorflow as tf
import core.loader as ld
import core.url_mapper as um
import core.trainer as tn
import core.network as nw
import operator
import core.config as cf
from multiprocessing import Process

cf = cf.Config()
allwds = dict()
url_to_cnt = dict()
tot_cnt = 0
single_select_cnt = 0
with open(cf.url_unique) as f:
    for line in f:
        la = line.replace("\n", "").split(",")

        if len(la) != 2: continue

        cnt = int(la[0])
        url_str = la[1]
        url_to_cnt[url_str] = cnt
        tot_cnt = tot_cnt + cnt
        wds = re.split(r'[\/&\?]', url_str)

        if cnt > 0 and cnt < 2:# and "order" in url_str:
            print(la)
            single_select_cnt = single_select_cnt + cnt

            for wd in wds:
                num = allwds.get(wd, 0)
                if num == 0:
                    allwds[wd] = 1
                else:
                    allwds[wd] = allwds[wd] + 1

for w in sorted(allwds, key=allwds.get, reverse=False):
    if allwds[w] > 1000:
        print(str(allwds[w]) + "  " + w)

print("tot_cnt: " + str(tot_cnt))
print("single_select_cnt: " + str(single_select_cnt))
