# -*- coding: utf-8 -*-
import jieba
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

path_in=u'./testing/'

file_in=u'CTB7_test.utf8'
#file_in=u'as_test.utf8'
#file_in=u'cityu_test.utf8'
#file_in=u'msr_test.utf8'
#file_in=u'pku_test.utf8'

fr = open(path_in+file_in)

path_out=u'./result/jieba/'

file_out=u'CTB7_result.utf8'
#file_out=u'as_result.utf8'
#file_out=u'cityu_result.utf8'
#file_out=u'msr_result.utf8'
#file_out=u'pku_result.utf8'

fw = open(path_out+file_out,'a+')

for line in fr.readlines():
	seg_list = jieba.cut(line, cut_all=False)  # 精确模式
	fw.write('　'.join(seg_list))


fr.close()
fw.close()