#! /usr/bin/python2
# -*-coding: UTF-8 -*-
import jpype
import os.path
import sys,time
import os

if __name__ == "__main__":
	#打开jvm虚拟机
	jar_path = os.path.abspath(u'/root/SegTest/hlsegsource/lib/')
	hlSegJarPath = os.path.join(jar_path, u"hlSegment-5.2.15.jar")
	
	#太坑了，版本只能用openjdk 8，否则-Djava.ext.dirs不支持，继而产生N+个BUG！！！
	jvmPath = u'/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/libjvm.so'
	jpype.startJVM(jvmPath, "-ea", "-Djava.class.path=%s" % hlSegJarPath, "-Djava.ext.dirs=%s" % jar_path)

	#取得类定义
	BasicSegmentor = jpype.JClass('com.hylanda.segmentor.BasicSegmentor')
	SegOption = jpype.JClass('com.hylanda.segmentor.common.SegOption')
	SegGrain = jpype.JClass('com.hylanda.segmentor.common.SegGrain')
	SegResult = jpype.JClass('com.hylanda.segmentor.common.SegResult')

	#创建分词对象
	segmentor = BasicSegmentor()
	

	#加载词典, 如果没有用户自定义词典，可以用下面的语句加载，自定义词典需要注意文件码制
	#if not segmentor.loadDictionary("../dictionary/CoreDict.dat", "../dictionary/userDict_utf8.txt"):
	if not segmentor.loadDictionary("../dictionary/CoreDict.dat", None):
		print "字典加载失败！"
		exit()

	#创建SegOption对象，如果使用默认的分词选项，也可以直接传空
	option = SegOption()
	option.mergeNumeralAndQuantity = False
	#可以使用下面的语句调整分词颗粒度
#	option.grainSize = SegGrain.LARGE

	#分词
	segResult = segmentor.segment(u"欢迎使用由天津海量信息技术股份有限公司出品的海量中文分词系统", option)

	#遍历并打印分词结果
	word = segResult.getFirst()
	print "\nWords: ",
	while(word != None):
		print word.wordStr,
		word = word.next
	
	jpype.shutdownJVM()
	exit()
