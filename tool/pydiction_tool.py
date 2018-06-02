#!/usr/bin/python
#coding: utf-8
#
#功能：整理python中的词汇生成到complete-dict文件，替换原有的complete-dict文件，就能在使用vim的时候产生自己需要的联想词汇。
#作者：gouqiang
#时间：2018/6/2
#用法：如果不带参数，运行python pydiction_toor.py，会搜索当前目录下的所有.py文件，并且把文件内所有词汇整理到当前目录中生成complete-dict文件
#	   如果带参数，运行python pydiction_toor.py [路径]，会搜索参数路径目录下的所有.py文件，并且把文件内所有词汇整理到当前目录中生成complete-dict文件
#	   用生成的complete-dict文件替换~/.vim/pydiction目录的文件。或则追加到~/.vim/pydiction目录的文件后
#

import os
import re
import sys

list_word = []

def main():
	file_path = os.path.abspath(sys.argv[0])
	dir_path_exe, file_exe = os.path.split(file_path)
	if len(sys.argv) == 2:
		dir_path = os.path.abspath(sys.argv[1])
	else:
		dir_path = dir_path_exe
	print "dir_path = " + dir_path
	get_py_path(dir_path)
#	print list_word
	print "word number = %s" % len(list_word)
	
	file_out = dir_path_exe + "/complete-dict"
	print "file_out = " + file_out
	f = open(file_out, 'a')
	for word in list_word:
		f.write(word + '\n')
	f.close()
	

#遍历输入的文件夹，得到所有.py文件的路径
def get_py_path(dir_path):
	for root, dirs, files in os.walk(dir_path):
		for filename in files:
		    if '.py' in os.path.splitext(filename):
				file_path = root + '/' + filename
				print "list = " + file_path
				file_handle(file_path)

#文件处理
def file_handle(file_path):
	f = open(file_path, 'r')
	for line in f.readlines():
		str_handle(line)
	f.close()

#字符串处理
flag_num = 0
def str_handle(line_str):
	global flag_num				#去掉多引号注释
	flag_num += line_str.count("\"\"\"")
	flag_num += line_str.count("\'\'\'")
	if flag_num != 0:
		if flag_num == 2:
			flag_num = 0
		return
	num = line_str.find('#')	#去掉‘#’以后的字符串
	if line_str.find('#') != -1 :
		line_str = line_str[:num]
	if line_str.count("\"") >= 2:	#去掉字符串内容
		num = line_str.find('\"')
		line_str1 = line_str[:num]
		line_str2 = line_str[num+1:]
		num = line_str2.find('\"')
		line_str2 = line_str2[num+1:]
		line_str = line_str1 + line_str2
	if line_str.count("\'") >= 2:	#去掉字符串内容
		num = line_str.find('\'')
		line_str1 = line_str[:num]
		line_str2 = line_str[num+1:]
		num = line_str2.find('\'')
		line_str2 = line_str2[num+1:]
		line_str = line_str1 + line_str2
	line_str = line_str.strip()	#去掉前后空白
	if line_str.strip():		#字符串不为空
#		print line_str
		words = re.split('=|\+| |\(|\)|\[|\]|\{|\}|\-|:|\/|\*|>|<|\'|\"|,', line_str)	#字符串切割
		for word in words:
			if len(word) > 2:	#长度必须大于2
				if not is_number(word):
#					print word
					if word not in list_word:	#去掉重复
						list_word.append(word)

#判断是否是数字
def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		pass 
	try:
		import unicodedata
		unicodedata.numeric(s)
		return True
	except (TypeError, ValueError):
		pass
	return False

if __name__ == '__main__':
	main()


