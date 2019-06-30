# -*- coding: utf-8 -*
import json
import os, sys

ori_file = sys.argv[1]

with open(ori_file, 'r') as fin:
	ori = fin.read()
	single_json = "[" + ori.replace('}\n{','},{') + "]"
	samples = json.loads(single_json)
	print 'transformed and load {} samples'.format(len(samples))
	out_file = ori_file+'.trans'
	with open(out_file, 'w') as fout:
		fout.write(json.dumps(samples, indent=4, ensure_ascii=True)+ '\n')
	print 'transformed, and write to {}'.format(out_file)
	
	


