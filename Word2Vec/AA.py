
# coding: utf-8

# In[1]:


import json
import re
from multiprocessing import Pool
import os


# In[2]:


#regs
regs=[]
#restring="(%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s)" % (r'\((\w*\s*)+\)',r'\[(\w*\s*)+\]', r#'\(.*\)', r'\)',r'\[',r'\]',r'\w*',r'\d*',
 #                           r"[/$&+,:;=?@#|'<>^–*()%!-]",
 #                          r'”.*”',r'’.*’',r'".*"')
#regs=re.compile(restring)

regs.append(re.compile(r'\w+')) 
#regs.append(re.compile(r'\[(\w*\s*)+\]'))
regs.append(re.compile(r'\('))
regs.append(re.compile(r'\)'))
regs.append(re.compile(r'\['))
regs.append(re.compile(r'\]'))
regs.append(re.compile(r'\w*'))
regs.append(re.compile(r'\d*'))
regs.append(re.compile(r"[/$&+,:;=?@#|'<>^–*()%!-]"))
regs.append(re.compile(r'”'))
regs.append(re.compile(r'’'))
regs.append(re.compile(r'"'))

def clean(value):
    
    for i in regs:
    	value=i.sub('', value) 
    #print 'x'
    value=re.sub(r'"',"",value)
    #print 'y'    
    value=re.sub(r',',"",value)
    #print 'z'
    value=re.sub(r"'","",value)
    #print 'w'
    value=re.sub(r"\."," .",value)
    
    return value


# In[3]:

for r in ['AA/','AB/','AC/','AD/','AE/','source/AA/']:

	IP='/home/mak/Documents/corpus_raw/'+r
	OP='/home/mak/Documents/corpus_clean/'
	listing=os.listdir(IP)


	# In[5]:



	OUTPUTFILE=OP
	def clean_file(fileloc,filecount):
	    
	    h=open(fileloc)
	    f=h.read()
	    w=open(OUTPUTFILE+'corpus.txt','a+')
	    
	    g=f.split('\n')
	    dels=[]
	    for i in range(len(g)):
		if g[i].startswith('<doc') or g[i].startswith('</doc') or len(g[i])<100:
		    dels.append(i)
	    for i in dels[::-1]:
		del g[i]
	    lineno=0
	    for i in g:
		j=clean(i)
		if len(j)>100:
		    w.write(j)
		    w.write('\n')
		    #print lineno,j,len(j) 
		    #.encode('utf8'))
		    print lineno
		    #rint lineno
		    lineno+=1
	    w.close()
	    h.close()
	#et.parse('/home/mak/Documents/corpus_raw/AA/wiki_01')


	# In[ ]:

	print listing
	for i in listing:
	    clean_file(IP+i,i)
	    print i
	# In[ ]:



