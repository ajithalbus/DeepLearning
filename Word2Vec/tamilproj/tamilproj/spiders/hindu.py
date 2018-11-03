# -*- coding: utf-8 -*-
import scrapy
import re
import unicodedata
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

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
regs.append(re.compile(r"[/$&+,:;{}=?@#|'<>^–*()%!-]"))
regs.append(re.compile(r'”'))
regs.append(re.compile(r'’'))
regs.append(re.compile(r'"'))
regs.append(re.compile(r'\.{2,3}'))

def _clean(value):
    
    for i in regs:
    	value=i.sub('', value) 
    #print 'x'
    value=re.sub(r'"'," ",value)
    #print 'y'    
    value=re.sub(r','," ",value)
    #print 'z'
    value=re.sub(r"'"," ",value)
    value=re.sub(r"’"," ",value)
    value=re.sub(r"‘"," ",value)
    value=re.sub(r"\."," ",value)
    value=re.sub(r"\“","",value)
    value=re.sub(r"\‘","",value)
    value=re.sub(r"“","",value)
    value=re.sub(r"‘","",value)
    
    #value=re.sub(r"\”","",value)
    #value=re.sub(r'”',"",value)
    
    #value=re.sub(r"\“","",value)
    #value=re.sub(r'“',"",value)
    
    #value=re.sub(r"\’","",value)
    #value=re.sub(r"’","",value)
    #print 'w'
    
    return value


class HinduSpider(CrawlSpider):
    name = 'hindu'
    allowed_domains = ['thehindu.com']
    start_urls = ['http://tamil.thehindu.com']
    rules = (
        Rule(LinkExtractor(allow="http://tamil\.thehindu\.com/.+_.+"),
             callback='parse'),
    )
    def parse(self, response):
        
        filename = 'log.txt'
        f=open(filename, 'a+')
        #self.log('HERE')
        for line in response.css('body > div.container > section > article > div > div > div > div.bodycontent.mtop').css('p::text').extract() :
            line=line.replace(',',u' ')
            line=_clean(line)
             
            if len(line)<=30:
                
                #self.log("----")
                continue
            self.log("----")
            f.write(line.encode('utf8'))
            #self.log(len(line))
            #f.write('\n')
        '''
        for line in response.css('body > div.container > section > article > div > div > div > div:nth-child(11)').css('div').css('p::text').extract() :
            line=line.replace(',',u' ')
            line=_clean(line)
             
            if len(line)<=30:
                
                #self.log("----")
                continue
            self.log("----")
            f.write(line.encode('utf8'))
            #self.log(len(line))
            #f.write('\n')
        '''
        f.close()
        
        next_page =response.css('a::attr(href)').extract()
        for link in next_page:
            #nself.log(link)
            #nlink = response.urljoin(link)
            if link.startswith('http://tamil.'):
                yield scrapy.Request(link, callback=self.parse)        
        
        

# -*- coding: utf-8 -*-


  
