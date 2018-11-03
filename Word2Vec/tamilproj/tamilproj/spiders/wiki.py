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
regs.append(re.compile(r"[/$&+,:;=?@#|'<>^–*()%!-]"))
regs.append(re.compile(r'”'))
regs.append(re.compile(r'’'))
regs.append(re.compile(r'"'))

def _clean(value):
    
    for i in regs:
    	value=i.sub('', value) 
    print 'x'
    value=re.sub(r'"',"",value)
    print 'y'    
    value=re.sub(r',',"",value)
    print 'z'
    value=re.sub(r"'","",value)
    print 'w'
    
    return value

  
class WikiSpider(CrawlSpider):
    name = 'wiki'
    allowed_domains = ['wikipedia.org']
    start_urls = ['https://ta.wikipedia.org/wiki/']#%E0%AE%95%E0%AF%8A%E0%AE%B4%E0%AF%81%E0%AE%AA%E0%AF%8D%E0%AE%AA%E0%AF%81_%E0%AE%85%E0%AE%AE%E0%AE%BF%E0%AE%B2%E0%AE%AE%E0%AF%8D']
    rules = (
        Rule(LinkExtractor(allow="https://en\.wikipedia\.org/wiki/.+_.+"),
             callback='parse'),
    )
    def parse(self, response):
        
        filename = 'log.txt'
        f=open(filename, 'a+')
        
        for line in response.css('#mw-content-text > div').css('p').css('p::text,a::text,b::text,i::text').extract():
            #line=line.replace('\n','')
            line=line.replace('.',u'\u000d') #new line
            line=line.replace(',',u' ')
            line=_clean(line)
             
            if len(line)<=2:
                
                #self.log("----")
                continue
            f.write(line.encode('utf8'))
            #self.log(len(line))
            #f.write('\n')
        f.close()
        
        next_page =response.css('#mw-content-text > div').css('p').css('a::attr(href)').extract()
        for link in next_page:
            self.log(link)
            link = response.urljoin(link)
            yield scrapy.Request(link, callback=self.parse)        
        
        
