---
title: "Web crawler"
permalink: /pdocs/webcrawler/
excerpt: "Web crawler"
sitemap: true

sidebar:
  nav: "docs"

---

This is a small project that I use python scrapy to get stock information automatically, and connect to a telegram chat bot

## Clawer
- Base on Python scrapy, and use xpath to extract the information I need
- Target website include
    - https://www.boursedirect.fr/
    - http://www.aastocks.com/

```python
class BSSpider(scrapy.Spider):
    name = "usnotice"

    def __init__(self):
        self.collect = pd.DataFrame(columns=["open","close","vari"],index=["ShangHai:","GoldER:",
                                                                           "DJI  :","NASDAQ:","PSI:","sp500:","ftse:"])
        self.count = 0

    def start_requests(self):
        shanzen = "http://www.aastocks.com/tc/stocks/market/index/china-index.aspx"
        yield scrapy.Request(url = shanzen , callback = self.craw_shanzen)
        golder = "https://www.boursedirect.fr/fr/marche/chicago-mercantile-exchange/s-p-gsci-gold-index-excess-ret-SPGSGCP-USD-XCME/seance"
        yield scrapy.Request(url=golder, callback=self.craw)
        dji = "https://www.boursedirect.fr/fr/marche/new-york-stock-exchange-inc/dow-jones-industrial-average-DJI-USD-XNYS/seance"
        yield scrapy.Request(url=dji, callback=self.craw)
        nsd = "https://www.boursedirect.fr/fr/marche/nasdaq-all-markets/nasdaq-100-NDX-USD-XNAS/seance"
        yield scrapy.Request(url=nsd, callback=self.craw)
        psi = "https://www.boursedirect.fr/fr/marche/nasdaq-all-markets/phlx-semiconductor-SOX-USD-XNAS/seance"
        yield scrapy.Request(url=psi, callback=self.craw)
        sp500 = "https://www.boursedirect.fr/fr/marche/chicago-mercantile-exchange/s-p-500-SP500-USD-XCME/seance"
        yield scrapy.Request(url=sp500, callback=self.craw)
        ftse = "https://www.boursedirect.fr/fr/marche/no-market-e-g-unlisted/ftse-china-a50-index-XIN9-CNY-XXXX/seance"
        yield scrapy.Request(url=ftse, callback=self.craw)

    def craw_shanzen(self,response):
        for idx in [2]:
            table = response.xpath('//div[contains(@class , "content")]//table//tr')
            all = table[idx].xpath('td//text()').extract()
            # print(all)
            open = float(all[-2].replace(",",""))
            close = float(all[2].replace(",",""))
            vari_value = round((close-open),2)
            vari = str(vari_value) + " (" + str(round(vari_value*100/open,2))+"%)"
            combine = [open, close, vari]
```
- Crawler output
![](https://i.imgur.com/i3svAIl.png)
- Use windows schedule to update daily

## Chat bot

```python
    def sendbot(self,text):
        TOKEN = "MY TOKEN"
        UPDATE = "PATH OF MY BOT"
        # rtmess = urllib.parse.quote_plus(text)
        path = r"D:\python learn\scrapy\notice\notice\spiders\table.png"
        url = "https://api.telegram.org/bot"+TOKEN+"/sendPhoto"
        for ID in ["***********" ,"************" ]:
            files = {'photo': open('table.png', 'rb')}
            data = {'chat_id': ID}
            requests.post(url, files=files, data=data)
```
