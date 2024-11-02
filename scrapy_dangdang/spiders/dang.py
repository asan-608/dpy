import scrapy
from scrapy_dangdang.items import ScrapyDangdangItem


class DangSpider(scrapy.Spider):
    # 爬虫的名字 一般运行爬虫的时候 使用的值
    name = 'dang'

    # 允许访问的域名
    # 如果是多页下载的话 那么必须要调整的是allowed_domains的范围 一般情况下只写域名
    allowed_domains = ['category.dangdang.com']

    # 起始的url地址  指的是第一次要访问的域名
    # start_urls   是在allowed_domains的前面添加一个http：//
    #              是在allowed_domains的后面添加一个/
    # 如果以html结尾 就不用加/
    start_urls = ['http://category.dangdang.com/cp01.01.02.00.00.00.html']
    base_url = 'http://category.dangdang.com/pg'
    page = 1

    # 是执行了start_urls之后  执行的方法
    # 方法中的response  就是返回的那个对象
    # 相当于 response = urllib.request.urlopen()
    #       response = requests.get()
    def parse(self, response):
        # pipelines     下载数据
        # items         定义数据结构的
        # src = //ul[@id="component_59"]/li//img/@src
        # alt = //ul[@id="component_59"]/li//img/@alt
        # price = //ul[@id="component_59"]/li//p[@class="price"]/span[1]/text()
        # 所有的seletor的对象 都可以再次调用xpath方法
        li_list = response.xpath('//ul[@id="component_59"]/li')
        for li in li_list:
        #  第一张图片和其他的图片的标签是属性是不一样的
        #  第一张图片src是可以使用的 其他图片的地址data-original
            src = li.xpath('.//img/@data-original').extract_first()
            if src:
               src = src
            else:
               src = li.xpath('.//img/@src').extract_first()

            name = li.xpath('.//img/@alt').extract_first()
            price = li.xpath('.//p[@class="price"]/span[1]/text()').extract_first()

            book = ScrapyDangdangItem(src=src,name=name,price=price)
            # 获取一个book就交给pipelines
            yield book

        # 每一页爬取的业务逻辑都是一样的
        # 所以我们只需要将执行的那个页的请求再次调用parse方法就可以了
        if self.page < 100:
            self.page = self.page + 1
            url = self.base_url + str(self.page) + '-cp01.01.02.00.00.00.html'
            # 怎么去调用parse方法
            # scrapy.Request就是scrpay的get方法
            # url就是请求地址
            # callback是你要执行的那个函数 注意不需要加圆括号
            yield scrapy.Request(url=url,callback=self.parse)
