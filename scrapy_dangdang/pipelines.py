import os
# 如果想使用管道的话 那么就必须在settings中开启管道
class ScrapyDangdangPipeline:

    def open_spider(self,spider):
        self.fp = open('book.json','w',encoding='utf-8')

    # item就是yield后面的book对象
    def process_item(self, item, spider):
        # 一下这种模式不推荐 因为每传递一个对象 那么就打开一次文件对文件的操作过于频繁
        # # write方法必须要写一个字符串 而不能是其他的对象
        # # w模式 会每一个对象都打开一次文件 覆盖之前的内容
        # with open('book.json','a',encoding='utf-8') as fp:
        #     fp.write(str(item))

        self.fp.write(str(item))
        return item

    def close_spider(self,spider):
        self.fp.close()

# 多条管道开启
    # 定义管道类
    # 在settings中开启管道
    # 'scrapy_dangdang.pipelines.DangDangDownloadPiepline': 301,
import urllib.request


class DangDangDownloadPiepline:

    def process_item(self,item,spider):
        url = 'http:' + item.get('src')
        if not os.path.exists('./books/'):
            os.mkdir('./books/')
        filename = './books/' + item.get('name') + '.jpg'
        urllib.request.urlretrieve(url=url,filename=filename)
        return item

