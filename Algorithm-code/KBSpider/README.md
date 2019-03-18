## 构建爬虫系统实现网络知识获取 
### 爬虫框架图  
![Scrapy](./image/Scrapy.png)  
### 基础文件说明  
scrapy.cfg: 项目的配置文件。  
mySpider/: 项目的Python模块，将会从这里引用代码。  
mySpider/items.py: 项目的目标文件。  
mySpider/pipelines.py: 项目的管道文件。  
mySpider/settings.py: 项目的设置文件。  
mySpider/spiders/: 存储爬虫代码目录。  
