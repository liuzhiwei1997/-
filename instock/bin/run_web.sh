#!/bin/bash

/usr/local/bin/python3 /data/InStock/instock/web/web_service.py

echo ------Web服务已启动 请不要关闭------
echo 访问地址 : http://${web_host:-localhost}:${web_port:-9988}/
