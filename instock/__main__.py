#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified command line entrypoint for running InStock from source.

Examples:
    python -m instock init
    python -m instock job
    python -m instock job 2023-03-01
    python -m instock web
"""

import argparse
import importlib
import os.path
import sys

__author__ = 'myh '
__date__ = '2026/5/10 '

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
JOB_DIR = os.path.join(PROJECT_ROOT, 'instock', 'job')

COMMANDS = {
    'init': {
        'module': 'instock.job.init_job',
        'description': '初始化数据库和基础表',
        'extra_path': JOB_DIR,
    },
    'job': {
        'module': 'instock.job.execute_daily_job',
        'description': '执行整体数据抓取、处理和分析任务',
        'extra_path': JOB_DIR,
    },
    'realtime': {
        'module': 'instock.job.basic_data_daily_job',
        'description': '执行实时股票和 ETF 基础数据任务',
        'extra_path': JOB_DIR,
    },
    'selection': {
        'module': 'instock.job.selection_data_daily_job',
        'description': '执行综合选股任务',
        'extra_path': JOB_DIR,
    },
    'other': {
        'module': 'instock.job.basic_data_other_daily_job',
        'description': '执行非实时基础数据任务',
        'extra_path': JOB_DIR,
    },
    'after-close': {
        'module': 'instock.job.basic_data_after_close_daily_job',
        'description': '执行收盘后一到两小时才有的数据任务',
        'extra_path': JOB_DIR,
    },
    'indicators': {
        'module': 'instock.job.indicators_data_daily_job',
        'description': '执行股票指标数据任务',
        'extra_path': JOB_DIR,
    },
    'pattern': {
        'module': 'instock.job.klinepattern_data_daily_job',
        'description': '执行 K 线形态识别任务',
        'extra_path': JOB_DIR,
    },
    'strategy': {
        'module': 'instock.job.strategy_data_daily_job',
        'description': '执行策略选股任务',
        'extra_path': JOB_DIR,
    },
    'backtest': {
        'module': 'instock.job.backtest_data_daily_job',
        'description': '执行回测数据任务',
        'extra_path': JOB_DIR,
    },
    'web': {
        'module': 'instock.web.web_service',
        'description': '启动 Web 可视化服务',
    },
    'trade': {
        'module': 'instock.trade.trade_service',
        'description': '启动自动交易服务',
    },
}


def _prepend_sys_path(path):
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _load_command_module(command):
    command_info = COMMANDS[command]
    _prepend_sys_path(PROJECT_ROOT)
    _prepend_sys_path(command_info.get('extra_path'))
    return importlib.import_module(command_info['module'])


def _run_command(command, command_args):
    module = _load_command_module(command)
    sys.argv = [f"python -m instock {command}", *command_args]
    module.main()


def main():
    parser = argparse.ArgumentParser(
        prog='python -m instock',
        description='InStock 源码直运行入口，不需要记忆各个脚本路径。',
    )
    parser.add_argument(
        'command',
        choices=tuple(COMMANDS),
        help='要执行的功能命令',
    )
    parser.add_argument(
        'command_args',
        nargs='*',
        help='传给任务脚本的日期参数，例如 2023-03-01、2023-03-01,2023-03-02 或 2023-03-01 2023-03-21',
    )

    args = parser.parse_args()
    if len(args.command_args) > 2:
        parser.error('日期参数最多两个：单日/枚举日期传一个，区间日期传两个。')

    _run_command(args.command, args.command_args)


if __name__ == '__main__':
    main()
