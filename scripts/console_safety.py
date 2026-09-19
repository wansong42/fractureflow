# -*- coding: utf-8 -*-
"""产品入口的控制台兜底 (R261-A-3 那条模式 lifted 到唯一实现, R268 T2).

客户机器上中文 Windows 的 cmd 默认码是 cp936/GBK —— 一个组合字符 (κ̂ 里的 U+0302) 或一个
对勾 (✓ U+2713) 就能让 `print` 抛 UnicodeEncodeError, 把整条链崩在**交付件已经算完但还没
打印完**的地方。R268 实测: 同一份合规输入, utf8 轴 rc=0, gbk 轴 rc=2 + 裸栈
(实测于内部控制台编码用例 d4/f4 两例; 证据件未随本发布分发)。

只放宽**错误处理**, 不改编码:
  * UTF-8 环境下没有字符会触发替换 ⇒ 对既有输出逐位无影响 (本单 J2 用 raw 字节 sha 证 no-op);
  * GBK 环境下不可编码字符退化为 `?` 而不是整单失败。

为什么是这一个文件而不是各抄一份: R261 F1 的教训就是"同一公式多份手写副本 = 八期 1/sin
事故的成因形态"。已知残余 = `scripts/auto_label_borehole.py` 内仍留它自己的同名实现
(R261 属主件 + R261 扫描器 GUARD_TOKENS 认这个 token), 由本单的 L5 扫描面显式登记。
"""
from __future__ import annotations

import sys


def install_console_safe_streams(streams=None) -> tuple:
    """把 stdout/stderr 的编码错误处理放宽为 replace; 返回各流的编码名 (探针记账用)。

    幂等: 已经是 replace 就原样返回。被测试框架/管道替换掉的流 (io.StringIO 之类) 没有
    reconfigure ⇒ 静默跳过, 绝不让"兜底"本身成为新的崩点。
    """
    done = []
    for s in ((sys.stdout, sys.stderr) if streams is None else streams):
        try:
            if (getattr(s, "errors", "") or "") != "replace":
                s.reconfigure(errors="replace")
            done.append(getattr(s, "encoding", ""))
        except (AttributeError, ValueError, OSError):
            pass
    return tuple(done)
