---
title: node如何开启多进程
date: 2024-3-2 13:29:00 +0800
categories: [node]
tags: [node]
description: nodejs如何开启进程，进程如何通讯
---


## 进程 process vs 线程 thread
- 进程， OS 进行资源分配和调度的最小单位，有独立的内存空间
- 线程， OS 进行运算调度的最小单位，共享进程内存空间
- JS 是单线程的，但可以开启多进程执行，如 `WebWorker`
