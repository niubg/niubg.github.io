---
title: JavaScript常见算法题

date: 2023-12-11 17:00:00 +0800

categories: [Deep Learning, Transformers]

tags: [deep learning, transformers]

math: true

mermaid: true

description: JavaScript算法题：二分查找、阶乘、数组合并排序
---


## 二分查找

>二分查找是一种非常高效的查找算法，前提是数组必须是有序的。下面是一个用 JavaScript 手写的简单二分查找算法：

``` javascript
function binarySearch(arr, target) {
  let left = 0;
  let right = arr.length - 1;

  while (left <= right) {
    let mid = Math.floor((left + right) / 2);

    if (arr[mid] === target) {
      return mid; // 找到目标元素，返回索引
    } else if (arr[mid] < target) {
      left = mid + 1; // 目标在右半部分
    } else {
      right = mid - 1; // 目标在左半部分
    }
  }

  return -1; // 未找到目标元素
}

```
- 这个二分查找函数接受一个有序数组 arr 和目标元素 target 作为参数，返回目标元素在数组中的索引，如果未找到则返回 -1。

- 时间复杂度：O(log n)
- 空间复杂度：O(1)

- 时间复杂度是 O(log n) 是因为每次比较都能将待搜索范围减半，直到找到目标元素或搜索范围为空。空间复杂度是 O(1) 是因为算法使用了固定数量的变量，不随输入规模的增大而增加。