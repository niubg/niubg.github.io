---
title: JavaScript常见算法题
date: 2023-05-23 17:00:00 +0800
categories: [JavaScript]
tags: [JavaScript]
description: 算法题：二分查找、阶乘、数组合并排序
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


## 阶乘

> 阶乘是一个数学概念，表示一个非负整数n与小于等于n的所有正整数的乘积。递归是一种解决问题的方法，其中函数调用自身。下面使用递归计算阶乘的JavaScript函数

``` js
    function factorial(n) {
      // 基本情况：阶乘的定义是 0 和 1 的阶乘均为 1
      if (n === 0 || n === 1) {
        return 1;
      } else {
        // 递归调用：n 的阶乘等于 n 乘以 (n-1) 的阶乘
        return n * factorial(n - 1);
      }
    }

    // 示例
    console.log(factorial(5)); // 输出 120

```

## 数组合并排序

1、第一中方可以使用`concat()`方法将两个数组合并，然后使用`sort()`方法对合并后的数组进行排序。
```js
  let array1 = [1, 3, 5];
  let array2 = [2, 4, 6];

  // 合并两个数组
  let mergedArray = array1.concat(array2);

  // 对合并后的数组进行排序
  let sortedArray = mergedArray.sort(function(a, b) {
      return a - b; // 升序排序
      // 如果需要降序排序，可以使用 return b - a;
  });

  console.log(sortedArray);

```

2、第一种固然是可以但是如果我们考虑到了时间复杂度情况下，针对两个已排序的数组，想要合并保持有序，可以使用归并排序（Merge Sort）的思想。

```js
  function mergeSortedArrays(arr1, arr2) {
    const mergedArray = [];
    let i = 0; // 索引指向arr1
    let j = 0; // 索引指向arr2

    // 合并排序
    while (i < arr1.length && j < arr2.length) {
      if (arr1[i] < arr2[j]) {
        mergedArray.push(arr1[i]);
        i++;
      } else {
        mergedArray.push(arr2[j]);
        j++;
      }
    }

    // 将剩余元素添加到mergedArray中
    while (i < arr1.length) {
      mergedArray.push(arr1[i]);
      i++;
    }

    while (j < arr2.length) {
      mergedArray.push(arr2[j]);
      j++;
    }

    return mergedArray;
  }

  // 示例
  const arr1 = [1, 3, 5, 7];
  const arr2 = [2, 4, 6, 8];

  const mergedAndSortedArray = mergeSortedArrays(arr1, arr2);
  console.log(mergedAndSortedArray);

```
这个函数通过比较两个数组的元素，逐一选择较小的元素添加到结果数组中，以此保持有序性。最后，将剩余的元素添加到结果数组中。这个过程的时间复杂度是O(m + n)，其中m和n分别是两个数组的长度。

如果要合并多个有序数组，可以多次调用这个函数。这里需要注意的是，如果有多个数组，合并过程可能不是最优的，可以考虑使用优先队列（Heap）等数据结构来更高效地进行多个有序数组的合并。