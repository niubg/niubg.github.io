---
title: JavaScript常见算法题
date: 2023-05-23 17:00:00 +0800
categories: [JavaScript]
tags: [JavaScript]
keyword: JavaScript 算法题 二分查找 阶乘 数组合并排序 js去重
description: 算法题：二分查找、阶乘、数组合并排序
---

## 二分查找

>二分查找是一种非常高效的查找算法，前提是数组必须是有序的。下面是一个简单二分查找算法：

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

const arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const target = 7;
const result = binarySearch(arr, target);
console.log(result); // Output: 6

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


## 冒泡排序

- 冒泡排序是一种简单的排序算法，它通过多次遍历要排序的数组，比较相邻的两个元素，并交换它们（如果它们在错误的顺序中）。在每次遍历过程中，最大（或最小，取决于排序顺序）的元素会“冒泡”到数组的末尾。

```js
function bubbleSort(arr) {
    const len = arr.length;

    // 外部循环，控制需要进行比较的轮数
    for (let i = 0; i < len; i++) {
        // 内部循环，进行元素比较并交换
        for (let j = 0; j < len - 1 - i; j++) {
            // 比较相邻的两个元素，如果前一个元素比后一个元素大，则交换它们
            if (arr[j] > arr[j + 1]) {
                [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]]; // 使用解构赋值进行交换
            }
        }
    }

    return arr;
}

// 示例
const arr = [5, 3, 8, 4, 2, 1];
console.log(bubbleSort(arr)); // 输出：[1, 2, 3, 4, 5, 8]

```


## 获取首个不重复的字符
- 给出一个字符串如：`aabbcddefgg`,获取当前字符串首个不重复的字符。

``` js
  function firstUniqChar(str) {
    let charCount = {};

    for (let i = 0; i < str.length; i++) {
      const char = str[i]
      if (charCount[char]) {
        // 如果有重复的将重复出现在charCount里的字符删除
        delete charCount[char]
      } else {
        charCount[char] = 1; // 当前赋值只是标记作用，可以为任何值
      }
    }

    for (let key in charCount) {
      return key
    }
  }


// 使用示例：
const str = 'aabbcddefgg';
const result = firstUniqChar(str);
console.log(result); // 输出 'c'

```
或使用数组方式优化
```js
  function firstUniqChar(str) {
    let charArr = [];
    
    for (let i = 0; i < str.length; i++) {
      const char = str[i]
      const arrIndex = charArr.indexOf(char)
      if (arrIndex > -1) {
        // delete charArr[arrIndex]
        charArr.splice(arrIndex, 1)
      } else {
        charArr.push(char)
      }
    }

    return charArr[0]
  }

const str = 'aabbcddefgg';
const result = firstUniqChar(str);
console.log(result); // 输出 'c'
```


## 有效括号

检查一个字符串中的括号是否有效（即左右括号匹配）：

```javascript
function isValidParentheses(s) {
  const stack = [];
  const parenthesesMap = {
    '(': ')',
    '[': ']',
    '{': '}'
  };

  for (let i = 0; i < s.length; i++) {
    const char = s[i];

    if (parenthesesMap[char]) {
      // 如果是左括号，将其推入栈中
      stack.push(char);
    } else {
      // 如果是右括号，检查栈顶元素是否匹配
      const top = stack.pop();
      if (parenthesesMap[top] !== char) {
        return false; // 括号不匹配
      }
    }
  }

  return stack.length === 0; // 栈为空表示括号全部匹配
}

// 示例
console.log(isValidParentheses("()"));        // true
console.log(isValidParentheses("()[]{}"));    // true
console.log(isValidParentheses("(]"));        // false
console.log(isValidParentheses("([)]"));      // false
console.log(isValidParentheses("{[]}"));      // true
```

这个函数使用了栈（stack）的数据结构来跟踪左括号，遇到右括号时检查是否与栈顶的左括号匹配。如果匹配，将左括号出栈；如果不匹配，返回 `false`。最终，如果栈为空，说明所有括号都有匹配，返回 `true`；否则，返回 `false`。

这个算法的时间复杂度是 O(n)，其中 n 是输入字符串的长度。



## 链表式队列

讲到队列我们就会想到栈，那么栈和队列的区别：
- 栈 - 后进先出
- 队列 - 先进先出

通常我们要使用js实现一个队列的话，首先想到的就是数组形式，如：

```js
// 声明一个数组存放队列集合
let queue = [];

queue.unshfit(1) //  [1]
queue.unshfit(2) // [2, 1]
queue.unshfit(3) // [3, 2, 1]

// 先进先出原则使用pop退出第一个进入的值
queue.pop() // [3, 2]

```
如果考虑时间复杂度的话，上面的数组形式就不是最优选择，所以要使用链表形式。


如下是链表形式：

```js
//  链表式队列
class MyQueue {
    constructor() {
        this.length = 0 // 长度
        this.head = null
        this.tail = null
    }

    // 从tail入队
    add(value) {
        const newNode = {value}

        if (this.length === 0) {
            // 长度是0
            this.head = newNode
            this.tail = newNode
        } else {
            // 长度 > 0,把 newNode 拼接到 tail 位置
            this.tail.next = newNode
            this.tail = newNode
        }

        this.length++ // 累加长度
    }

    // 从 head 出队
    delet() {
        if (this.length <= 0) return null
        
        let value = null

        if (this.length === 1) {
            // 长度是 1 ，只有一个元素了
            value = this.head.value // 先找到结果
            // 重置 head tail
            this.head = null
            this.tail = null            
        } else {
            // 长度 > 1, 多个元素
            value = this.head.value // 先找到结果
            this.head = this.head.next  // 重置 head
        }

        // 减少长度， 返回结果
        this.length--
        return value
    }

}

// 功能测试
const queue = new MyQueue()
queue.add(100)
queue.add(200)
queue.add(300)

queue.delet()   // 返回 100
queue.delet()   // 返回 200
queue.delet()   // 返回 300
```