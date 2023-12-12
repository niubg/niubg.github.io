---
title: WeakMap和weakSet
date: 2023-12-12 16:42:00 +0800
categories: [JavaScript]
tags: [JavaScript]
description: WeakMap、WeakSet
---

- WeakMap 和 WeakSet 是 ES6 中引入的两个新数据结构，它们可以用来存储弱引用。弱引用是指不会阻止对象被垃圾回收的引用。

- WeakMap 和 WeakSet 与 Map 和 Set 类似，但它们的行为略有不同。Map 和 Set 中的键和值都是强引用，这意味着它们不会被垃圾回收。如果一个 Map 或 Set 中的键或值被垃圾回收，那么整个 Map 或 Set 都会被垃圾回收。

- WeakMap 和 WeakSet 中的键和值都是弱引用，这意味着它们可以被垃圾回收。如果一个 WeakMap 或 WeakSet 中的键或值被垃圾回收，那么该键或值将从 WeakMap 或 WeakSet 中自动删除。

- WeakMap 和 WeakSet 可以用来存储对不可变对象的引用。不可变对象不会被修改，因此它们不会被垃圾回收。这意味着 WeakMap 和 WeakSet 中的键和值不会被垃圾回收，即使它们不再被使用。

- WeakMap 和 WeakSet 也可以用来存储对函数的引用。函数是可变对象，因此它们可能会被垃圾回收。如果一个 WeakMap 或 WeakSet 中的函数被垃圾回收，那么该函数将从 WeakMap 或 WeakSet 中自动删除。

- WeakMap 和 WeakSet 可以用来实现一些有用的功能。例如，它们可以用来存储对 DOM 元素的引用，或者用来存储对事件处理程序的引用。

以下是一个使用 WeakMap 的例子：
```js
const weakMap = new WeakMap();

weakMap.set(myObject, 'value');

console.log(weakMap.get(myObject)); // 'value'

myObject = null;

console.log(weakMap.get(myObject)); // undefined
```

以下是一个使用 WeakSet 的例子：
```js
const weakSet = new WeakSet();

weakSet.add(myObject);

console.log(weakSet.has(myObject)); // true

myObject = null;

console.log(weakSet.has(myObject)); // false
```