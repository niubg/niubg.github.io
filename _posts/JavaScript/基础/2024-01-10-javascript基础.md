---
title: JavaScript基本知识
date: 2024-01-10 16:56:10 +0800
categories: [JavaScript]
tags: [JavaScript]
description: JavaScript基本知识
---


## 基本类型与引用类型（栈 stack / 堆 heap）
当我们定义值类型时（基本类型），他是存储在内存 `栈 stack`， 引用类型的值它存储在 `堆 heap` 中

```js
const obj = {
    name: '饼干'
    city: '北京'
}
```
![Alt text](/assets/img/javaScript/stack.jpg)


## 深拷贝

示例代码：
```js
function deepClone(obj = {}) {
    if (typeof obj !== 'object' || obj == null) {
        return obj
    }

    // 初始化返回结果
    let result
    if (obj instanceof Array) {
        result = []
    } else {
        result = {}
    }

    for (let key in obj) {
        // 保证 key 不是原型的属性 hasOwnProperty 作用是判断否为为实例上的属性
        if (obj.hasOwnProperty(key)) {
            // 递归调用，deepClone(), 代码第一个 if 判断 处理了不是引用类型值时返回对应的值，此时这里直接就直接使用递归调用
            result[key] = deepClone(obj[key])
        }
    }

    // 返回结果
    return result
}
```

## 节流
节流经常在分页，按钮单击，表单提交等场景使用

代码示例：
```js
function throttle(fn, delay) {
    let timer = null;
    return function() {
        const _this = this
        if (!timer) {
            timer = setTimeout(() => {
                fn.apply(_this, arguments)
                clearTimeout(timer)
                timer = null
            })
        }
        
    }
}

let a1 = throttle(()=>{console.log(1)}, 1000)
a1()
a1()
a1()
// 结果输出是有一次 1
```

## 防抖
在特定的时间内重复触发只会在最后一次触发行为之后执行，使用场景：搜索输入、滚动监听，窗口大小拖拽等

代码示例：
```js
function debounce(fn, delay) {
    let timer = null
    return function() {
        clearTimeout(timer)
        timer = setTimeout(() => {
            fn.apply(this, arguments)
        }, delay)
    }
}

const a1 = debounce(() => {console.log(1)}, 1000)
a1()
a1()
a1()
// 会在最后一次调用方法1秒后执行输出结果 1
```

## 创建一个 new 方法
- `new` 一个对象发生了什么， 手写代码

```js
function customNew(constructor, ...args) {
    // 创建一个空对象,继承 constructor 的原型
    const obj = Object.create(constructor.prototype)
    // 将 obj 作为 this， 执行 constructor， 传入参数
    constructor.apply(obj, args)
    return obj
}
```

使用示例：
```js
function Foo(params) {
    this.name = params
}
Foo.prototype.getName = function() {
    console.log(this.name)
}
const f = customNew(Foo, '饼干')
f.getName() // 饼干
```

## 遍历数组 for 和 forEach 哪个更快
- `for` 执行效率更高

原因： `forEach` 每次都要创建一个函数来调用，而 `for` 不会创建函数。函数需要独立的作用域，会有额外的开销。