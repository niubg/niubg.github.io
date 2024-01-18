---
title: javascript手写实现 call apply bind
date: 2024-01-09 20:15:00 +0800
categories: [JavaScript]
tags: [JavaScript]
description: javascript手写实现 call、apply、bind
---

## 实现 call
```js
Function.prototype.myCall = function(context, ...args) {
    // 如果 context 为 null 或 undefined，则设置为全局对象（浏览器环境下为 window）
    context = context || window

    // 将当前调用 myCall 方法的函数设置为传入的函数(当前的 this 指向调用者)
    context.fn = this

    // 执行函数并出入参数
    const result = context.fn(...args)
    
    // 删除临时追加的 fn 属性
    delete context.fn

    return result
}

// 示例
function greet() {
    console.log(`hello, ${this.name}`)
}
const person = {
    name: '张三'
}
greet.myCall(person) // 输出 hell，张三
```

## 实现 apply
```js
Function.prototype.myApply = function(context, argsArray) {
  // 如果 context 为 null 或 undefined，则设置为全局对象（浏览器环境下为 window）
  context = context || globalThis;

  // 将当前调用 myApply 方法的函数设置为传入的函数
  context.fn = this;

  // 执行函数并传入参数数组
  const result = context.fn(...argsArray);

  // 删除临时添加的函数属性
  delete context.fn;

  return result;
};

// 示例
function greet(greeting, punctuation) {
  console.log(`${greeting}, ${this.name}${punctuation}`);
}

const person = {
  name: 'Alice'
};

greet.myApply(person, ['Hello', '!']);  // 输出：Hello, Alice!

```

## 手写实现 bind
`Function.prototype.bind()` 方法创建一个新函数，当被调用时，它的 `this` 值设置为提供的值，并在调用新函数时，将给定的参数列表作为原始函数的参数序列的前几个参数。

以下是 `bind` 方法的手动实现：

```javascript
Function.prototype.myBind = function(context, ...boundArgs) {
  const originalFunction = this;

  return function(...args) {
    // 当作为构造函数使用时，this 指向新实例，原始函数应绑定到新实例
    if (this instanceof originalFunction) {
      return new originalFunction(...boundArgs, ...args);
    } else {
      // 正常调用，将原始函数的 this 指向传入的 context
      return originalFunction.apply(context, [...boundArgs, ...args]);
    }
  };
};

// 示例
function greet(greeting, punctuation) {
  console.log(`${greeting}, ${this.name}${punctuation}`);
}

const person = {
  name: 'Alice'
};

const boundGreet = greet.myBind(person, 'Hello');
boundGreet('!');  // 输出：Hello, Alice!
```

在这个实现中，`myBind` 方法返回一个新函数。当新函数被调用时，它首先检查是否是通过 `new` 关键字调用的（即 `this instanceof originalFunction`）。如果是这样，它会将原始函数绑定到新的实例上，否则，它会将原始函数的 `this` 设置为传入的 `context`，并传入预先绑定的参数以及新的参数。



## 手写实现一个 bind 方法 第二种

```js
Function.prototype.myApply = function(context, ...args) {
	context = context || window
	context.fn = this
	const crrentVal = context.fn(...args)
	delete context.fn
	return context
}

Function.prototype.myBind = function(context, ...args) {
	const instanceFun = this
	
	return function (...val) {
		instanceFun.myApply(context, val.concat(args))
	}
}



function getName(e) {
	console.log(`我的名字叫：${this.name},自定义参数：${e}`)
}

const fun1 = getName.myBind({name: '张三'})
fun1('传递') // 输出结果： 我的名字叫：张三,自定义参数：传递
```