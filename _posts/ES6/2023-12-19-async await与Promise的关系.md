---
title: async/await与Promise的关系
date: 2023-12-19 22:29:00 +0800
categories: [ECMAScript]
tags: [ECMAScript]
description: sync/await与Promise的关系ES6
---

## async/await与Promise的关系

- 执行async函数，返回的是promise对象
- await相当于Promise的then
- try...catch可捕获异常，代替了Promise的catch

```js
 async function foo() {
    return 10 // 相当于 return Promise.resolve(10)
    // return Promise.resolve(10)
 }

 const res = foo() // 执行async函数，返回的是Promise对象
 // 可以使用then获取数据结果
 res.then(data => {
    console.log(data) // 10
 })
```


```js
async function demo {
    const p1 = Promise.resolve(100)
    // await 相当于 Promise 里的 then，所以这里的data就等同于 then 方法中的响应参数data
    const data = await p1
    console.log(data)   // 100
}

// 执行
demo()
```

``` js
async function demo() {
    const p1 = Promise.reject('error异常')  // rejected 状态
    try {
        const res = await p1
        // await 相当于 then，但上面执行的是 Promise.reject 所以下面的打印res(then)返回结果是不会执行的，直接就走了catch 
        console.log(res)
    } catch (error) {
        // try...catch 相当于 Promise catch
        console.log(error)  // error异常
    }
}

demo()

```


## 异步的本质

- async/await是解决异步回调的终极武器
- JS还是单线程，还有异步，还是基于 event loop
- async/await 只是一个语法糖

```js

async function async1 () {
    console.log(1)
    await async2()
    /**
     * await 的后面执行的代码，都可以看作是 callback 里的内容，既异步
     * 类似 event loop， setTimeout()
     * setTimeout(function() { console.log(2) })
     * Promise.resolve().then(() => { console.log(2) })
     */
    console.log(2)
}

async function async2() {
    console.log(3)
}

console.log(4)
async1()
console.log(5)

// 输出结果: 4 1 3 5 2


```
