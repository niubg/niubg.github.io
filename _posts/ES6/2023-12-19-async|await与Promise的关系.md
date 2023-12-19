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
        console.log(res)
    } catch (error) {
        // try...catch 相当于 Promise catch
        console.log(error)  // error异常
    }
}

demo()

```
