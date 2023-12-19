---
title: 自定义一个useLocation
date: 2023-12-19 17:25:00 +0800
categories: [vue]
tags: [vue]
description: vue自定义自定义一个useLocation
---

## 自定义useLocation

模拟获取一个地理位置

```js
import {reactive, onMounted, toRefs } from 'vue';

// 模拟异步获取
function getLocation(fail) {
    debugger
    return new Promise(resolve => {
        setTimeout(() => {
            if (fail) {
                resolve({errno: 1, msg: fail})
            } else {
                resolve({errno: 0, data: { x: 100, y: 200}})
            }
        }, 1000)
    })
}

function useLocation() {
    debugger
    const info = reactive({
        isLoadingRef: true,
        data: {},
        errMsg: ''
    })
    
    onMounted(async () => {
        const res = await getLocation()
        debugger
        if (res.errno === 0) {
            info.data = res.data
        } else {
            info.errMsg = res.msg
        }
        info.isLoadingRef = false
    })
    return toRefs(info)
}

export default useLocation

```

引用如下

```vue
<template>
    
    <div>{{ isLoadingRef }}:{{ data }}:{{ errMsg }}</div>
</template>

<script >
import useLocation from 'useLocation.js'

export default {
    // 在选项组件中使用Composition API
    setup() {
        const { isLoadingRef, data, errMsg } =  useLocation()
        // const useLocation = useLocation()
        return { isLoadingRef, data, errMsg }
    },

    mounted(){
        
    }
}
</script>
```
