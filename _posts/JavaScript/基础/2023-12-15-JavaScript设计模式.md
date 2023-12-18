---
title: JavaScript设计模式
date: 2023-12-15 18:12:10 +0800
categories: [JavaScript]
tags: [JavaScript]
description: JavaScript设计模式:工厂模式、单例模式、适配器模式、装饰者模式、观察者模式、代理模式、策略模式、模板方法模式
---

## 工厂模式

工厂函数是一种设计模式，它是一个返回对象或函数的函数。其目的是用于创建和初始化对象，使得代码更具可读性、可维护性和可扩展性。

在 JavaScript 中，工厂函数通常用于创建对象，允许你在创建对象的过程中执行一些初始化逻辑。这样可以封装对象的创建细节，同时提供一个清晰的接口。

一个简单的工厂函数的示例：

```javascript
function createPerson(name, age) {
  return {
    name: name,
    age: age,
    sayHello: function() {
      console.log("Hello, my name is " + this.name + " and I'm " + this.age + " years old.");
    }
  };
}

// 使用工厂函数创建对象
const person1 = createPerson("Alice", 25);
const person2 = createPerson("Bob", 30);

person1.sayHello();  // 输出：Hello, my name is Alice and I'm 25 years old.
person2.sayHello();  // 输出：Hello, my name is Bob and I'm 30 years old.
```

在上面的例子中，`createPerson` 是一个工厂函数，用于创建包含姓名、年龄和一个打招呼方法的人物对象。

在 Vue.js 中，`createApp` 也是一种工厂函数，用于创建 Vue.js 的应用实例。这个函数的目的是为了提供一个统一的入口，使得应用实例的创建和配置更加清晰和易于管理。


## 单例模式

单例模式是一种设计模式，其目的是确保一个类只有一个实例，并提供一个全局访问点。这意味着当你多次请求该类的实例时，始终会返回相同的实例。单例模式通常用于管理全局状态、资源共享或限制某个类只能有一个实例。

在实现单例模式时，一般需要考虑以下几个要素：

1. **私有构造函数（Private Constructor）：** 单例类的构造函数应该是私有的，以防止直接通过 `new` 操作符来创建多个实例。

2. **静态成员变量（Static Member Variable）：** 单例类应该有一个静态成员变量来保存实例，以确保全局唯一性。

3. **静态方法（Static Method）：** 提供一个静态方法来获取或创建唯一的实例。这个方法会负责检查是否已经存在实例，如果存在则返回已有的实例，否则创建一个新实例并返回。

示例实现单例模式：

```javascript
class Singleton {
  // 静态变量用于保存唯一实例
  static instance = null;

  // 私有构造函数，防止通过 new 操作符直接创建实例
  constructor() {
    if (!Singleton.instance) {
      Singleton.instance = this;
    }

    return Singleton.instance;
  }

  // 静态方法用于获取唯一实例
  static getInstance() {
    if (!Singleton.instance) {
      Singleton.instance = new Singleton();
    }

    return Singleton.instance;
  }
}

// 使用单例模式创建实例
const instance1 = new Singleton();
const instance2 = Singleton.getInstance();

console.log(instance1 === instance2); // 输出 true，说明两者是同一个实例
```

在上述例子中，`Singleton` 类的构造函数是私有的，通过 `getInstance` 方法获取实例，保证了全局只有一个实例存在。在实际应用中，单例模式可以用于管理全局状态、配置信息、数据库连接等需要唯一实例的场景。



`上述代码单例代码很容易被篡改instance实例状态，所以可以使用es2022版私有变量‘#’`，代码如下

```js
class Singleton {
  // 私有变量，用于保存唯一实例
  static #instance = null;

  // 私有构造函数，防止通过 new 操作符直接创建实例
  constructor() {
    if (!Singleton.#instance) {
      Singleton.#instance = this;
    }

    return Singleton.#instance;
  }

  // 静态方法用于获取唯一实例
  static getInstance() {
    if (!Singleton.#instance) {
      Singleton.#instance = new Singleton();
    }

    return Singleton.#instance;
  }
}

// 使用单例模式创建实例
const instance1 = new Singleton();
const instance2 = Singleton.getInstance();

console.log(instance1 === instance2); // 输出 true，说明两者是同一个实例

```


在这个修改版本中，`#instance`使用了私有字段（private field）的语法，它使得`instance`成为类的私有属性，外部无法直接访问。这样就增加了防篡改的难度。请注意，私有字段目前是JavaScript的ECMAScript标准的一部分，但在某些旧的环境中可能不被支持。

如果你的环境不支持私有字段，另一种方法是使用闭包来隐藏`instance`：

```js
class Singleton {
  // 私有变量，用于保存唯一实例
  static #instance = null;

  // 私有构造函数，防止通过 new 操作符直接创建实例
  constructor() {
    if (!Singleton.#instance) {
      Singleton.#instance = this;
    }

    return Singleton.#instance;
  }

  // 静态方法用于获取唯一实例
  static getInstance() {
    if (!Singleton.#instance) {
      Singleton.#instance = new Singleton();
    }

    return Singleton.#instance;
  }
}

// 使用单例模式创建实例
const createSingleton = (() => {
  let instance = null;

  return function () {
    if (!instance) {
      instance = new Singleton();
    }

    return instance;
  };
})();

const instance1 = createSingleton();
const instance2 = Singleton.getInstance();

console.log(instance1 === instance2); // 输出 true，说明两者是同一个实例

```



## 适配器模式

适配器模式（Adapter Pattern）是一种结构设计模式，用于将一个接口转换成另一个客户端希望的接口。适配器模式允许原本由于接口不兼容而不能一起工作的类能够一起工作。

适配器模式通常涉及到一个被称为适配器的类，该类包装了一个或多个不兼容接口的对象，并使它们与客户端代码协同工作。

主要的参与者包括：

1. **目标接口（Target Interface）：** 客户端期望的接口，适配器会实现这个接口，以便客户端可以与适配器一起工作。

2. **适配器（Adapter）：** 实现了目标接口，并包装了一个或多个不兼容接口的对象。适配器负责将客户端的请求转发给被适配的对象。

3. **被适配者（Adaptee）：** 拥有不兼容接口的类，被适配器包装起来以满足客户端的期望。

适配器模式的经典示例是连接器适配器，例如，将欧洲的插头适配到北美的插座。在软件开发中，适配器模式可以用于整合新旧系统、整合第三方库等场景。

示例适配器模式：

```javascript
// 目标接口
class Target {
  request() {
    console.log("Target: The default target's behavior.");
  }
}

// 被适配者
class Adaptee {
  specificRequest() {
    console.log("Adaptee: The specific request's behavior.");
  }
}

// 适配器
class Adapter extends Target {
  constructor(adaptee) {
    super();
    this.adaptee = adaptee;
  }

  request() {
    this.adaptee.specificRequest();
    console.log("Adapter: Adapted behavior.");
  }
}

// 客户端代码
function clientCode(target) {
  target.request();
}

// 在不改变客户端代码的情况下，适配器使得新旧类能够一起工作
const adaptee = new Adaptee();
const adapter = new Adapter(adaptee);

console.log("Client: Using the Adaptee class:");
clientCode(adaptee);

console.log("\nClient: Using the Adapter class:");
clientCode(adapter);
```

`Target` 是目标接口，`Adaptee` 是被适配者，而 `Adapter` 是适配器。适配器继承了目标接口，并包装了被适配者的对象，以便客户端可以调用目标接口的方法。适配器的 `request` 方法实现了适配逻辑，将调用转发给了被适配者的 `specificRequest` 方法。



## 装饰者模式

装饰者模式（Decorator Pattern）是一种结构设计模式，它允许你通过将对象放入包含行为的特殊封装类中来为原始对象动态添加新的行为。这种模式的设计原则是封闭修改，开放扩展，即允许在不改变现有代码的情况下扩展对象的功能。

在装饰者模式中，通常会有一个基础组件（Concrete Component），即具体的原始对象，以及一个或多个装饰者（Decorator），它们包装了基础组件，并可以在运行时动态地添加额外的功能。

主要的参与者包括：

1、**组件接口（Component Interface）：** 定义了基础组件和装饰者的共同接口，确保它们可以互相替换。

2、**具体组件（Concrete Component）：** 实现了组件接口，是被装饰的原始对象。

3、**装饰者（Decorator）：** 实现了组件接口，并包装了一个具体组件，可以在运行时动态地为其添加额外的功能。

装饰者模式：

```js
    // 组件接口
    class Coffee {
    cost() {
        return 5; // 基础咖啡价格
    }
    }

    // 具体组件
    class SimpleCoffee extends Coffee {
    cost() {
        return super.cost();
    }
    }

    // 装饰者
    class MilkDecorator extends Coffee {
    constructor(coffee) {
        super();
        this.coffee = coffee;
    }

    cost() {
        return this.coffee.cost() + 2; // 添加牛奶的价格
    }
    }

    // 装饰者
    class SugarDecorator extends Coffee {
    constructor(coffee) {
        super();
        this.coffee = coffee;
    }

    cost() {
        return this.coffee.cost() + 1; // 添加糖的价格
    }
    }

    // 客户端代码
    const myCoffee = new SimpleCoffee();
    console.log("Cost of simple coffee:", myCoffee.cost());

    const milkCoffee = new MilkDecorator(myCoffee);
    console.log("Cost of milk coffee:", milkCoffee.cost());

    const sugarMilkCoffee = new SugarDecorator(milkCoffee);
    console.log("Cost of sugar milk coffee:", sugarMilkCoffee.cost());

```

`Coffee` 是组件接口，`SimpleCoffee` 是具体组件。`MilkDecorator` 和 `SugarDecorator` 是装饰者，它们都继承自 `Coffee` 并包装了其他 `Coffee` 对象。这使得我们可以在运行时动态地组合咖啡并添加额外的功能（牛奶、糖等），而不需要修改原始的 `SimpleCoffee` 类。这符合装饰者模式的核心思想。



## 观察者模式

观察者模式（Observer Pattern）是一种行为设计模式，它定义了一种一对多的依赖关系，当一个对象的状态发生变化时，所有依赖于它的对象都得到通知并自动更新。在观察者模式中，有两个主要角色：观察者（Observer）和被观察者（Subject）。

1、**观察者（Observer）：** 定义了一个更新接口，使得在被观察者状态改变时能够接收到通知并进行相应的处理。

2、**被观察者（Subject）：** 负责维护一组观察者对象，并在自身状态发生变化时通知观察者。

观察者模式的主要优点是实现了对象间的松耦合，被观察者和观察者之间彼此独立，可以灵活地添加或删除观察者而不影响系统的其他部分。

观察者模式：

```javascript
// 观察者接口
class Observer {
  update(message) {
    // 在子类中实现
  }
}

// 被观察者
class Subject {
  constructor() {
    this.observers = [];
  }

  addObserver(observer) {
    this.observers.push(observer);
  }

  removeObserver(observer) {
    this.observers = this.observers.filter(obs => obs !== observer);
  }

  notifyObservers(message) {
    this.observers.forEach(observer => observer.update(message));
  }
}

// 具体观察者
class ConcreteObserver extends Observer {
  constructor(name) {
    super();
    this.name = name;
  }

  update(message) {
    console.log(`${this.name} received message: ${message}`);
  }
}

// 客户端代码
const subject = new Subject();

const observer1 = new ConcreteObserver('Observer 1');
const observer2 = new ConcreteObserver('Observer 2');

subject.addObserver(observer1);
subject.addObserver(observer2);

subject.notifyObservers('Hello, observers!');
```

`Observer` 是观察者接口，`Subject` 是被观察者。`ConcreteObserver` 是具体观察者。当 `Subject` 的状态发生变化时，它会通知所有注册的观察者，观察者会收到通知并执行相应的操作。这种模式通常用于实现事件处理系统、UI界面更新等场景。



## 代理模式

代理模式（Proxy Pattern）是一种结构设计模式，它提供了一个代理对象，以控制对其他对象的访问。代理对象通常充当其他对象的接口，允许在访问这些对象时进行间接控制。

代理模式的常见用途包括：

1. **远程代理（Remote Proxy）：** 控制对远程对象的访问，例如在网络上的对象。
  
2. **虚拟代理（Virtual Proxy）：** 控制对昂贵对象的访问，例如延迟加载对象的创建或加载大型图像的过程。

3. **保护代理（Protection Proxy）：** 控制对敏感对象的访问，例如检查用户权限。

4. **缓存代理（Cache Proxy）：** 提供对对象的临时存储，避免重复计算或加载。

在代理模式中，主要的参与者包括：

- **抽象主题（Subject）：** 定义了代理和真实主题的共同接口，以便代理可以替代真实主题。

- **真实主题（Real Subject）：** 定义了代理所代表的真实对象。

- **代理（Proxy）：** 包含对真实主题的引用，并提供与真实主题相同的接口，从而使得代理可以替代真实主题。

以下是一个简单的 JavaScript 示例，演示了虚拟代理的概念：

```javascript
// 抽象主题
class Image {
  display() {
    // 在子类中实现
  }
}

// 真实主题
class RealImage extends Image {
  constructor(filename) {
    super();
    this.filename = filename;
    this.loadFromDisk();
  }

  loadFromDisk() {
    console.log(`Loading image: ${this.filename}`);
  }

  display() {
    console.log(`Displaying image: ${this.filename}`);
  }
}

// 代理
class ProxyImage extends Image {
  constructor(filename) {
    super();
    this.filename = filename;
    this.realImage = null;
  }

  display() {
    if (!this.realImage) {
      this.realImage = new RealImage(this.filename);
    }
    this.realImage.display();
  }
}

// 客户端代码
const image1 = new ProxyImage('image1.jpg');
const image2 = new ProxyImage('image2.jpg');

image1.display(); // 第一次加载
image1.display(); // 直接显示，无需再次加载

image2.display(); // 第一次加载
```

`Image` 是抽象主题，`RealImage` 是真实主题，`ProxyImage` 是代理。当创建 `ProxyImage` 时，并不立即加载真实主题 `RealImage`，而是在调用 `display` 方法时才加载。这样就实现了延迟加载的效果，节省了资源。这是一个虚拟代理的应用。


## 策略模式

策略模式（Strategy Pattern）是一种行为设计模式，它定义了一系列算法，将每个算法封装并使它们可以互相替换。策略模式使得算法的变化独立于使用算法的客户端。

在策略模式中，有三个主要的参与者：

1. **环境（Context）：** 维护一个对策略对象的引用，负责将客户端的请求委派给具体的策略。

2. **策略（Strategy）：** 定义了一个算法族的接口，所有具体策略都实现了这个接口。

3. **具体策略（Concrete Strategy）：** 实现了策略接口的具体算法。

策略模式的主要目标是让算法的变化独立于使用算法的客户端，从而实现了算法的自由切换和扩展。

以下是一个简单的 JavaScript 示例，演示了策略模式的概念：

```javascript
// 策略接口
class PaymentStrategy {
  pay(amount) {
    // 在子类中实现
  }
}

// 具体策略1
class CreditCardPayment extends PaymentStrategy {
  pay(amount) {
    console.log(`Paid $${amount} via credit card.`);
  }
}

// 具体策略2
class PayPalPayment extends PaymentStrategy {
  pay(amount) {
    console.log(`Paid $${amount} via PayPal.`);
  }
}

// 环境
class ShoppingCart {
  constructor(paymentStrategy) {
    this.paymentStrategy = paymentStrategy;
  }

  checkout(amount) {
    this.paymentStrategy.pay(amount);
  }
}

// 客户端代码
const creditCardPayment = new CreditCardPayment();
const payPalPayment = new PayPalPayment();

const cart1 = new ShoppingCart(creditCardPayment);
cart1.checkout(100);

const cart2 = new ShoppingCart(payPalPayment);
cart2.checkout(50);
```

在上述例子中，`PaymentStrategy` 是策略接口，`CreditCardPayment` 和 `PayPalPayment` 是具体策略。`ShoppingCart` 是环境，根据传入的不同策略对象，在调用 `checkout` 方法时会使用相应的支付策略。这样，客户端可以灵活地选择支付策略，而不需要修改 `ShoppingCart` 类的代码。这就是策略模式的核心思想。


## 模板方法模式

模板方法模式（Template Method Pattern）是一种行为设计模式，它定义了一个算法的骨架，但将一些步骤的具体实现延迟到子类。模板方法使得子类可以在不改变算法结构的情况下重新定义算法的某些步骤。

在模板方法模式中，有两个主要的参与者：

1. **模板类（Abstract Class）：** 定义了算法的骨架，其中包含一个或多个抽象方法，这些方法由子类实现。

2. **具体子类（Concrete Class）：** 实现了抽象类中定义的抽象方法，完成了算法的具体步骤。

以下是一个简单的 JavaScript 示例，演示了模板方法模式的概念：

```javascript
// 模板类
class CoffeeTemplate {
  makeCoffee() {
    this.boilWater();
    this.brewCoffeeGrounds();
    this.pourInCup();
    this.addCondiments();
    console.log("Coffee is ready!");
  }

  boilWater() {
    console.log("Boiling water");
  }

  brewCoffeeGrounds() {
    console.log("Brewing coffee grounds");
  }

  pourInCup() {
    console.log("Pouring coffee into cup");
  }

  // 抽象方法，由子类实现
  addCondiments() {
    throw new Error("This method must be overridden by subclass");
  }
}

// 具体子类1
class CoffeeWithHook extends CoffeeTemplate {
  addCondiments() {
    console.log("Adding sugar and milk");
  }
}

// 具体子类2
class BlackCoffee extends CoffeeTemplate {
  addCondiments() {
    console.log("Adding nothing");
  }
}

// 客户端代码
const coffeeWithHook = new CoffeeWithHook();
coffeeWithHook.makeCoffee();

const blackCoffee = new BlackCoffee();
blackCoffee.makeCoffee();
```

在上述例子中，`CoffeeTemplate` 是模板类，定义了制作咖啡的算法骨架。`CoffeeWithHook` 和 `BlackCoffee` 是具体子类，分别实现了抽象方法 `addCondiments`，从而提供了不同的调料方式。通过模板方法 `makeCoffee`，客户端可以调用相同的算法骨架，但根据具体子类的不同实现，得到不同口味的咖啡。