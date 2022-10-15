```toc
```

笔记来源：[Introduction · C++并发编程(中文版) (jb51.net)](http://shouce.jb51.net/cpp_concurrency_in_action/)

# 线程管理
## 1. 线程管理的基础
### 1. 启动新线程 
1. 启动新线程，即构造 std::thread 对象
~~~c++
void do_some_work();
std::thread my_thread(do_some_work);
~~~

2. 可调用对象进行构造
~~~c++
class background_task
{
public:
	void operator()() const
	{
		do_something();
		do_something_else();
	}
};
background_task f;
std::thread my_thread(f);

// 不能传入临时变量
// 如
std::thread my_thread(background_task()); // 编译器会认为是定义了一个函数，而不是创建了一个线程
// 使用如下方式创建
std::thread my_thread((background_task()));
std::thread my_thread{background_task()};
// 亦或者是lambda表达式
std::thread my_thread([]() {
	do_something();
	do_something_else();
});
~~~

3. 注意点
	1. 要在thread对象销毁前决定该线程是join() 还是 detach()，否则会运行std::terminate()(析构函数调用)，程序终止

### 2. 等待线程完成 
1. join()
	- 在线程结束前，会阻塞main()线程
2. detach()
	- 与主线程分离，执行互不影响

### 3. 特殊情况下的等待
1. 异常处理中的join()
	~~~c++
	try {
		do_something();
	}
	catch(...) {
		my_thread.join();
		throw ...;
	}
	my_thread.join();
	~~~
2. 使用RAII来等待线程完成
	~~~c++
	class thread_guard
	{
	  std::thread& t;
	public:
	  explicit thread_guard(std::thread& t_):
	    t(t_)
	  {}
	  ~thread_guard()
	  {
	    if(t.joinable()) // 1
	    {
	      t.join();      // 2
	    }
	  }
	  // 禁止拷贝构造和赋值拷贝
	  thread_guard(thread_guard const&)=delete;   // 3
	  thread_guard& operator=(thread_guard const&)=delete;
	};


	struct func
	{
	  int& i;
	  func(int& i_) : i(i_) {}
	  void operator() ()
	  {
	    for (unsigned j=0 ; j<1000000 ; ++j)
	    {
	      do_something(i);           // 1. 潜在访问隐患：悬空引用
	    }
	  }
	};


	int main() {
		int some_local_state=0;
		func my_func(some_local_state);
		std::thread t(my_func);
		thread_guard g(t);
		do_something_in_current_thread();
	}
	
	~~~

### 4. 后台运行线程
1. detach()
~~~c++
std::thread t(do_background_work);
t.detach();  // 线程t的执行，与主线程无关，主线程结束后，t可以继续执行
assert(!t.joinable());  // t不可join()
~~~

2. 例子
~~~c++
// 分离线程去处理其他文档
void edit_document(std::string const& filename)
{
  open_document_and_display_gui(filename);
  while(!done_editing())
  {
    user_command cmd=get_user_input();
    if(cmd.type==open_new_document)
    {
      std::string const new_name=get_filename_from_user();
      std::thread t(edit_document,new_name);  // 1，带参数的线程创建
      t.detach();  // 2
    }
    else
    {
       process_user_input(cmd);
    }
  }
}
~~~


## 2. 向线程传递参数
### 1. 传递参数时尽量不要依赖隐式转换，要显式转换参数
~~~c++
void f(int i,std::string const& s);

void oops(int some_param)
{
  char buffer[1024]; // 1
  sprintf(buffer, "%i",some_param);
  std::thread t(f,3,buffer); // 此处依赖隐式装换
  std::thread t(f, 3, string(buffer));  // 最好使用显式类型转换
  t.detach();
}
~~~

### 2. 没有使用 std::ref 时，thread的构造函数只会拷贝线程函数的参数，无论参数类型是否为引用，所以需要传递引用参数时，最好使用 std::ref
~~~c++
// 使用上面的例子
std::thread t(f, 3, std::ref(string(buffer)));
~~~

### 3. 传递类的成员函数给线程
~~~c++
class X
{
public:
	void do_some_thing();
};
X my_x;
std::thread t(&X::do_some_thing(), &my_x);  // 实际上运行my_x的do_some_thing

// 传递参数给类的成员函数
class X
{
public:
	void do_some_thing(int);
};
X my_x;
int num(4);
// 提供的参数可以移动，但是不能拷贝
// 详情见[2.2 向线程函数传递参数 · C++并发编程(中文版) (jb51.net)](http://shouce.jb51.net/cpp_concurrency_in_action/content/chapter2/2.2-chinese.html)
std::thread t(&X::do_some_thing, &my_x, num);
~~~




## 3. 转移线程所有权
### 1. 转移所有权
转移所有权之前要注意等式左边的 `std::thread`对象没有与任何执行线程关联，即
不能通过赋一个新值给`std::thread`对象的方式来"丢弃"一个线程
~~~c++
void some_function();
void some_other_function();
std::thread t1(some_function);            // 1
std::thread t2=std::move(t1);            // 2
t1=std::thread(some_other_function);    // 3
std::thread t3;                            // 4
t3=std::move(t2);                        // 5
t1=std::move(t3);                        // 6 赋值操作将使程序崩溃
~~~



## 4. 运行时决定线程数量
~~~c++
// 多线程累加
#include <thread>
#include <algorithm>
#include <numeric>
#include <vector>
#include <functional>
#include <iostream>
#include <ctime>

clock_t Begin, End;
double cost;

using namespace std;
using ull = unsigned long long;

// 计算块
template<typename Iterator, typename T>
struct accumulate_block
{
    void operator() (Iterator first, Iterator last, T &result) {
        result = accumulate(first, last, result);
    }
};

// 并行累加
template<typename Iterator, typename T>
T parallel_accumulate(Iterator first, Iterator last, T init) {
    ull const length = distance(first, last)
    if (length == 0)
        return init;
    // 每个线程累加的数量
    unsigned int const min_per_thread = 25;
    // 最大线程数
    unsigned int const max_threads = (length + min_per_thread - 1) / min_per_thread;
    // 机器的线程数
    unsigned int const hardware_threads = thread::hardware_concurrency();  
    // 实际用到的线程数
    unsigned int const num_threads = min(hardware_threads ? hardware_threads : 2, max_threads);

    // 计算块的数量
    unsigned int const block_size = length / num_threads;
    // 创建线程数组以及结果存储数组
    vector<T> results(num_threads);
    vector<thread> threads(num_threads-1); // 留一个给主线程

	// 开始累加
    Iterator block_start = first;
    for (unsigned int i = 0; i < (num_threads - 1); ++i) {
        Iterator block_end = block_start;
        advance(block_end, block_size); // 将迭代器block_end向后移动block_size个位置
  
        threads[i] = thread(accumulate_block<Iterator, T>(), block_start, block_end, ref(results[i]));
        block_start = block_end;
    }
    // 计算尾部剩下元素的累加（元素数量大于block_size）
    accumulate(block_start, last, results[num_threads-1]);
    for_each(threads.begin(), threads.end(), mem_fn(&thread::join));

    return accumulate(results.begin(), results.end(), init);
}


using type = unsigned long long;
void test(type size) {
    cout << "size = " << size << " ";
    vector<type> v (size);
    for (type i = 0; i < v.size(); ++i) {
        v[i] = i;
    }
    // 原生accumulate
    Begin = clock();
    volatile type r1 = accumulate(v.begin(), v.end(), 0);
    End = clock();
    cout << End - Begin << " ";

    // 并行计算
    Begin = clock();
    volatile type r2 = parallel_accumulate<vector<type>::iterator, type>(v.begin(), v.end(), 0);
    End = clock();
    cout << End - Begin << endl;

}
~~~



## 5. 识别线程
~~~c++
thread::id tid;
this_thread::get_id();
~~~







# 线程间共享数据
## 1. 共享数据带来的问题

### 1. 条件竞争
1. 不变量遭到破坏，产生条件竞争；比如对同一个双向链表进行结点删除

### 2. 避免恶性条件竞争
1. 采用数据保护机制，比如互斥量
2. 对数据结构和不变了进行设计，即无锁编程
3. 使用事务方式处理数据结构的更新，比如数据库中的事务--软件事务内存


## 2. 使用互斥量保护共享数据

### 1. C++中使用互斥量（mutex头文件）
1. eg：使用互斥量保护列表
~~~c++
#include <list>
#include <mutex>
#include <algorithm>

std::list<int> some_list;    // 1
std::mutex some_mutex;    // 2

void add_to_list(int new_value)
{
  // 对mutex 的一种RAII封装
  std::lock_guard<std::mutex> guard(some_mutex);    // 3
  some_list.push_back(new_value);
}

bool list_contains(int value_to_find)
{
  std::lock_guard<std::mutex> guard(some_mutex);    // 4
  return std::find(some_list.begin(),some_list.end(),value_to_find) != some_list.end();
}
~~~
2. eg: 无意中传递了保护数据的引用
~~~c++
class some_data
{
  int a;
  std::string b;
public:
  void do_something();
};

class data_wrapper
{
private:
  some_data data;
  std::mutex m;
public:
  template<typename Function>
  void process_data(Function func)
  {
    std::lock_guard<std::mutex> l(m);
    func(data);    // 1 传递“保护”数据给用户函数
  }
};

some_data* unprotected;

void malicious_function(some_data& protected_data)
{
  unprotected=&protected_data;
}

data_wrapper x;
void foo()
{
  x.process_data(malicious_function);    // 2 传递一个恶意函数
  unprotected->do_something();    // 3 在无保护的情况下访问保护数据
}
~~~
**切勿将受保护数据的指针或引用传递到互斥锁作用域之外，无论是函数返回值，还是存储在外部可见内存，亦或是以参数的形式传递到用户提供的函数中去。**

3. eg：线程安全堆栈
~~~c++
#include <exception>
#include <stack>
#include <mutex>
#include <memory>
  
using namespace std;

struct empty_stack : exception
{
    const char* what() const throw() {
        return "empty stack!";
    }
};

template <typename T>
class threadsafe_stack
{
private:
    stack<T> data;
    mutable mutex m;    
public:
    threadsafe_stack() : data(stack<T>()) {  }
    threadsafe_stack(const threadsafe_stack* other) {
        lock_guard<mutex> g(other.m);
        data = other.data;
    }
    threadsafe_stack& operator=(const threadsafe_stack&) = delete;

    void push(T new_value) {
        lock_guard<mutex> lock(m);
        data.push(new_value);
    }
  
    shared_ptr<T> pop() {
        lock_guard<mutex> lock(m);
        if (data.empty()) throw empty_stack();
        shared_ptr<T> const res(make_shared<T>(data.top()));
        data.pop();
        return res;
    }
  
    void pop(T& value) {
        lock_guard<mutex> lock(m);
        if (data.empty()) throw empty_stack();
        value = data.top();
        data.pop();
    }

    bool empty() const {
        lock_guard<mutex> lock(m);
        return data.empty();
    }
};
~~~

### 2. 死锁
1. 通常出现在有多个互斥量的时候，线程A锁上了互斥量X，等待互斥量Y；线程B占有了互斥量Y，等待互斥量X。这时候就发生了死锁。
2. 解决方案
	1. 按照固定的顺序给互斥量上锁
		1. C++中使用`std::lock()`，同时锁住多个互斥量或者都不锁
	2. 避免嵌套锁，一个线程只获得一个锁
	3. 避免在持有锁的时候调用用户代码
	4. 使用固定顺序获得锁，比如链表删除中
	5. 使用锁的层次结构
		1. 当代码试图对一个互斥量上锁，在该层锁已被低层持有时，上锁是不允许的。你可以在运行时对其进行检查，通过分配层数到每个互斥量上，以及记录被每个线程上锁的互斥量。锁住层级大的才能锁小的，小的不能锁大的
		2. 代码演示
	~~~c++
	// 层级锁的设计
	```
	class hierarchical_mutex
	{
	  std::mutex internal_mutex;
	
	  unsigned long const hierarchy_value;
	  unsigned long previous_hierarchy_value;
	
	  static thread_local unsigned long this_thread_hierarchy_value;  // 1
	
	  void check_for_hierarchy_violation()
	  {
	    if(this_thread_hierarchy_value <= hierarchy_value)  // 2
	    {
	      throw std::logic_error(“mutex hierarchy violated”);
	    }
	  }
	
	  void update_hierarchy_value()
	  {
	    previous_hierarchy_value=this_thread_hierarchy_value;  // 3
	    this_thread_hierarchy_value=hierarchy_value;
	  }
	
	public:
	  explicit hierarchical_mutex(unsigned long value):
	      hierarchy_value(value),
	      previous_hierarchy_value(0)
	  {}
	
	  void lock()
	  {
	    check_for_hierarchy_violation();
	    internal_mutex.lock();  // 4
	    update_hierarchy_value();  // 5
	  }
	
	  void unlock()
	  {
	    this_thread_hierarchy_value=previous_hierarchy_value;  // 6
	    internal_mutex.unlock();
	  }
	
	  bool try_lock()
	  {
	    check_for_hierarchy_violation();
	    if(!internal_mutex.try_lock())  // 7
	      return false;
	    update_hierarchy_value();
	    return true;
	  }
	};
	thread_local unsigned long
	     hierarchical_mutex::this_thread_hierarchy_value(ULONG_MAX);  // 8
	~~~

### 3. 使用std::unique_lock（更加灵活的互斥量）
1. 锁定策略
	1. `std::defer_lock_t` 、 `std::try_to_lock_t` 和 `std::adopt_lock_t` 是用于为 [std::lock_guard](https://zh.cppreference.com/w/cpp/thread/lock_guard "cpp/thread/lock guard") 、 std::scoped_lock 、 [std::unique_lock](https://zh.cppreference.com/w/cpp/thread/unique_lock "cpp/thread/unique lock") 和 [std::shared_lock](https://zh.cppreference.com/w/cpp/thread/shared_lock "cpp/thread/shared lock") 指定锁定策略的空类标签类型。
	2. `std::defer_lock` 、 `std::try_to_lock` 和 `std::adopt_lock` 分别是空结构体标签类型 [std::defer_lock_t](https://zh.cppreference.com/w/cpp/thread/lock_tag_t "cpp/thread/lock tag t") 、 [std::try_to_lock_t](https://zh.cppreference.com/w/cpp/thread/lock_tag_t "cpp/thread/lock tag t") 和 [std::adopt_lock_t](https://zh.cppreference.com/w/cpp/thread/lock_tag_t "cpp/thread/lock tag t") 的实例。
	3. 类型                                          效果
		`defer_lock_t`             不获得互斥的所有权
		`try_to_lock_t`           尝试获得互斥的所有权而不阻塞
		`adopt_lock_t`             假设调用方线程已拥有互斥的所有权

2. 特性
	1. 比`lock_guard`更加灵活，可以主动释放锁(unlock())

3. 使用示例
~~~C++
#include <mutex>
std mutex m1, m2;
{
	std::lock(m1, m2);
	std::lock_guard<std::mutex> lock1(m1, std::adopt_lock);
	std::lock_guard<std::mutex> lock2(m2, std::adopt_lock);
	operator_data();
}
// 等价于
{
	std::unique_lock<std::mutex> ulock1(m1, std::defer_lock);
	std::unique_lock<std::mutex> ulock2(m2, std::defer_lock);
	std::lock(ulock1, ulock2);
	operator_data();
}
~~~

### 4. 不同域中互斥量的所有权的传递

### 5. 锁的粒度

## 3. 保护共享数据的替代设施
### 1. 保护共享数据的初始化过程
有时一个共享源的初始化会消耗很多资源，所以需要`延迟初始化`：对数据进行操作前，先判断是否初始化了，未初始化就先初始化再进行操作
1. 延迟初始化
~~~c++
// 单线程下没有任何问题
std::shared_ptr<some_resource> resource_ptr;
void foo()
{
  if(!resource_ptr)
  {
    resource_ptr.reset(new some_resource);  // 1
  }
  resource_ptr->do_something();
}
~~~

2. 多线程下的初始化
~~~c++
// 线程安全
std::shared_ptr<some_resource> resource_ptr;
std::mutex resource_mutex;

void foo()
{
  std::unique_lock<std::mutex> lk(resource_mutex);  // 所有线程在此序列化 
  if(!resource_ptr)
  {
    resource_ptr.reset(new some_resource);  // 只有初始化过程需要保护 
  }
  lk.unlock();
  resource_ptr->do_something();
}
// 双重检查锁模式
// 有可能线程A执行到1，线程B已经执行到3了，然后还没有初始化完，线程A已经执行4了，这就会出现错误的结果
void undefined_behaviour_with_double_checked_locking()
{
  if(!resource_ptr)  // 1
  {
    std::lock_guard<std::mutex> lk(resource_mutex);
    if(!resource_ptr)  // 2
    {
      resource_ptr.reset(new some_resource);  // 3
    }
  }
  resource_ptr->do_something();  // 4
}
~~~

3. 引入`std::once_flag` `std::call_once`
比起锁住互斥量，并显式的检查指针，每个线程只需要使用`std::call_once`，在`std::call_once`的结束时，就能安全的知道指针已经被其他的线程初始化了。
~~~c++
// 还可以用于类的非静态成员的延迟初始化
std::shared_ptr<some_resource> resource_ptr;
std::once_flag resource_flag;  // 1

void init_resource()
{
  resource_ptr.reset(new some_resource);
}

void foo()
{
  std::call_once(resource_flag,init_resource);  // 可以完整的进行一次初始化
  resource_ptr->do_something();
}
~~~

4. 类的静态成员初始化
~~~c++
// 在支持C++11的编译器上是线程安全的
class my_class;
my_class& get_my_class_instance()
{
  static my_class instance;  // 线程安全的初始化过程
  return instance;
}
~~~

### 2. 保护很少更新的数据结构
场景：DNS缓存的更新
方案：读写锁`std::shared_mutex`(C++17)（没有互斥量那么严格）-- 只有写时需要上锁
~~~c++
// 示例，std::map 表示DNS缓存表
#include <map>
#include <string>
#include <mutex>

class dns_entry;

class dns_cache
{
  std::map<std::string,dns_entry> entries;
  mutable std::shared_mutex entry_mutex;
public:
  dns_entry find_entry(std::string const& domain) const
  {
    std::shared_lock<std::shared_mutex> lk(entry_mutex);  // 1 读锁
    std::map<std::string,dns_entry>::const_iterator const it=
       entries.find(domain);
    return (it==entries.end())?dns_entry():it->second;
  }
  void update_or_add_entry(std::string const& domain,
                           dns_entry const& dns_details)
  {
    std::lock_guard<std::shared_mutex> lk(entry_mutex);  // 2 写锁
    entries[domain]=dns_details;
  }
};
~~~

### 3. 嵌套锁
#### 1. 什么是嵌套锁(递归锁)
`C++`标准库提供了`std::recursive_mutex`类。其功能与`std::mutex`类似，除了你可以从同一线程的单个实例上获取多个锁。互斥量锁住其他线程前，你必须释放你拥有的所有锁，所以当你调用lock()三次时，你也必须调用unlock()三次。正确使用`std::lock_guard<std::recursive_mutex>`和`std::unique_lock<std::recursice_mutex>`可以帮你处理这些问题。

#### 2. 使用场景
嵌套锁一般用在可并发访问的类上，所以其拥互斥量保护其成员数据。
每个公共成员函数都会对互斥量上锁，然后完成对应的功能，之后再解锁互斥量。
不过，有时成员函数会调用另一个成员函数，这种情况下，第二个成员函数也会试图锁住互斥量，这就会导致未定义行为的发生。“变通的”解决方案会将互斥量转为嵌套锁，第二个成员函数就能成功的进行上锁，并且函数能继续执行。

#### 3. 评价
1. 这样的使用方式是不推荐的，因为其过于草率，并且不合理。
2. 当锁被持有时，对应类的不变量通常正在被修改。这意味着，当不变量正在改变的时候，第二个成员函数还需要继续执行。
3. 一个比较好的方式是，从中提取出一个函数作为类的私有成员，并且让其他成员函数都对其进行调用，这个私有成员函数不会对互斥量进行上锁(在调用前必须获得锁)。







# 同步并发操作
`C++`标准库提供了一些工具可用于同步操作，形式上表现为 _条件变量_ (condition variables)和 _期望_(futures)。
## 1. 等待一个事件或者其他条件
一个线程等待另一个线程完成任务。
### 解决方案
1. 类似轮询
不断检查共享数据标志(一个互斥量)，比较消耗cpu资源
2. 定时休眠
~~~c++
bool flag;
std::mutex m;

void wait_for_flag()
{
  std::unique_lock<std::mutex> lk(m);
  while(!flag)
  {
    lk.unlock();  // 1 解锁互斥量
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 2 休眠100ms
    lk.lock();   // 3 再锁互斥量
  }
}
~~~
休眠期间，另外的线程就有机会获得锁并设置标志。但是这个休眠时间很难设置，太短了等于没有休眠；太长了可能会让等待线程醒来，并且不适用与实时性要求较高的场景中
3. 使用条件变量 condition variables

### 条件变量
#### 介绍
`std::condition_variable`和`std::condition_variable_any`。
1. 这两个实现都包含在`<condition_variable>`头文件的声明中。两者都需要与一个互斥量一起才能工作(互斥量是为了同步)
2. 前者仅限于与`std::mutex`一起工作，而后者可以和任何满足最低标准的互斥量一起工作，从而加上了__any_的后缀。
3. 因为`std::condition_variable_any`更加通用，这就可能从体积、性能，以及系统资源的使用方面产生额外的开销，所以`std::condition_variable`一般作为首选的类型，当对灵活性有硬性要求时，我们才会去考虑`std::condition_variable_any`。

#### 使用
1. 例子，使用`std::condition_variable`处理数据等待
~~~c++
std::mutex mut;
std::queue<data_chunk> data_queue;  // 1
std::condition_variable data_cond;

// 准备数据
void data_preparation_thread()
{
  while(more_data_to_prepare())
  {
    data_chunk const data=prepare_data();
    std::lock_guard<std::mutex> lk(mut);
    data_queue.push(data);  // 2 完成数据的入队
    data_cond.notify_one();  // 3 通知等待线程
  }
}

// 处理数据
void data_processing_thread()
{
  while(true)
  {
    std::unique_lock<std::mutex> lk(mut);  // 4
    data_cond.wait(
         lk,[]{return !data_queue.empty();});  // 5
    data_chunk data=data_queue.front();
    data_queue.pop();
    lk.unlock();  // 6
    process(data);
    if(is_last_chunk(data))
      break;
  }
}
// 5的说明：当wait()中的第二个参数为ture时，继续往下执行并且保留锁；为false时，将会解锁并且阻塞在这。所以使用unique_lock()；当阻塞时需要被与该互斥量绑定的条件变量调用noticy_one()或者调用notice_all()唤醒，并且重新获得锁，并且再次对第二个参数进行判断
~~~

## 2. 使用期望等待一次性事件

### 1. 带返回值的后台任务
1. 使用`std::async`启动一个异步任务，`std::async`会返回一个`std::future`对象，这个对象会持有该任务返回的结果，调用`std::future`的get()方法可以得到返回值
~~~c++
#include <future>
#include <iostream>

int find_the_answer_to_ltuae();
void do_other_stuff();
int main()
{
  std::future<int> the_answer=std::async(find_the_answer_to_ltuae);
  do_other_stuff();
  std::cout<<"The answer is "<<the_answer.get()<<std::endl;
}
~~~

2. 使用`std::async`向函数传入参数
- 传参方式与`std::thread`一样
- 在默认情况下，“期望”是否进行等待取决于`std::async`是否启动一个线程，或是否有任务正在进行同步。
	- `std::launch::async`会使用新线程执行，默认使用这个
	- `std::launch::deferred::async`在wait()或get()调用时执行，不会启动新线程

### 2. 任务与期望
除了`std::async`，还可以使用`std::packaged_task<>` 和`std::promise<>`类型模板，前者比后者具有更高的抽象

#### 1. `std::packaged_task<>`
模板参数为函数签名，初始化是要传入对应函数签名的函数，然后通过 thread 来执行，通过 get_future() 可以获得 future 对象，通过get() 可以得到package_task对象中函数执行的结果

#### 2. `std::promise<>`
通过`promise`的 _get_future()_ 得到与`promise`绑定的`future`，通过`future`的 _get()_ 可以得到`promise`的 _set_value()_ 的值

#### 3. `std::future`
普通的 future 不可拷贝，只能移动，而 future 只能调用一次 get()，所以只有一个线程实例可以获得其中的值。为了让多个线程获得其值，可以将其转为 shared_future，通过future.share()。并且share()后，原来普通的future就不能用了

















