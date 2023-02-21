import ray
import signal
import traceback
import asyncio
import threading
import time
import sys

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    print("Signal timeout, printing stack")
    traceback.print_stack(frame)
    raise TimeoutError('Timed out')

def async_timeout(seconds, coro):
    loop = asyncio.get_event_loop()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        result = loop.run_until_complete(coro)
    except TimeoutError:
        result = None

    signal.alarm(0)

    return result



print("Calling ray.init")

# Print pstack in another thread every 5 seconds
def print_pstack_thread():
    while(True):
        # 获取所有线程的栈信息
        stacks = sys._current_frames()
        # 遍历所有线程并打印栈信息
        for thread_id, stack in stacks.items():
            print(f"Thread ID: {thread_id}")
            traceback.print_stack(stack)
            print("....")
        print()
        print("... Sleep for 5 seconds ...")
        print()
        time.sleep(5)

t = threading.Thread(target=print_pstack_thread)
t.start()

ray.init(
    object_store_memory=100 * 1024 * 1024,
    _temp_dir="/host/tmp/ray",
)

def wait_cmd_to_pstack():
    import signal
    import traceback
    def print_pstack(sig, frame):
        print("Got signal: ", sig)
        traceback.print_stack(frame)
    signal.signal(signal.SIGQUIT, print_pstack) # 注册 SIGQUIT，在 Ctrl + \ 时触发

@ray.remote
def foo(x):
    return ray.put(f"Result data is: {x + 1}")

@ray.remote
class Owner:
    def warmup(self):
        return ray.get(ray.put("warmup"))

@ray.remote
class Borrower:
    def borrow(self, obj_ref):
        self.obj = ray.get(obj_ref)
        return self.obj

print("Calling ray.get")
value = ray.get(ray.get(foo.remote(1)))
print(value)
assert value == "Result data is: 2"
owner = Owner.remote()
# wait_cmd_to_pstack()
# obj = async_timeout(100, ray.get(owner.warmup.remote()))
for idx in range(20):
    owner = Owner.remote()
    obj = ray.get(owner.warmup.remote())
    print(f"idx {idx}: {obj}")