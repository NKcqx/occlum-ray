import ray


ray.init(
    object_store_memory=100 * 1024 * 1024,
    _temp_dir="/host/tmp/ray",
)

@ray.remote
def foo(x):
    return ray.put(f"Result data is: {x + 1}")

value = ray.get(ray.get(foo.remote(1)))
print(value)
assert value == "Result data is: 2"
