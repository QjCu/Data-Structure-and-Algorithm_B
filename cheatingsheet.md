# TREE
## 遍历
### 前序（根->左->右）
```python
def PreOrder(root):
    if root is not null:
        visit(root)
        PreOrder(root.left)
        PreOrder(root.right)
```
### 中序（左->根->右）
```python
def InOrder(root):
    if root is not null:
        InOrder(root.left)
        visit(root)
        InOrder(root.right)
```
### 后序（左->右->根）
```python
def PostOrder(root):
    if root is not null:
        PostOrder(root.left)
        PostOrder(root.right)
        visit(root)
```
## Huffman Code
```python
import heapq
import copy


class node:
    def __init__(self, w, v):
        self.w = w
        self.v = v
        self.l = None
        self.r = None

    def __lt__(self, other):
        if self.w == other.w:
            return self.v < other.v
        return self.w < other.w


def decode(n, string):
    ans = ""
    for i in string:
        if i == "1":
            n = n.r
        else:
            n = n.l
        if not n.l and not n.r:
            ans += n.v
            n = copy.deepcopy(root)

    return ans


def encode(n, chars):
    dic = {}
    ans = ""

    def add(n, string):
        if not n.l and not n.r:
            dic[n.v] = string
        else:
            add(n.l, string + "0")
            add(n.r, string + "1")
    add(n, "")

    for i in chars:
        ans += dic[i]
    return ans


lst = []
for _ in range(int(input())):
    v, w = input().split()
    w = int(w)
    n = node(w, v)
    lst.append(n)
heapq.heapify(lst)

while len(lst) > 1:
    small = heapq.heappop(lst)
    big = heapq.heappop(lst)
    n = node(small.w + big.w, None)
    n.l = small
    n.r = big
    heapq.heappush(lst, n)

root = heapq.heappop(lst)
n = copy.deepcopy(root)
```
## 二叉搜索树

**左 < 根 < 右**

**用中序遍历即位顺序**

## Trie
```python
class Trie:
    def __init__(self):
        self.child = {}

    def add(self, num):
        cur = self.child
        for i in num:
            if i not in cur:
                cur[i] = {}
            cur = cur[i]

    def search(self, i):
        cur = self.child
        for j in i:
            if j not in cur:
                return False
            cur = cur[j]
        return True


for _ in range(int(input())):
    n = int(input())
    nums = [input() for __ in range(n)]
    nums.sort(reverse=True)
    dictree = Trie()
    bo = True
    for i in nums:
        if dictree.search(i):
            bo = False
            break
        dictree.add(i)
    if bo:
        print("YES")
    else:
        print("NO")
```

# GRAPH
## 遍历
### DFS
```python
def dfs(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    print(node, end=' ')
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
```
### BFS
```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        print(node, end=' ')
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```
## 最短路
### Dijkstra
>**思路**
>1. 初始化距离表，起点距离为 0，其他点为 ∞；
>2. 使用优先队列（通常是最小堆）维护待处理节点；
>3. 每次从队列取出距离最小的节点，更新其邻居的距离；
> - 重复直到队列为空。
```python
import heapq

def dijkstra(graph, start):
    # graph 格式：节点 -> list of (邻居, 权重)
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]  # (距离, 节点)

    while pq:
        current_dist, node = heapq.heappop(pq)

        # One way
        if visited[node]:
            continue

        # Another way
        if current_dist > distances[node]:
            continue  # 已有更短路径

        for neighbor, weight in graph[node]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances
```
### SPFA
>**思路：**
>1. 初始化距离数组，起点距离设为 0，其它为 ∞；
>2. 起点入队；
>3. 队列非空时：
> - 出队一个顶点 u；
> - 遍历 u 的所有邻居 v，尝试松弛边 (u, v)；
> - 如果更新了 dist[v]，且 v 不在队列中，则将 v 入队。
```python
from collections import deque

def spfa(graph, start):
    # graph 格式：节点 -> list of (邻居, 权重)
    dist = {node: float('inf') for node in graph}
    in_queue = {node: False for node in graph}
    dist[start] = 0

    queue = deque([start])
    in_queue[start] = True

    while queue:
        u = queue.popleft()
        in_queue[u] = False

        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True

    return dist
```
## DAG
### Kahn
```python
from collections import deque

def topo_sort_kahn(graph):
    indegree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            indegree[v] += 1

    queue = deque([u for u in graph if indegree[u] == 0])
    result = []

    while queue:
        u = queue.popleft()
        result.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    if len(result) == len(graph):
        return result
    else:
        raise ValueError("图中有环，无法拓扑排序")
```
## MST
### Kruskal
>**思路**
>1. 将所有边按权重排序；
>2. 初始化并查集，每个节点自成集合；
>3. 遍历排序后的边：
>   - 若两个端点不属于同一集合，则加入该边，并合并集合；
>4. 直到所有节点被连接。
```python
def find(self, x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def kruskal(n, edges):
    """
    n: 节点数，节点编号假设为 0~n-1
    edges: [(u, v, weight), ...]
    """
    edges.sort(key=lambda x: x[2])  # 按权重排序

    mst = []
    total_weight = 0
    for u, v, w in edges:
        U = find(u)
        V = find(v)
        if U != V:
            parent[u] = V
            mst.append((u, v, w))
            total_weight += w

    return mst, total_weight
```
### Prim（适合临接表）
>**思路**
>1. 初始化一个集合 mst_set 存储已加入节点，起点入集合；
>2. 使用最小堆存储可扩展的边；
>3. 每次选择权重最小的未选节点加入集合，并更新边；
>4. 直到所有节点都加入。
```python
import heapq

def prim(graph, start=0):
    """
    graph: dict，节点 -> list of (邻居, 权重)
    start: 起点
    """
    mst_set = set()
    min_heap = [(0, start)]  # (权重, 节点)
    total_weight = 0
    mst_edges = []

    while min_heap and len(mst_set) < len(graph):
        weight, u = heapq.heappop(min_heap)
        if u in mst_set:
            continue
        mst_set.add(u)
        total_weight += weight
        # 第一次起点弹出，weight=0，不加入边
        if weight != 0:
            mst_edges.append((u, weight))

        for v, w in graph[u]:
            if v not in mst_set:
                heapq.heappush(min_heap, (w, v))

    if len(mst_set) != len(graph):
        raise ValueError("图不连通，无法构成生成树")
    return mst_edges, total_weight
```
## Union Set
```python
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  # 递归路径压缩
    return parent[x]

while True:
    X = find(x)
    Y = find(y)
    parent[x] = Y

```
# Sort
## Merge
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    # 分
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    # 合并
    return merge(left, right)
```
## Qsort
```python
def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi)
        quick_sort(arr, pi + 1, high)

def partition(arr, low, high):
    pivot = arr[low]  # 用第一个元素作为 pivot
    left = low - 1
    right = high + 1

    while True:
        # 从左向右找第一个大于等于 pivot 的元素
        left += 1
        while arr[left] < pivot:
            left += 1

        # 从右向左找第一个小于等于 pivot 的元素
        right -= 1
        while arr[right] > pivot:
            right -= 1

        if left >= right:
            return right

        # 交换 arr[left] 和 arr[right]
        arr[left], arr[right] = arr[right], arr[left]
```
# Other
## 调度场算法（中置->后置）
```python
def shunting_yard(expression):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output = []
    stack = []
    tokens = expression.split()

    for token in tokens:
        if token.isnumeric():
            output.append(token)
        elif token in precedence:
            while (stack and stack[-1] != '(' and
                   precedence.get(stack[-1], 0) >= precedence[token]):
                output.append(stack.pop())
            stack.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # Remove '('

    while stack:
        output.append(stack.pop())

    return ' '.join(output)
```
## KMP
```python
for i in range(1, n):
    while l > 0 and s[i] != s[l]:
        l = next[l - 1]
    if s[i] == s[l]:
        l += 1
    next[i] = l
    
for i in range(2, n + 1):
    c = i - next[i - 1]
    if not i % c and (i // c) > 1:
        print(i, i // c)
```