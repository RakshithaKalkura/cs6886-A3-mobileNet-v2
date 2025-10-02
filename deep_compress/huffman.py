"""
Simple Huffman coding utilities for final packaging. Not optimized — used to compute empirical bits.
"""
from collections import Counter
import heapq


class HuffmanNode:
def __init__(self, freq, sym=None, left=None, right=None):
self.freq = freq; self.sym = sym; self.left = left; self.right = right
def __lt__(self, other):
return self.freq < other.freq


def build_huffman_codes(symbols):
freq = Counter(symbols)
heap = [HuffmanNode(f, s) for s,f in freq.items()]
heapq.heapify(heap)
if len(heap)==1:
return {heap[0].sym: '0'}
while len(heap)>1:
a = heapq.heappop(heap); b = heapq.heappop(heap)
heapq.heappush(heap, HuffmanNode(a.freq + b.freq, None, a, b))
root = heap[0]
codes = {}
def walk(node, prefix=''):
if node.sym is not None:
codes[node.sym] = prefix
else:
walk(node.left, prefix+'0'); walk(node.right, prefix+'1')
walk(root, '')
return codes