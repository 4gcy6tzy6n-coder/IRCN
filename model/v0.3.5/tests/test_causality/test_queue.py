import heapq
from itertools import count

def test_equal_time_events_follow_insertion_sequence():
    seq=count(); q=[]
    for node in [3,1,2]: heapq.heappush(q,(1.0,next(seq),node))
    assert [heapq.heappop(q)[2] for _ in range(3)]==[3,1,2]
