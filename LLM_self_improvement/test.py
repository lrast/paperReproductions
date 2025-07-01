import multiprocessing as mp
import time


def worker(queue, worker_id):
    while True:
        item = queue.get()  # Blocks until item is available
        if item is None:
            print(f"Worker {worker_id} shutting down.")
            break  # Shutdown signal
        print(f"Worker {worker_id} processing item: {item}")
        # Simulate processing
        time.sleep(1)


class ProcessQueue:
    def __init__(self, num_workers=2):
        self.queue = mp.Queue()
        self.workers = []
        for i in range(num_workers):
            p = mp.Process(target=worker, args=(self.queue, i))
            p.start()
            self.workers.append(p)

    def add_task(self, item):
        self.queue.put(item)

    def shutdown(self):
        for _ in self.workers:
            self.queue.put(None)  # Send shutdown signal to each worker
        for p in self.workers:
            p.join()


if __name__ == '__main__':
    pq = ProcessQueue(num_workers=3)

    # Add some tasks
    for i in range(10):
        pq.add_task(f"Task-{i}")

    # Allow some time for processing
    time.sleep(5)

    pq.shutdown()
