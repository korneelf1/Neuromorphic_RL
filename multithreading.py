import threading

# Define a function that represents the work to be done by each thread
def worker(thread_id):
    print(f"Thread {thread_id} is starting")
    # Perform some work here
    print(f"Thread {thread_id} is ending")

def main():
    num_threads = 5
    threads = []

    # Create and start multiple threads
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All threads have finished")

if __name__ == "__main__":
    main()

import threading

# Define a function that represents the work to be done by each thread
def worker(thread_id):
    print(f"Thread {thread_id} is starting")
    # Perform some work here
    print(f"Thread {thread_id} is ending")

def main():
    num_threads = 5
    threads = []

    # Create and start multiple threads
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All threads have finished")

if __name__ == "__main__":
    main()
