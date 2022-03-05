from os import system
from multiprocessing import Process
from multiprocessing.connection import wait

def invoke():
    print("invokee BORN")
    system("python3 crasher.py")
    print("invokee DIED")

def main():
    print("main BORN")
    p = Process(target=invoke)
    p.start()
    wait([p.sentinel])
    p.join()
    print("main DIED")




if __name__ == "__main__":
    main()