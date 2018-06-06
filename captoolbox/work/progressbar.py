from time import sleep
import sys

n = 21
for i in range(n):
    sys.stdout.write('\r')

    # Process code goes here 

    sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
    sys.stdout.flush()
    sleep(0.25)
