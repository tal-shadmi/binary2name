#include <stdio.h>
#include <stdbool.h>

const char* f(int x, bool b) {
    if (b) {
        x += 3;
    }
    else {
        x -= 1;
    }
    x *= 2;
    if (x < 5) {
        return "GREEN";
    } else {
        return "RED";
    }
}

void main() {
    const char* res = f(8, true);
    printf(res);
}